"""
Inference for GTCRN-BC.

Usage from notebook:
    from inference import enhance_file, enhance_vibravox_sample, load_model
    
    model = load_model("checkpoints/best_model.pt")
    enhance_file(model, "noisy_ac.wav", "bc.wav", "enhanced.wav")
"""
import torch
import torchaudio
import numpy as np
import soundfile as sf
from pathlib import Path

from gtcrn_bc import GTCRN_BC, GTCRN


NFFT = 512
HOP = 256
SR = 16000


def load_model(checkpoint_path, device=None):
    """
    Load trained GTCRN-BC (or GTCRN) from checkpoint.
    
    Returns: model, config, device
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = ckpt.get("config", {})
    model_type = config.get("model_type", "gtcrn_bc")

    if model_type == "gtcrn_bc":
        model = GTCRN_BC()
    else:
        model = GTCRN()

    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()

    print(f"Loaded {model_type} from {checkpoint_path}")
    print(f"  Epoch: {ckpt.get('epoch', '?')} | Val loss: {ckpt.get('val_loss', '?'):.4f}")

    return model, config, device


def _load_wav(path, target_sr=SR):
    """Load a WAV file and resample to target_sr."""
    waveform, sr = torchaudio.load(path)
    waveform = waveform[0]  # mono
    if sr != target_sr:
        waveform = torchaudio.transforms.Resample(sr, target_sr)(waveform)
    return waveform


def _stft(waveform):
    """Compute STFT → (F, T, 2)"""
    window = torch.hann_window(NFFT).pow(0.5)
    return torch.stft(waveform, NFFT, HOP, NFFT, window=window, return_complex=False)


def _istft(spec):
    """Inverse STFT from (F, T, 2) → waveform"""
    window = torch.hann_window(NFFT).pow(0.5).to(spec.device)
    real = spec[..., 0]
    imag = spec[..., 1]
    return torch.istft(real + 1j * imag, NFFT, HOP, NFFT, window=window)


def enhance_file(model, ac_path, bc_path, output_path, device=None):
    """
    Enhance a pair of WAV files (noisy AC + BC) and save result.
    
    Args:
        model: loaded GTCRN_BC model
        ac_path: path to noisy air-conducted WAV
        bc_path: path to bone-conducted WAV
        output_path: where to save enhanced WAV
        device: torch device (auto-detect if None)
    """
    if device is None:
        device = next(model.parameters()).device

    ac = _load_wav(ac_path)
    bc = _load_wav(bc_path)

    # Align lengths
    min_len = min(ac.shape[-1], bc.shape[-1])
    ac = ac[:min_len]
    bc = bc[:min_len]

    ac_stft = _stft(ac).unsqueeze(0).to(device)   # (1, F, T, 2)
    bc_stft = _stft(bc).unsqueeze(0).to(device)    # (1, F, T, 2)

    with torch.no_grad():
        if isinstance(model, GTCRN_BC):
            enh_stft = model(ac_stft, bc_stft)
        else:
            enh_stft = model(ac_stft)

    enh_wav = _istft(enh_stft[0].cpu())
    sf.write(output_path, enh_wav.numpy(), SR)
    print(f"Enhanced audio saved to: {output_path}")

    return enh_wav


def enhance_waveforms(model, ac_waveform, bc_waveform, device=None):
    """
    Enhance raw waveform tensors (already at 16kHz).
    
    Args:
        model: loaded model
        ac_waveform: (N,) tensor — noisy AC
        bc_waveform: (N,) tensor — BC
        
    Returns:
        enhanced: (N,) tensor
    """
    if device is None:
        device = next(model.parameters()).device

    min_len = min(ac_waveform.shape[-1], bc_waveform.shape[-1])
    ac = ac_waveform[:min_len]
    bc = bc_waveform[:min_len]

    ac_stft = _stft(ac).unsqueeze(0).to(device)
    bc_stft = _stft(bc).unsqueeze(0).to(device)

    with torch.no_grad():
        if isinstance(model, GTCRN_BC):
            enh_stft = model(ac_stft, bc_stft)
        else:
            enh_stft = model(ac_stft)

    return _istft(enh_stft[0].cpu())


def enhance_vibravox_sample(
    model,
    split="test",
    sample_idx=0,
    bc_sensor="soft_in_ear_microphone",
    snr_db=5,
    output_dir="enhanced_samples",
    device=None,
):
    """
    Enhance a sample directly from Vibravox test set.
    
    Loads AC + BC from Vibravox, adds noise at given SNR, enhances, saves all versions.
    
    Returns: dict with paths to saved files and waveform tensors
    """
    from datasets import load_dataset

    if device is None:
        device = next(model.parameters()).device

    print(f"Loading Vibravox sample {sample_idx} from {split}...")
    ds = load_dataset("Cnam-LMSSC/vibravox", "speech_clean", split=split)
                    #    trust_remote_code=True)
    sample = ds[sample_idx]

    # Load audio
    resampler = torchaudio.transforms.Resample(48000, SR)

    ac_dict = sample["audio.headset_microphone"]
    bc_dict = sample[f"audio.{bc_sensor}"]

    ac_clean = resampler(torch.from_numpy(np.array(ac_dict["array"], dtype=np.float32)))
    bc_raw = resampler(torch.from_numpy(np.array(bc_dict["array"], dtype=np.float32)))

    # Align
    min_len = min(ac_clean.shape[-1], bc_raw.shape[-1])
    ac_clean = ac_clean[:min_len]
    bc_raw = bc_raw[:min_len]

    # Add noise
    signal_power = (ac_clean ** 2).mean()
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = torch.randn_like(ac_clean) * torch.sqrt(noise_power)
    ac_noisy = ac_clean + noise

    # Enhance
    enhanced = enhance_waveforms(model, ac_noisy, bc_raw, device)

    # Save
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    sf.write(out / "ac_clean.wav", ac_clean.numpy(), SR)
    sf.write(out / "ac_noisy.wav", ac_noisy.numpy(), SR)
    sf.write(out / "bc_raw.wav", bc_raw.numpy(), SR)
    sf.write(out / "enhanced.wav", enhanced.numpy(), SR)

    print(f"Saved to {output_dir}/:")
    print(f"  ac_clean.wav  — original clean AC")
    print(f"  ac_noisy.wav  — noisy AC (SNR={snr_db} dB)")
    print(f"  bc_raw.wav    — raw bone-conducted")
    print(f"  enhanced.wav  — enhanced output")

    return {
        "ac_clean": ac_clean,
        "ac_noisy": ac_noisy,
        "bc_raw": bc_raw,
        "enhanced": enhanced,
        "text": sample.get("raw_text", ""),
    }
