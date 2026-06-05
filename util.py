import torch
from tqdm.auto import tqdm
import math

DEFAULT_SPEC_CONFIG = {
    'n_fft': 512,
    'hop_length': 128,
    'win_length': 512,
    'center': True,
    'pad_mode': 'reflect',
    'onesided': True,
    'length': 32000
}

class spec_transformator():
    def __init__(self, spec_config={}):
        cfg = DEFAULT_SPEC_CONFIG
        cfg.update(spec_config)

        self.n_fft = cfg['n_fft']
        self.hop_length = cfg['hop_length']
        self.win_length = cfg['win_length']
        self.center = cfg['center']
        self.pad_mode = cfg['pad_mode']
        self.onesided = cfg['onesided']
        self.window = torch.hann_window(self.win_length)
        self.length = cfg['length']
    
    def stft(self, X):
        X_stft = torch.stft(
            X,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(device=X.device, dtype=X.dtype),
            center=self.center,
            pad_mode=self.pad_mode,
            normalized=False,
            onesided=self.onesided,
            return_complex=True,
        )
        # X_stft = torch.view_as_real(X_stft)
        return X_stft

    def istft(self, X):
        X_istft = torch.istft(
            X,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(device=X.device),
            center=self.center,
            normalized=False,
            onesided=self.onesided,
            length=self.length
        )
        return X_istft
    
def _istft(spec: torch.Tensor, length: int = 32000) -> torch.Tensor:
    """ 
        spec: (B, F, T, 2)
        out : (B, length)
    """
    device = spec.device
    spec = torch.complex(spec[...,0], spec[...,1]).to(device)
    out = torch.istft(
        spec,
        512,
        256,
        512,
        window=torch.hann_window(512).pow(0.5).to(device=device, dtype=spec.real.dtype),
        length=length
    )
    return out

def _stft(signal: torch.Tensor) -> torch.Tensor:
    """ 
        signal: (B, 1?, 32000)
        out   : (B, F, T)  complex
    """
    
    device = signal.device

    out = torch.stft(
        signal,
        512,
        256,
        512,
        torch.hann_window(512).pow(0.5).to(device),
        return_complex=True
    )
    return out

def make_pbar(iterable, total=None, desc=None):
    # Colab/TTY can be flaky; these settings are usually stable.
    return tqdm(
        iterable,
        total=total,
        desc=desc,
        dynamic_ncols=True,
        mininterval=0.2,
        maxinterval=1.0,
        smoothing=0.0,
        ascii=True,          # more robust in terminals
        leave=False,         # avoid accumulating bars
    )


def get_device(requested=None):
    """Auto-detect best device: CUDA > MPS > CPU."""
    if requested is not None:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def infinite_loader(dataloader):
    while True:
        for batch in dataloader:
            yield batch

def match_batch(noise: torch.Tensor, B: int) -> torch.Tensor:
    """Repeat/crop noise on batch dim to exactly B."""
    B1 = noise.size(0)
    if B1 == B:
        return noise
    reps = math.ceil(B / B1)
    return noise.repeat(reps, 1)[:B]

def random_crop_1d(x: torch.Tensor, T: int) -> torch.Tensor:
    """
    x: [B, L] -> return [B, T] by random per-sample crop.
    """
    B, L = x.shape
    if L < T:
        # pad if ever needed
        pad = T - L
        x = torch.nn.functional.pad(x, (0, pad))
        L = T
    max_start = L - T
    starts = torch.randint(0, max_start + 1, (B,), device=x.device)
    idx = starts[:, None] + torch.arange(T, device=x.device)[None, :]
    return x.gather(1, idx)

EPS = 1e-8

def add_noise_snr(signal: torch.Tensor, noise: torch.Tensor, snr_db: torch.Tensor) -> torch.Tensor:
    """
    signal: [B, T]
    noise:  [B, T]
    snr_db: [B]  (power SNR in dB)
    """
    # power per sample
    sig_pow = signal.pow(2).mean(dim=1, keepdim=True)          # [B, 1]
    noi_pow = noise.pow(2).mean(dim=1, keepdim=True) + EPS     # [B, 1]

    snr_lin = (10.0 ** (snr_db / 10.0)).view(-1, 1)            # [B, 1]
    target_noi_pow = sig_pow / (snr_lin + EPS)                 # [B, 1]
    scale = torch.sqrt(target_noi_pow / noi_pow)               # [B, 1]

    return signal + noise * scale

# def rebatch_noise(loader, n_samples=128, new_bs=8):
#     collected = []
#     total = 0

#     # 1) collect first n_samples
#     for batch in loader:
#         B = batch['noise'].size(0)

#         if total + B >= n_samples:
#             collected.append(batch['noise'][: n_samples - total])
#             break
#         else:
#             collected.append(batch['noise'])
#             total += B

#     # 2) concatenate into single tensor
#     data = torch.cat(collected, dim=0)   # [128, ...]

#     # 3) split into batches of new_bs
#     rebatches = torch.split(data, new_bs)  # tuple of tensors

#     return rebatches

def rebatch_signal(loader, n_samples=128, new_bs=8):
    collected_ac = []
    collected_bc = []
    total = 0

    # 1) collect first n_samples
    for batch in loader:
        B = batch['ac_clean'].size(0)

        if total + B >= n_samples:
            collected_ac.append(batch['ac_clean'][: n_samples - total])
            collected_bc.append(batch['bc'][: n_samples - total])
            break
        else:
            collected_ac.append(batch['ac_clean'])
            collected_bc.append(batch['bc'])
            total += B

    # 2) concatenate into single tensor
    data_ac = torch.cat(collected_ac, dim=0)   # [128, ...]
    data_bc = torch.cat(collected_bc, dim=0)   # [128, ...]

    # 3) split into batches of new_bs
    rebatches_ac = torch.split(data_ac, new_bs)  # tuple of tensors
    rebatches_bc = torch.split(data_bc, new_bs)  # tuple of tensors

    return [{'ac_clean': ac, 'bc': bc} for ac, bc in zip(rebatches_ac, rebatches_bc)]
