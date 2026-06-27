import torch
import torch.nn as nn
from util import _istft


class WavLMFeatureLoss(nn.Module):
    def __init__(self, bundle_name="WAVLM_BASE", layer=-1):
        super().__init__()
        try:
            import torchaudio
        except ImportError as exc:
            raise ImportError(
                "torchaudio is required for WavLM feature loss. "
                "Install torchaudio or disable use_wavlm_loss."
            ) from exc

        try:
            bundle = getattr(torchaudio.pipelines, bundle_name)
        except AttributeError as exc:
            raise ValueError(f"Unknown torchaudio WavLM bundle: {bundle_name}") from exc

        sample_rate = getattr(bundle, "sample_rate", None)
        if sample_rate is not None and sample_rate != 16000:
            raise ValueError(f"{bundle_name} expects {sample_rate} Hz audio, but training audio is 16000 Hz")

        self.encoder = bundle.get_model()
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad_(False)
        self.layer = layer

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def _select_layer(self, features):
        if not features:
            raise RuntimeError("WavLM returned no hidden states")
        return features[self.layer]

    @staticmethod
    def _masked_l1(pred, target, lengths):
        if lengths is None:
            return torch.nn.functional.l1_loss(pred, target)

        n_frames = pred.size(1)
        frame_ids = torch.arange(n_frames, device=pred.device).unsqueeze(0)
        mask = frame_ids < lengths.unsqueeze(1)
        mask = mask.to(dtype=pred.dtype).unsqueeze(-1)
        denom = mask.sum().clamp_min(1.0) * pred.size(-1)
        return (pred - target).abs().mul(mask).sum() / denom

    def forward(self, y_pred, y_true, lengths=None):
        feature_lengths = lengths.to(device=y_pred.device) if lengths is not None else None
        pred_features, pred_lengths = self.encoder.extract_features(y_pred, lengths=feature_lengths)
        with torch.no_grad():
            true_features, true_lengths = self.encoder.extract_features(y_true, lengths=feature_lengths)

        pred_layer = self._select_layer(pred_features)
        true_layer = self._select_layer(true_features).detach()
        layer_lengths = pred_lengths if pred_lengths is not None else true_lengths
        if layer_lengths is not None:
            layer_lengths = layer_lengths.to(device=pred_layer.device)
        return self._masked_l1(pred_layer, true_layer, layer_lengths)


class HybridLoss(nn.Module):
    def __init__(
            self,
            n_fft=512,
            hop_length=256,
            real_weight=30.0,
            imag_weight=30.0,
            mag_weight=70.0,
            sisnr_weight=1.0,
            use_wavlm_loss=False,
            wavlm_loss_weight=0.0,
            wavlm_bundle="WAVLM_BASE",
            wavlm_layer=-1
        ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.real_weight = real_weight
        self.imag_weight = imag_weight
        self.mag_weight = mag_weight
        self.sisnr_weight = sisnr_weight
        self.use_wavlm_loss = use_wavlm_loss
        self.wavlm_loss_weight = wavlm_loss_weight
        self.wavlm_loss = None
        if use_wavlm_loss:
            self.wavlm_loss = WavLMFeatureLoss(bundle_name=wavlm_bundle, layer=wavlm_layer)

    def _stft_mask(self, lengths, n_frames, device):
        frame_centers = torch.arange(n_frames, device=device) * self.hop_length
        return frame_centers.unsqueeze(0) < lengths.unsqueeze(1)

    @staticmethod
    def _masked_mse(pred, target, mask):
        mask = mask.to(dtype=pred.dtype)
        while mask.dim() < pred.dim():
            mask = mask.unsqueeze(1)
        denom = mask.sum().clamp_min(1.0) * pred.size(1)
        return (pred - target).pow(2).mul(mask).sum() / denom

    @staticmethod
    def _masked_sisnr(y_pred, y_true, lengths):
        n_samples = y_true.size(-1)
        sample_ids = torch.arange(n_samples, device=y_true.device)
        mask = (sample_ids.unsqueeze(0) < lengths.unsqueeze(1)).to(y_true.dtype)

        dot = (y_true * y_pred * mask).sum(dim=-1, keepdim=True)
        true_energy = y_true.pow(2).mul(mask).sum(dim=-1, keepdim=True) + 1e-8
        y_target = dot * y_true / true_energy

        target_pow = y_target.pow(2).mul(mask).sum(dim=-1, keepdim=True)
        noise_pow = (y_pred - y_target).pow(2).mul(mask).sum(dim=-1, keepdim=True)
        return -torch.log10(target_pow / (noise_pow + 1e-8) + 1e-8).mean()

    def forward(
            self,
            pred_stft,
            true_stft,
            lengths=None,
            return_components=False,
            return_grad_components=False
        ):
        device = pred_stft.device

        pred_stft_real, pred_stft_imag = pred_stft[:,:,:,0], pred_stft[:,:,:,1]
        true_stft_real, true_stft_imag = true_stft[:,:,:,0], true_stft[:,:,:,1]
        pred_mag = torch.sqrt(pred_stft_real**2 + pred_stft_imag**2 + 1e-12)
        true_mag = torch.sqrt(true_stft_real**2 + true_stft_imag**2 + 1e-12)
        
        pred_real_c = pred_stft_real / (pred_mag**(0.7))
        pred_imag_c = pred_stft_imag / (pred_mag**(0.7))
        true_real_c = true_stft_real / (true_mag**(0.7))
        true_imag_c = true_stft_imag / (true_mag**(0.7))

        if lengths is None:
            real_loss = nn.MSELoss()(pred_real_c, true_real_c)
            imag_loss = nn.MSELoss()(pred_imag_c, true_imag_c)
            mag_loss = nn.MSELoss()(pred_mag**(0.3), true_mag**(0.3))
            istft_length = 32000
        else:
            lengths = lengths.to(device=device)
            stft_mask = self._stft_mask(lengths, pred_stft.size(2), device)
            real_loss = self._masked_mse(pred_real_c, true_real_c, stft_mask)
            imag_loss = self._masked_mse(pred_imag_c, true_imag_c, stft_mask)
            mag_loss = self._masked_mse(pred_mag**(0.3), true_mag**(0.3), stft_mask)
            istft_length = int(lengths.max().item())

        y_pred = _istft(pred_stft, length=istft_length)
        # Yp = torch.complex(pred_stft_real, pred_stft_imag)
        # y_pred = torch.istft(Yp, 512, 256, 512, window=torch.hann_window(512).pow(0.5).to(device))
        
        y_true = _istft(true_stft, length=istft_length)
        # Yt = torch.complex(true_stft_real, true_stft_imag)
        # y_true = torch.istft(Yt, 512, 256, 512, window=torch.hann_window(512).pow(0.5).to(device))

        if lengths is None:
            y_true = torch.sum(y_true * y_pred, dim=-1, keepdim=True) * y_true / (torch.sum(torch.square(y_true),dim=-1,keepdim=True) + 1e-8)
            sisnr =  - torch.log10(torch.norm(y_true, dim=-1, keepdim=True)**2 / (torch.norm(y_pred - y_true, dim=-1, keepdim=True)**2+1e-8) + 1e-8).mean()
        else:
            sisnr = self._masked_sisnr(y_pred, y_true, lengths)

        weighted_real = self.real_weight * real_loss
        weighted_imag = self.imag_weight * imag_loss
        weighted_mag = self.mag_weight * mag_loss
        weighted_spectral = weighted_real + weighted_imag + weighted_mag
        weighted_sisnr = self.sisnr_weight * sisnr
        wavlm = None
        weighted_wavlm = None
        total = weighted_spectral + weighted_sisnr
        if self.wavlm_loss is not None and self.wavlm_loss_weight > 0:
            wavlm = self.wavlm_loss(y_pred, y_true, lengths)
            weighted_wavlm = self.wavlm_loss_weight * wavlm
            total = total + weighted_wavlm

        if not return_components and not return_grad_components:
            return total

        components = {
            "total": total.detach(),
            "real": real_loss.detach(),
            "imag": imag_loss.detach(),
            "mag": mag_loss.detach(),
            "sisnr": sisnr.detach(),
            "weighted_real": weighted_real.detach(),
            "weighted_imag": weighted_imag.detach(),
            "weighted_mag": weighted_mag.detach(),
            "weighted_sisnr": weighted_sisnr.detach(),
        }
        if wavlm is not None:
            components["wavlm"] = wavlm.detach()
            components["weighted_wavlm"] = weighted_wavlm.detach()

        if not return_grad_components:
            return total, components

        grad_components = {
            "total": total,
            "weighted_real": weighted_real,
            "weighted_imag": weighted_imag,
            "weighted_mag": weighted_mag,
            "weighted_spectral": weighted_spectral,
            "weighted_sisnr": weighted_sisnr,
        }
        if weighted_wavlm is not None:
            grad_components["weighted_wavlm"] = weighted_wavlm
        return total, components, grad_components


if __name__ == "__main__":
    loss_func = HybridLoss()

    pred_stft = torch.randn(1, 257, 63, 2)
    true_stft = torch.randn(1, 257, 63, 2)
    loss = loss_func(pred_stft, true_stft, torch.tensor([16000]))
    print(loss)
