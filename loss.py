import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridLoss(nn.Module):
    def __init__(self,
        nfft=512,
        hop_length=128,
        center=True,
        onesided=True,
        length=32000,
        compress_exp=0.3,
    ):
        super().__init__()
        self.nfft = nfft
        self.hop = hop_length
        self.center = center
        self.onesided = onesided
        self.length = length
        self.compress_exp = compress_exp

        # Match the window you used in torch.stft
        self.register_buffer("window", torch.hann_window(nfft))

    def forward(self, pred_stft, true_stft):
        pred_real, pred_imag = pred_stft[..., 0], pred_stft[..., 1]
        true_real, true_imag = true_stft[..., 0], true_stft[..., 1]

        pred_mag = torch.sqrt(pred_real**2 + pred_imag**2 + 1e-12)
        true_mag = torch.sqrt(true_real**2 + true_imag**2 + 1e-12)

        c = 1.0 - self.compress_exp
        pred_real_c = pred_real / (pred_mag ** c)
        pred_imag_c = pred_imag / (pred_mag ** c)
        true_real_c = true_real / (true_mag ** c)
        true_imag_c = true_imag / (true_mag ** c)

        real_loss = F.mse_loss(pred_real_c, true_real_c)
        imag_loss = F.mse_loss(pred_imag_c, true_imag_c)
        mag_loss  = F.mse_loss(pred_mag ** self.compress_exp, true_mag ** self.compress_exp)

        # ISTFT with the SAME window as STFT
        win = self.window.to(device=pred_real.device, dtype=pred_real.dtype)

        Yp = torch.complex(pred_real, pred_imag)
        Yt = torch.complex(true_real, true_imag)
        y_pred = torch.istft(
            Yp, n_fft=self.nfft, hop_length=self.hop, win_length=self.nfft,
            window=win, center=self.center, normalized=False,
            onesided=self.onesided, length=self.length
        )
        y_true = torch.istft(
            Yt, n_fft=self.nfft, hop_length=self.hop, win_length=self.nfft,
            window=win, center=self.center, normalized=False,
            onesided=self.onesided, length=self.length
        )

        # SI-SNR (your formula kept)
        y_true_proj = (torch.sum(y_true * y_pred, dim=-1, keepdim=True) * y_true
                       / (torch.sum(y_true ** 2, dim=-1, keepdim=True) + 1e-8))
        sisnr = -torch.log10(
            torch.norm(y_true_proj, dim=-1, keepdim=True) ** 2
            / (torch.norm(y_pred - y_true_proj, dim=-1, keepdim=True) ** 2 + 1e-8)
            + 1e-8
        ).mean()

        return 30 * (real_loss + imag_loss) + 70 * mag_loss + sisnr