"""
Loss functions for GTCRN-BC training.

HybridLoss combines:
1. Compressed complex spectral loss (real + imag with power-law compression)
2. Compressed magnitude loss  
3. SI-SNR in time domain

This is the same loss from the original GTCRN training, unchanged.
It works well for BC-aided enhancement because:
- The spectral losses handle frequency-domain reconstruction
- SI-SNR ensures time-domain signal quality
- Power-law compression (0.3) emphasizes quiet components
"""
import torch
import torch.nn as nn


class HybridLoss(nn.Module):
    """
    Multi-objective loss for speech enhancement.
    
    Components:
        - Compressed complex loss: MSE on power-compressed real/imag parts
        - Compressed magnitude loss: MSE on mag^0.3
        - SI-SNR: Scale-invariant signal-to-noise ratio
    
    Input:
        pred_stft: (B, F, T, 2) — predicted STFT [real, imag]
        true_stft: (B, F, T, 2) — target clean STFT [real, imag]
    """
    def __init__(self, nfft=512, hop=256, compress_exp=0.3):
        super().__init__()
        self.nfft = nfft
        self.hop = hop
        self.compress_exp = compress_exp

    def forward(self, pred_stft, true_stft):
        device = pred_stft.device

        pred_real = pred_stft[:, :, :, 0]
        pred_imag = pred_stft[:, :, :, 1]
        true_real = true_stft[:, :, :, 0]
        true_imag = true_stft[:, :, :, 1]

        pred_mag = torch.sqrt(pred_real ** 2 + pred_imag ** 2 + 1e-12)
        true_mag = torch.sqrt(true_real ** 2 + true_imag ** 2 + 1e-12)

        # Power-law compressed complex components
        c = 1.0 - self.compress_exp  # 0.7
        pred_real_c = pred_real / (pred_mag ** c)
        pred_imag_c = pred_imag / (pred_mag ** c)
        true_real_c = true_real / (true_mag ** c)
        true_imag_c = true_imag / (true_mag ** c)

        real_loss = nn.functional.mse_loss(pred_real_c, true_real_c)
        imag_loss = nn.functional.mse_loss(pred_imag_c, true_imag_c)
        mag_loss = nn.functional.mse_loss(pred_mag ** self.compress_exp,
                                           true_mag ** self.compress_exp)

        # SI-SNR in time domain
        window = torch.hann_window(self.nfft).pow(0.5).to(device)
        y_pred = torch.istft(pred_real + 1j * pred_imag,
                             self.nfft, self.hop, self.nfft, window=window)
        y_true = torch.istft(true_real + 1j * true_imag,
                             self.nfft, self.hop, self.nfft, window=window)

        # SI-SNR calculation
        y_true_proj = (torch.sum(y_true * y_pred, dim=-1, keepdim=True) * y_true
                       / (torch.sum(y_true ** 2, dim=-1, keepdim=True) + 1e-8))
        sisnr = -torch.log10(
            torch.norm(y_true_proj, dim=-1, keepdim=True) ** 2
            / (torch.norm(y_pred - y_true_proj, dim=-1, keepdim=True) ** 2 + 1e-8)
            + 1e-8
        ).mean()

        return 30 * (real_loss + imag_loss) + 70 * mag_loss + sisnr
