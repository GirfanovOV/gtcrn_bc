import torch
import torch.nn as nn
import torch.nn.functional as F
from util import spec_transformator

class HybridLoss(nn.Module):
    def __init__(self,
        spec_config={},
        length=32000,
        compress_exp=0.3,
    ):
        super().__init__()
        self.s_tr = spec_transformator(spec_config=spec_config)
        self.length = length
        self.compress_exp = compress_exp

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
        Yp = torch.complex(pred_real, pred_imag)
        Yt = torch.complex(true_real, true_imag)

        y_pred = self.s_tr.istft(Yp)
        y_true = self.s_tr.istft(Yt)
        
        # SI-SNR (your formula kept)
        y_true_proj = (torch.sum(y_true * y_pred, dim=-1, keepdim=True) * y_true
                       / (torch.sum(y_true ** 2, dim=-1, keepdim=True) + 1e-8))
        sisnr = -torch.log10(
            torch.norm(y_true_proj, dim=-1, keepdim=True) ** 2
            / (torch.norm(y_pred - y_true_proj, dim=-1, keepdim=True) ** 2 + 1e-8)
            + 1e-8
        ).mean()

        return 30 * (real_loss + imag_loss) + 70 * mag_loss + sisnr