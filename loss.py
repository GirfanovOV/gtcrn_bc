import torch
import torch.nn as nn
from util import _istft


class HybridLoss(nn.Module):
    def __init__(self, n_fft=512, hop_length=256):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length

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

    def forward(self, pred_stft, true_stft, lengths=None):
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

        return 30*(real_loss + imag_loss) + 70*mag_loss + sisnr


if __name__ == "__main__":
    loss_func = HybridLoss()

    pred_stft = torch.randn(1, 257, 63, 2)
    true_stft = torch.randn(1, 257, 63, 2)
    loss = loss_func(pred_stft, true_stft, torch.tensor([16000]))
    print(loss)
