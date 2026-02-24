import soundfile as sf
import torch
from torch.utils.data import Dataset
import pandas as pd

class VibravoxLocal(Dataset):
    def __init__(self, parquet_path, mode='forehead'):
        self.df = pd.read_parquet(parquet_path)
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        hs, _ = sf.read(row["headset_path"])
        if self.mode == 'forehead':
            bc, _ = sf.read(row["forehead_path"])
        elif self.mode == 'temple':
            bc, _ = sf.read(row["temple_path"])
        else:
            raise

        return torch.from_numpy(hs).float(), torch.from_numpy(bc).float()

class STFTNoiseCollate:
    def __init__(
        self,
        snr_min: float,
        snr_max: float,
        n_fft: int = 512,
        hop_length: int = 128,
        win_length: int = 512,
        center: bool = True,
        pad_mode: str = "reflect",
        onesided: bool = True,
    ):
        self.snr_min = float(snr_min)
        self.snr_max = float(snr_max)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length)
        self.center = bool(center)
        self.pad_mode = pad_mode
        self.onesided = bool(onesided)
        # Keep as plain tensor attribute; it will be pickled and recreated in workers.
        self.window = torch.hann_window(self.win_length)

    def __call__(self, batch):
        # batch: list of (ac_clean [32000], bc [32000])
        ac_list, bc_list = zip(*batch)
        ac_clean = torch.stack(ac_list, dim=0)  # [B, 32000]
        bc = torch.stack(bc_list, dim=0)        # [B, 32000]

        X = torch.stft(
            ac_clean,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(device=ac_clean.device, dtype=ac_clean.dtype),
            center=self.center,
            pad_mode=self.pad_mode,
            normalized=False,
            onesided=self.onesided,
            return_complex=True,
        )  # [B, F, T] complex

        # Complex Gaussian noise
        N = torch.complex(torch.randn_like(X.real), torch.randn_like(X.real))

        # Random SNR per sample
        B = ac_clean.shape[0]
        snr_db = torch.empty(B, device=X.device).uniform_(self.snr_min, self.snr_max)
        snr_lin = 10 ** (snr_db / 10.0)

        eps = 1e-12
        P_sig = (X.real.square() + X.imag.square()).mean(dim=(1, 2)).clamp_min(eps)  # [B]
        P_n0  = (N.real.square() + N.imag.square()).mean(dim=(1, 2)).clamp_min(eps)  # [B]

        scale = torch.sqrt((P_sig / snr_lin) / P_n0)  # [B]
        Y = X + N * scale[:, None, None]

        # 2-channel RI: [B, 2, F, T]
        ac_clean_ri = torch.view_as_real(X).permute(0, 3, 1, 2).contiguous()
        ac_noisy_ri = torch.view_as_real(Y).permute(0, 3, 1, 2).contiguous()

        return ac_clean_ri, bc, ac_noisy_ri, snr_db
    

def create_dataloader(
        path,
        batch_size,
        num_workers=4,
        snr_min=0,
        snr_max=20,
        n_fft: int = 512,
        hop_length: int = 128,
        win_length: int = 512,
        center: bool = True,
        pad_mode: str = "reflect",
        onesided: bool = True
):
    dataset = VibravoxLocal(path)
    collate = STFTNoiseCollate(snr_min=0, snr_max=20)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=False,   # pinning mainly helps CUDA; can ignore on Mac
    )
    return loader