import torch
from util import _stft
from torch.utils.data import Dataset
from datasets import load_dataset
import torchaudio.functional as F
import soundfile as sf
import io


class VibravoxLocal(Dataset):
    def __init__(self, repo, split, mode):
        self.ds = load_dataset(repo, split=split)
        self.mode = mode

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        row = self.ds[idx]
        wav_ac, _ = sf.read(io.BytesIO(row['headset']['bytes']))
        ac = torch.from_numpy(wav_ac).float()

        if self.mode == 'forehead':
            wav_bc, _ = sf.read(io.BytesIO(row['forehead']['bytes']))
        elif self.mode == 'temple':
            wav_bc, _ = sf.read(io.BytesIO(row['temple']['bytes']))
        else:
            raise

        bc = torch.from_numpy(wav_bc).float()
        
        return dict(ac_clean=ac.squeeze(), bc=bc.squeeze())

class DemandNoiseSubset(Dataset):
    def __init__(self):
        self.ds = load_dataset('verbreb/demand_noise_subset_16k', split='noise')
    
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        row = self.ds[idx]
        wav, _ = sf.read(io.BytesIO(row['audio']['bytes']))
        noise = torch.from_numpy(wav).float()
        return dict(noise=noise.squeeze())


def create_dataloader(
        repo='verbreb/vibravox_16k_2s_ac_bc',
        split='train',
        mode='temple',
        batch_size=8,
        num_workers=0,
        pin_memory=False
):
    dataset = VibravoxLocal(repo, split, mode)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,   # pinning mainly helps CUDA; can ignore on Mac
    )
    return loader

def create_dataloader_noise(
        batch_size=8,
        num_workers=0,
        pin_memory=False
):
    dataset = DemandNoiseSubset()
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,   # pinning mainly helps CUDA; can ignore on Mac
    )
    return loader


if __name__ == '__main__':
    dl = create_dataloader()
    print(next(iter(dl)))
    dl_n = create_dataloader_noise()
    print(next(iter(dl_n)))