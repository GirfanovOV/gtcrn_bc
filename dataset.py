import torch
from util import _stft
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
import torchaudio.functional as F
import soundfile as sf
import io


class VibravoxLocal(Dataset):
    def __init__(self, repo, split):
        self.ds = load_dataset(repo, split=split)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        row = self.ds[idx]
        ac = row['headset_microphone'].get_all_samples().data
        bc = row['temple_vibration_pickup'].get_all_samples().data
        return dict(ac_clean=ac.squeeze(), bc=bc.squeeze())
    
def collate_vibravox(batch):
    ac_list = [item['ac_clean'] for item in batch]
    bc_list = [item['bc'] for item in batch]

    lengths = torch.tensor([x.shape[-1] for x in ac_list], dtype=torch.long)

    ac_clean = pad_sequence(ac_list, batch_first=True)
    bc = pad_sequence(bc_list, batch_first=True)

    return {
        'ac_clean': ac_clean,
        'bc': bc,
        'lengths': lengths
    }

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
        repo='verbreb/vibravox_16k_8s_headset_temple_full',
        split='train',
        batch_size=8,
        num_workers=0,
        pin_memory=False
):
    dataset = VibravoxLocal(repo, split)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_vibravox
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