import torch
from util import _stft
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from datasets import Audio, load_dataset
import torchaudio.functional as F
import soundfile as sf
import io


TARGET_SAMPLE_RATE = 16000


def read_audio(value, target_sample_rate=TARGET_SAMPLE_RATE):
    source = io.BytesIO(value['bytes']) if value.get('bytes') is not None else value['path']
    wav, sample_rate = sf.read(source, dtype='float32')
    wav = torch.from_numpy(wav)

    if wav.ndim > 1:
        wav = wav.mean(dim=-1)

    if sample_rate != target_sample_rate:
        wav = F.resample(wav, sample_rate, target_sample_rate)

    return wav.squeeze()


class VibravoxLocal(Dataset):
    def __init__(self, repo, split):
        self.ds = load_dataset(repo, split=split)
        for column in ('headset_microphone', 'temple_vibration_pickup'):
            self.ds = self.ds.cast_column(column, Audio(decode=False))

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        row = self.ds[idx]
        ac = read_audio(row['headset_microphone'])
        bc = read_audio(row['temple_vibration_pickup'])
        return dict(ac_clean=ac, bc=bc)
    
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
        self.ds = self.ds.cast_column('audio', Audio(decode=False))
    
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        row = self.ds[idx]
        return dict(noise=read_audio(row['audio']))


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
