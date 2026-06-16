import torch
from functools import partial
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset, Audio
import torchaudio.functional as F
import soundfile as sf
import io


TARGET_SAMPLE_RATE = 16000


def audio_length_to_samples(audio_length_sec, sample_rate=TARGET_SAMPLE_RATE):
    if audio_length_sec is None:
        return None
    return int(round(audio_length_sec * sample_rate))


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
        for column in ("headset_microphone", "temple_vibration_pickup"):
            self.ds = self.ds.cast_column(column, Audio(decode=False))

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        row = self.ds[idx]
        ac = read_audio(row['headset_microphone'])
        bc = read_audio(row['temple_vibration_pickup'])
        return dict(ac_clean=ac, bc=bc)


def _pad_or_trim_1d(x, target_num_samples, start=0):
    x = x[start:start + target_num_samples]
    if x.shape[-1] < target_num_samples:
        x = torch.nn.functional.pad(x, (0, target_num_samples - x.shape[-1]))
    return x


def crop_or_pad_pair(ac, bc, target_num_samples, random_crop=False):
    ac_len = ac.shape[-1]
    bc_len = bc.shape[-1]
    pair_len = min(ac_len, bc_len)

    if pair_len > target_num_samples:
        max_start = pair_len - target_num_samples
        if random_crop:
            start = torch.randint(0, max_start + 1, ()).item()
        else:
            start = max_start // 2
        length = target_num_samples
    else:
        start = 0
        length = pair_len

    ac = _pad_or_trim_1d(ac, target_num_samples, start)
    bc = _pad_or_trim_1d(bc, target_num_samples, start)
    return ac, bc, length


def collate_vibravox(batch, audio_num_samples=None, random_crop=False):
    if audio_num_samples is not None:
        cropped = [
            crop_or_pad_pair(
                item["ac_clean"],
                item["bc"],
                audio_num_samples,
                random_crop=random_crop,
            )
            for item in batch
        ]
        ac_list = [item[0] for item in cropped]
        bc_list = [item[1] for item in cropped]
        lengths = torch.tensor([item[2] for item in cropped], dtype=torch.long)
        ac_clean = torch.stack(ac_list)
        bc = torch.stack(bc_list)
        return {
            'ac_clean': ac_clean,
            'bc': bc,
            'lengths': lengths
        }

    ac_list = [item['ac_clean'] for item in batch]
    bc_list = [item['bc'] for item in batch]

    lengths = torch.tensor([x.shape[-1] for x in ac_list], dtype=torch.long)

    # Signals are padded only for batching. The original lengths drive masks
    # in training/loss so padded samples do not affect SNR or loss.
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
        pin_memory=False,
        audio_length_sec=8,
        random_crop=None,
):
    dataset = VibravoxLocal(repo, split)
    if random_crop is None:
        random_crop = split == "train"
    collate_fn = partial(
        collate_vibravox,
        audio_num_samples=audio_length_to_samples(audio_length_sec),
        random_crop=random_crop,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn
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
