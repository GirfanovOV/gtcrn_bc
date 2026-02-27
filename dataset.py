import torch
from util import _stft
from torch.utils.data import Dataset
from datasets import load_dataset
import torchaudio.functional as F


class VibravoxLocal(Dataset):
    def __init__(self, repo, split, mode):
        self.ds = load_dataset(repo, split=split)
        self.mode = mode

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        try:
            row = self.ds[idx]
            ac  = row['headset_path'].get_all_samples().data
        except:
            return

        if self.mode == 'forehead':
            bc = row["forehead_path"].get_all_samples().data
        elif self.mode == 'temple':
            bc = row["temple_path"].get_all_samples().data
        else:
            raise
        
        return dict(
            ac_clean=ac.squeeze(),
            bc=bc.squeeze()
        )

def make_collate_fn(snr_range, spec_config={}):

    def collate(batch):
        batch = [x for x in batch if x is not None]
        ac_list = [b['ac_clean'] for b in batch]
        bc_list = [b['bc'] for b in batch]
        ac_clean = torch.stack(ac_list, dim=0)  # [B, 32000]
        bc = torch.stack(bc_list, dim=0)        # [B, 32000]

        # Random SNR per sample
        B = ac_clean.shape[0]
        snr_db = torch.empty(B, device=ac_clean.device).uniform_(snr_range[0], snr_range[1])
        
        noise = torch.randn_like(ac_clean, device=ac_clean.device)
        ac_noisy = F.add_noise(ac_clean, noise, snr_db)
        
        return dict(
            ac_clean=ac_clean,
            ac_noisy=ac_noisy,
            bc=bc,
            snr_db=snr_db
        )
    return collate




def create_dataloader(
        repo='verbreb/vibravox_16k_2s_subset',
        split='train',
        mode='forehead',
        batch_size=8,
        num_workers=0,
        snr_range=(0,20),
        pin_memory=False
):
    dataset = VibravoxLocal(repo, split, mode)
    collate = make_collate_fn(snr_range=snr_range)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=pin_memory,   # pinning mainly helps CUDA; can ignore on Mac
    )
    return loader


if __name__ == '__main__':
    train_dl = create_dataloader(split='train')
    test_dl = create_dataloader(split='test')

    print(f'bsz: {train_dl.batch_size}, n_wrk: {train_dl.num_workers}')
    print(f'Train_len: {len(train_dl)}, test_len: {len(test_dl)}')

    batch = next(iter(test_dl))
    ac_clean=batch['ac_clean']
    ac_noisy=batch['ac_noisy']
    bc=batch['bc']
    snr_db=batch['snr_db']

    print(f'{ac_clean.shape = }')
    print(f'{bc.shape = }')
    print(f'{ac_noisy.shape = }')
    print(f'{snr_db.shape = }')