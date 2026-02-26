import torch
from util import spec_transformator
from torch.utils.data import Dataset
from datasets import load_dataset
import dill as pickle


class VibravoxLocal(Dataset):
    def __init__(self, repo, split, mode):
        self.ds = load_dataset(repo, split=split)
        self.mode = mode

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        try:
            row = self.ds[idx]
            hs = row['headset_path'].get_all_samples().data
        except:
            return

        if self.mode == 'forehead':
            bc = row["forehead_path"].get_all_samples().data
        elif self.mode == 'temple':
            bc = row["temple_path"].get_all_samples().data
        else:
            raise

        return hs.squeeze(), bc.squeeze()


def make_collate_fn(snr_range, spec_config={}):

    # s_tr = spec_transformator(spec_config=spec_config)

    def collate(batch):
        # batch: list of (ac_clean [32000], bc [32000])
        batch = [x for x in batch if x is not None]
        ac_list, bc_list = zip(*batch)
        ac_clean = torch.stack(ac_list, dim=0)  # [B, 32000]
        bc = torch.stack(bc_list, dim=0)        # [B, 32000]

        # X = s_tr.stft(ac_clean)
        # X_bc = s_tr.stft(bc)

        X = torch.stft(ac_clean, 512, 256, 512, torch.hann_window(512).pow(0.5), return_complex=True)
        X_bc = torch.stft(bc, 512, 256, 512, torch.hann_window(512).pow(0.5), return_complex=True)

        # Complex Gaussian noise
        N = torch.complex(torch.randn_like(X.real), torch.randn_like(X.real))

        # Random SNR per sample
        B = ac_clean.shape[0]
        snr_db = torch.empty(B, device=X.device).uniform_(snr_range[0], snr_range[1])
        snr_lin = 10 ** (snr_db / 10.0)

        eps = 1e-12
        P_sig = (X.real.square() + X.imag.square()).mean(dim=(1, 2)).clamp_min(eps)  # [B]
        P_n0  = (N.real.square() + N.imag.square()).mean(dim=(1, 2)).clamp_min(eps)  # [B]

        scale = torch.sqrt((P_sig / snr_lin) / P_n0)  # [B]
        Y = X + N * scale[:, None, None]

        # 2-channel RI: [B, 2, F, T]
        ac_clean_ri = torch.view_as_real(X).contiguous()
        bc_ri = torch.view_as_real(X_bc).contiguous()
        ac_noisy_ri = torch.view_as_real(Y).contiguous()
        
        return ac_clean_ri, bc_ri, ac_noisy_ri, snr_db
    
    return collate




def create_dataloader(
        repo='verbreb/vibravox_16k_2s_subset',
        split='train',
        mode='forehead',
        batch_size=8,
        num_workers=0,
        snr_range=(0,20)
):
    dataset = VibravoxLocal(repo, split, mode)
    collate = make_collate_fn(snr_range=snr_range)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=False,   # pinning mainly helps CUDA; can ignore on Mac
    )
    return loader


if __name__ == '__main__':
    train_dl = create_dataloader(split='train')
    test_dl = create_dataloader(split='test')

    print(f'bsz: {train_dl.batch_size}, n_wrk: {train_dl.num_workers}')
    print(f'Train_len: {len(train_dl)}, test_len: {len(test_dl)}')

    ac_clean_ri, bc_ri, ac_noisy_ri, snr_db = next(iter(train_dl))

    print(f'{ac_clean_ri.shape = }')
    print(f'{bc_ri.shape = }')
    print(f'{ac_noisy_ri.shape = }')
    print(f'{snr_db.shape = }')