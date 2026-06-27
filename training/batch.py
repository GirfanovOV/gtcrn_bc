import torch

from util import _stft, center_crop_1d, match_batch, random_crop_1d


def make_noisy_batch(batch, noise_iter, cfg, device, deterministic=False):
    bc = batch["bc"].to(device)
    ac_clean = batch["ac_clean"].to(device)
    lengths = batch["lengths"].to(device)
    batch_size, length = ac_clean.shape

    noise = next(noise_iter)["noise"].to(device)
    noise = match_batch(noise, batch_size)
    if deterministic:
        noise = center_crop_1d(noise, length)
        val_snr_db = cfg["val_snr_db"]
        if val_snr_db is None:
            val_snr_db = 0.5 * (cfg["snr_min"] + cfg["snr_max"])
        snr_db = torch.full((batch_size,), val_snr_db, device=device, dtype=ac_clean.dtype)
    else:
        noise = random_crop_1d(noise, length)
        snr_db = torch.empty(batch_size, device=device).uniform_(cfg["snr_min"], cfg["snr_max"])

    valid = torch.arange(length, device=device).unsqueeze(0) < lengths.unsqueeze(1)
    valid_f = valid.to(ac_clean.dtype)
    valid_count = lengths.clamp_min(1).to(ac_clean.dtype).unsqueeze(1)

    sig_pow = ac_clean.pow(2).mul(valid_f).sum(dim=1, keepdim=True) / valid_count
    noi_pow = noise.pow(2).mul(valid_f).sum(dim=1, keepdim=True) / valid_count + 1e-8
    snr_lin = (10.0 ** (snr_db / 10.0)).view(-1, 1)
    scale = torch.sqrt(sig_pow / (snr_lin * noi_pow + 1e-8))

    ac_noisy = (ac_clean + noise * scale) * valid_f
    noise_aware_coeff = cfg["noise_aware_coeff"]
    ac_target = noise_aware_coeff * ac_clean + (1.0 - noise_aware_coeff) * ac_noisy
    return ac_noisy, bc, ac_target, lengths


def to_model_inputs(ac_noisy, bc, ac_target=None):
    ac_noisy = torch.view_as_real(_stft(ac_noisy))
    bc = torch.view_as_real(_stft(bc))
    if ac_target is None:
        return ac_noisy, bc
    return ac_noisy, bc, torch.view_as_real(_stft(ac_target))
