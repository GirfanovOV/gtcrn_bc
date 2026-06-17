import csv
from pathlib import Path


GRAD_LOG_FIELDS = [
    "epoch",
    "batch",
    "global_step",
    "lr",
    "grad_norm_total",
    "grad_norm_weighted_real",
    "grad_norm_weighted_imag",
    "grad_norm_weighted_mag",
    "grad_norm_weighted_spectral",
    "grad_norm_weighted_sisnr",
    "cos_sisnr_real",
    "cos_sisnr_imag",
    "cos_sisnr_mag",
    "cos_sisnr_spectral",
]


def init_grad_log(path):
    if path is None:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=GRAD_LOG_FIELDS)
        writer.writeheader()
    return path


def append_grad_log(path, epoch, batch_idx, global_step, lr, grad_norms):
    if path is None or grad_norms is None:
        return
    row = {
        "epoch": epoch,
        "batch": batch_idx,
        "global_step": global_step,
        "lr": lr,
        **grad_norms,
    }
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=GRAD_LOG_FIELDS)
        writer.writerow(row)
