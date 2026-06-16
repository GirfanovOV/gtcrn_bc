import csv
from pathlib import Path


def init_loss_components():
    return {
        "total": 0.0,
        "real": 0.0,
        "imag": 0.0,
        "mag": 0.0,
        "sisnr": 0.0,
        "weighted_real": 0.0,
        "weighted_imag": 0.0,
        "weighted_mag": 0.0,
        "weighted_sisnr": 0.0,
    }


def add_loss_components(acc, components):
    for key in acc:
        acc[key] += components[key].item()


def average_loss_components(acc, n_batches):
    n = max(n_batches, 1)
    return {key: value / n for key, value in acc.items()}


def format_loss_components(components):
    return (
        f"r={components['real']:.4f}, "
        f"i={components['imag']:.4f}, "
        f"m={components['mag']:.4f}, "
        f"si={components['sisnr']:.4f}, "
        f"wr={components['weighted_real']:.4f}, "
        f"wi={components['weighted_imag']:.4f}, "
        f"wm={components['weighted_mag']:.4f}, "
        f"wsi={components['weighted_sisnr']:.4f}"
    )


LOSS_LOG_FIELDS = [
    "epoch",
    "split",
    "lr",
    "total",
    "real",
    "imag",
    "mag",
    "sisnr",
    "weighted_real",
    "weighted_imag",
    "weighted_mag",
    "weighted_sisnr",
]


def init_loss_log(path):
    if path is None:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOSS_LOG_FIELDS)
        writer.writeheader()
    return path


def append_loss_log(path, epoch, split, lr, components):
    if path is None:
        return
    row = {
        "epoch": epoch,
        "split": split,
        "lr": lr,
        **{key: components[key] for key in LOSS_LOG_FIELDS if key in components},
    }
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOSS_LOG_FIELDS)
        writer.writerow(row)


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
