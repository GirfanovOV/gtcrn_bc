import torch
import torch.nn as nn
from pathlib import Path
import argparse
import csv
from itertools import islice
from util import(
    _stft,
    make_pbar,
    get_device,
    infinite_loader,
    match_batch,
    random_crop_1d,
)
from gtcrn_bc import GTCRN
from loss import HybridLoss
from dataset import create_dataloader, create_dataloader_noise
from pprint import pprint

import warnings
warnings.filterwarnings('ignore')


# ── Default configuration ──────────────────────────────────────────────────

DATASET_REPOS = (
    "verbreb/vibravox_abcs_merge_headset_temple_16k",
    "verbreb/abcs_16k_headset_temple",
    "verbreb/vibravox_16k_8s_headset_temple_full",
)
DEFAULT_DATASET_REPO = "verbreb/vibravox_16k_8s_headset_temple_full"

DEFAULT_CONFIG = dict(
    # Training
    batch_size=128,
    lr=1e-3,
    epochs=265,
    grad_clip=5.0,
    max_train_batches=None,
    max_val_batches=None,
    sisnr_weight=1.0,

    # Data limits (set to None for full dataset)
    dataset_repo=DEFAULT_DATASET_REPO,
    max_train_samples=None,        # e.g. 2000 for quick test
    max_val_samples=None,          # e.g. 500 for quick test
    num_workers=2,                 # 0 for Mac, 2-4 for Colab

    snr_min=-5,
    snr_max=15,
    val_deterministic=True,
    val_snr_db=None,

    # Checkpointing
    save_dir="checkpoints",
    save_every=20,                  # save checkpoint every N epochs
    save_checkpoints=True,
    loss_log_path=None,             # optional CSV path for loss component logging
    grad_log_path=None,             # optional CSV path for last-batch gradient norm logging

    # Device
    device=None,                   # auto-detect if None
    mode='temple',
    pin_memory=False
)

CHECK_CONFIG = dict(
    batch_size=2,
    epochs=1,
    num_workers=0,
    pin_memory=False,
    max_train_batches=2,
    max_val_batches=1,
    save_every=0,
    save_checkpoints=False,
)

def limit_batches(loader, max_batches):
    if max_batches is None:
        return loader
    return islice(loader, max_batches)

def limited_len(loader, max_batches):
    if max_batches is None:
        return len(loader)
    return min(len(loader), max_batches)

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
    "lr",
    "grad_norm_total",
    "grad_norm_weighted_real",
    "grad_norm_weighted_imag",
    "grad_norm_weighted_mag",
    "grad_norm_weighted_sisnr",
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

def append_grad_log(path, epoch, lr, grad_norms):
    if path is None or grad_norms is None:
        return
    row = {
        "epoch": epoch,
        "lr": lr,
        **grad_norms,
    }
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=GRAD_LOG_FIELDS)
        writer.writerow(row)

def component_grad_norms(model, grad_components):
    params = [p for p in model.parameters() if p.requires_grad]
    norms = {}
    for name, component in grad_components.items():
        grads = torch.autograd.grad(
            component,
            params,
            retain_graph=True,
            allow_unused=True,
        )
        sq_norm = torch.zeros((), device=component.device)
        for grad in grads:
            if grad is not None:
                sq_norm = sq_norm + grad.detach().pow(2).sum()
        norms[f"grad_norm_{name}"] = sq_norm.sqrt().item()
    return norms

def center_crop_1d(x: torch.Tensor, T: int) -> torch.Tensor:
    """
    x: [B, L] -> return deterministic centered [B, T].
    """
    B, L = x.shape
    if L < T:
        pad = T - L
        x = torch.nn.functional.pad(x, (0, pad))
        L = T
    start = (L - T) // 2
    return x[:, start:start + T]

def make_noisy_batch(batch, noise_iter, cfg, device, deterministic=False):
    bc = batch['bc'].to(device)
    ac_clean = batch['ac_clean'].to(device)
    lengths = batch['lengths'].to(device)
    batch_size, length = ac_clean.shape

    noise = next(noise_iter)['noise'].to(device)
    noise = match_batch(noise, batch_size)
    if deterministic:
        noise = center_crop_1d(noise, length)
        val_snr_db = cfg['val_snr_db']
        if val_snr_db is None:
            val_snr_db = 0.5 * (cfg['snr_min'] + cfg['snr_max'])
        snr_db = torch.full((batch_size,), val_snr_db, device=device, dtype=ac_clean.dtype)
    else:
        noise = random_crop_1d(noise, length)
        snr_db = torch.empty(batch_size, device=device).uniform_(cfg['snr_min'], cfg['snr_max'])

    valid = torch.arange(length, device=device).unsqueeze(0) < lengths.unsqueeze(1)
    valid_f = valid.to(ac_clean.dtype)
    valid_count = lengths.clamp_min(1).to(ac_clean.dtype).unsqueeze(1)

    sig_pow = ac_clean.pow(2).mul(valid_f).sum(dim=1, keepdim=True) / valid_count
    noi_pow = noise.pow(2).mul(valid_f).sum(dim=1, keepdim=True) / valid_count + 1e-8
    snr_lin = (10.0 ** (snr_db / 10.0)).view(-1, 1)
    scale = torch.sqrt(sig_pow / (snr_lin * noi_pow + 1e-8))

    ac_noisy = (ac_clean + noise * scale) * valid_f
    return ac_noisy, bc, ac_clean, lengths

def to_model_inputs(ac_noisy, bc, ac_clean=None):
    ac_noisy = torch.view_as_real(_stft(ac_noisy))
    bc = torch.view_as_real(_stft(bc))
    if ac_clean is None:
        return ac_noisy, bc
    return ac_noisy, bc, torch.view_as_real(_stft(ac_clean))

def prepare_data(cfg):
    train_loader = create_dataloader(
        repo=cfg['dataset_repo'],
        split='train',
        batch_size=cfg['batch_size'],
        num_workers=cfg['num_workers'],
        pin_memory=cfg['pin_memory']
    )

    val_loader = create_dataloader(
        repo=cfg['dataset_repo'],
        split='test',
        batch_size=cfg['batch_size'],
        num_workers=cfg['num_workers'],
        pin_memory=cfg['pin_memory']
    )

    train_noise_loader = create_dataloader_noise(
        batch_size=cfg['batch_size'],
        num_workers=cfg['num_workers'],
        pin_memory=cfg['pin_memory']
    )
    val_noise_loader = create_dataloader_noise(
        batch_size=cfg['batch_size'],
        num_workers=cfg['num_workers'],
        pin_memory=cfg['pin_memory']
    )
    train_noise_iter = infinite_loader(train_noise_loader)

    return train_loader, val_loader, train_noise_iter, val_noise_loader

def train_epoch(
        pbar,
        cfg,
        noise_iter,
        device,
        model,
        optimizer,
        loss_fn
    ):    
    model.train()
    epoch_loss = 0.0
    component_sums = init_loss_components()
    last_batch_grad_norms = None
    n_batches = 0

    for batch in pbar:
        is_last_batch = pbar.total is not None and n_batches + 1 == pbar.total
        ac_noisy, bc, ac_clean, lengths = make_noisy_batch(batch, noise_iter, cfg, device)
        ac_noisy, bc, ac_clean = to_model_inputs(ac_noisy, bc, ac_clean)

        optimizer.zero_grad()
        pred = model(ac_noisy, bc)
        if cfg["grad_log_path"] is not None and is_last_batch:
            loss, components, grad_components = loss_fn(
                pred,
                ac_clean,
                lengths,
                return_components=True,
                return_grad_components=True,
            )
            last_batch_grad_norms = component_grad_norms(model, grad_components)
        else:
            loss, components = loss_fn(pred, ac_clean, lengths, return_components=True)
        loss.backward()

        if cfg["grad_clip"] > 0:
            nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])

        optimizer.step()
        epoch_loss += loss.item()
        add_loss_components(component_sums, components)
        n_batches += 1

        # Update right-side metrics every 10 batches (and on batch 1)
        if (n_batches % 10 == 0) or (n_batches == 1):
            avg_components = average_loss_components(component_sums, n_batches)
            pbar.set_postfix({
                "avg_loss": f"{avg_components['total']:.4f}",
                "sisnr": f"{avg_components['sisnr']:.4f}",
            }, refresh=False)

    avg_components = average_loss_components(component_sums, n_batches)
    return epoch_loss / max(n_batches, 1), avg_components, last_batch_grad_norms

def save_checkpoint(path, epoch, model, optimizer, val_loss, cfg):
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "val_loss": val_loss,
        "config": cfg,
    }, path)

def validate(model, val_loader, val_noise_loader, cfg, loss_fn, device):
    """Run validation loop, return average loss."""
    model.eval()
    total_loss = 0.0
    component_sums = init_loss_components()
    n_batches = 0
    pbar = make_pbar(
        limit_batches(val_loader, cfg["max_val_batches"]),
        total=limited_len(val_loader, cfg["max_val_batches"]),
    )
    noise_iter = infinite_loader(val_noise_loader)

    with torch.no_grad():
        for batch in pbar:
            ac_noisy, bc, ac_clean, lengths = make_noisy_batch(
                batch,
                noise_iter,
                cfg,
                device,
                deterministic=cfg["val_deterministic"],
            )
            ac_noisy, bc, ac_clean = to_model_inputs(ac_noisy, bc, ac_clean)

            pred = model(ac_noisy, bc)
            loss, components = loss_fn(pred, ac_clean, lengths, return_components=True)
            total_loss += loss.cpu().item()
            add_loss_components(component_sums, components)
            n_batches += 1

    avg_components = average_loss_components(component_sums, n_batches)
    return total_loss / max(n_batches, 1), avg_components

def train(config=None):
    # Merge config
    cfg = {**DEFAULT_CONFIG}
    if config:
        cfg.update(config)
    if cfg["dataset_repo"] not in DATASET_REPOS:
        raise ValueError(
            f"Unknown dataset_repo={cfg['dataset_repo']!r}. "
            f"Choose one of: {', '.join(DATASET_REPOS)}"
        )

    device = get_device(cfg["device"])
    print(f"Device: {device}")

    # ── Create model ───────────────────────────────────────────────────
    model = GTCRN()
    model = model.to(device)

    print('Train config:')
    pprint(cfg)

    train_loader, val_loader, train_noise_iter, val_noise_loader = prepare_data(cfg)
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # ── Loss & Optimizer ───────────────────────────────────────────────
    loss_fn = HybridLoss(sisnr_weight=cfg["sisnr_weight"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # ── Checkpointing ──────────────────────────────────────────────────
    save_dir = Path(cfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    loss_log_path = init_loss_log(cfg["loss_log_path"])
    if loss_log_path is not None:
        print(f"Loss component log: {loss_log_path}")
    grad_log_path = init_grad_log(cfg["grad_log_path"])
    if grad_log_path is not None:
        print(f"Gradient norm log: {grad_log_path}")

    best_val_loss = float("inf")

    for epoch in range(1, cfg["epochs"] + 1):
        pbar = make_pbar(
            limit_batches(train_loader, cfg["max_train_batches"]),
            total=limited_len(train_loader, cfg["max_train_batches"]),
            desc=f"Epoch {epoch}/{cfg['epochs']}",
        )
        
        train_loss, train_components, train_grad_norms = train_epoch(
            pbar,
            cfg,
            train_noise_iter,
            device,
            model,
            optimizer,
            loss_fn,
        )
        val_loss, val_components = validate(model, val_loader, val_noise_loader, cfg, loss_fn, device)
        
        scheduler.step(val_loss)
        lr_now = optimizer.param_groups[0]["lr"]
        
        print(f"Epoch {epoch}/{cfg['epochs']} | "
            f"train: {train_loss:.4f} | val: {val_loss:.4f} | "
            f"lr: {lr_now:.2e}"
        )
        print(f"  train components: {format_loss_components(train_components)}")
        print(f"  val components:   {format_loss_components(val_components)}")
        append_loss_log(loss_log_path, epoch, "train", lr_now, train_components)
        append_loss_log(loss_log_path, epoch, "val", lr_now, val_components)
        append_grad_log(grad_log_path, epoch, lr_now, train_grad_norms)

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if cfg["save_checkpoints"]:
                save_checkpoint(save_dir / "best_model.pt", epoch, model, optimizer, val_loss, cfg)

        # Periodic save
        if cfg["save_checkpoints"] and cfg["save_every"] > 0 and epoch % cfg["save_every"] == 0:
            save_checkpoint(save_dir / f"checkpoint_epoch{epoch}.pt", epoch, model, optimizer, val_loss, cfg)

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    if cfg["save_checkpoints"]:
        print(f"Best model saved to: {save_dir / 'best_model.pt'}")
    else:
        print("Checkpoint saving disabled.")

    return model

def parse_args():
    parser = argparse.ArgumentParser(description="Train GTCRN-BC")

    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--pin-memory",
        "--pin_memory",
        dest="pin_memory",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        "--no-pin-memory",
        "--no_pin_memory",
        dest="pin_memory",
        action="store_false",
    )
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--snr_min", type=int, default=None)
    parser.add_argument("--snr_max", type=int, default=None)
    parser.add_argument("--val_snr_db", type=float, default=None)
    parser.add_argument(
        "--random-val",
        "--random_val",
        dest="val_deterministic",
        action="store_false",
        default=None,
    )
    parser.add_argument("--mode", type=str, default=None)
    parser.add_argument(
        "--dataset_repo",
        "--dataset-repo",
        choices=DATASET_REPOS,
        default=None,
        help=f"HF dataset repo for training. Default: {DEFAULT_DATASET_REPO}",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_val_batches", type=int, default=None)
    parser.add_argument("--sisnr_weight", "--sisnr-weight", type=float, default=None)
    parser.add_argument("--loss_log_path", "--loss-log-path", type=str, default=None)
    parser.add_argument("--grad_log_path", "--grad-log-path", type=str, default=None)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    cli_config = {
        key: value
        for key, value in vars(args).items()
        if value is not None and key != "check"
    }

    if args.check:
        cli_config = {**CHECK_CONFIG, **cli_config}

    train(cli_config)
