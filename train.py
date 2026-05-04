import torch
import torch.nn as nn
from pathlib import Path
import argparse
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

DEFAULT_CONFIG = dict(
    # Training
    batch_size=128,
    lr=1e-3,
    epochs=265,
    grad_clip=5.0,
    max_train_batches=None,
    max_val_batches=None,

    # Data limits (set to None for full dataset)
    max_train_samples=None,        # e.g. 2000 for quick test
    max_val_samples=None,          # e.g. 500 for quick test
    num_workers=2,                 # 0 for Mac, 2-4 for Colab

    snr_min=-5,
    snr_max=15,

    # Checkpointing
    save_dir="checkpoints",
    save_every=20,                  # save checkpoint every N epochs
    save_checkpoints=True,

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
    return {"total": 0.0, "real": 0.0, "imag": 0.0, "mag": 0.0, "sisnr": 0.0}

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
        f"si={components['sisnr']:.4f}"
    )

def make_noisy_batch(batch, noise_iter, cfg, device):
    bc = batch['bc'].to(device)
    ac_clean = batch['ac_clean'].to(device)
    lengths = batch['lengths'].to(device)
    batch_size, length = ac_clean.shape

    noise = next(noise_iter)['noise'].to(device)
    noise = random_crop_1d(match_batch(noise, batch_size), length)
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
        split='train',
        batch_size=cfg['batch_size'],
        num_workers=cfg['num_workers'],
        pin_memory=cfg['pin_memory']
    )

    val_loader = create_dataloader(
        split='test',
        batch_size=cfg['batch_size'],
        num_workers=cfg['num_workers'],
        pin_memory=cfg['pin_memory']
    )

    noise_loader = create_dataloader_noise(
        batch_size=cfg['batch_size'],
        num_workers=cfg['num_workers'],
        pin_memory=cfg['pin_memory']
    )
    noise_iter = infinite_loader(noise_loader)

    return train_loader, val_loader, noise_iter

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
    n_batches = 0

    for batch in pbar:
        ac_noisy, bc, ac_clean, lengths = make_noisy_batch(batch, noise_iter, cfg, device)
        ac_noisy, bc, ac_clean = to_model_inputs(ac_noisy, bc, ac_clean)

        optimizer.zero_grad()
        pred = model(ac_noisy, bc)
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
    return epoch_loss / max(n_batches, 1), avg_components

def save_checkpoint(path, epoch, model, optimizer, val_loss, cfg):
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "val_loss": val_loss,
        "config": cfg,
    }, path)

def validate(model, val_loader, noise_iter, cfg, loss_fn, device):
    """Run validation loop, return average loss."""
    model.eval()
    total_loss = 0.0
    component_sums = init_loss_components()
    n_batches = 0
    pbar = make_pbar(
        limit_batches(val_loader, cfg["max_val_batches"]),
        total=limited_len(val_loader, cfg["max_val_batches"]),
    )

    with torch.no_grad():
        for batch in pbar:
            ac_noisy, bc, ac_clean, lengths = make_noisy_batch(batch, noise_iter, cfg, device)
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

    device = get_device(cfg["device"])
    print(f"Device: {device}")

    # ── Create model ───────────────────────────────────────────────────
    model = GTCRN()
    model = model.to(device)

    print('Train config:')
    pprint(cfg)

    train_loader, val_loader, noise_iter = prepare_data(cfg)
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # ── Loss & Optimizer ───────────────────────────────────────────────
    loss_fn = HybridLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # ── Checkpointing ──────────────────────────────────────────────────
    save_dir = Path(cfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")

    for epoch in range(1, cfg["epochs"] + 1):
        pbar = make_pbar(
            limit_batches(train_loader, cfg["max_train_batches"]),
            total=limited_len(train_loader, cfg["max_train_batches"]),
            desc=f"Epoch {epoch}/{cfg['epochs']}",
        )
        
        train_loss, train_components = train_epoch(pbar, cfg, noise_iter, device, model, optimizer, loss_fn)
        val_loss, val_components = validate(model, val_loader, noise_iter, cfg, loss_fn, device)
        
        scheduler.step(val_loss)
        lr_now = optimizer.param_groups[0]["lr"]
        
        print(f"Epoch {epoch}/{cfg['epochs']} | "
            f"train: {train_loss:.4f} | val: {val_loss:.4f} | "
            f"lr: {lr_now:.2e}"
        )
        print(f"  train components: {format_loss_components(train_components)}")
        print(f"  val components:   {format_loss_components(val_components)}")

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
    parser.add_argument("--mode", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_val_batches", type=int, default=None)

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
