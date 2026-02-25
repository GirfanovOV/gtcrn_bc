"""
Training script for GTCRN-BC.

Usage from notebook:
    from train import train
    train(config)

Usage from command line:
    python train.py
"""
import os
import time
import torch
import torch.nn as nn
from pathlib import Path
from tqdm.auto import tqdm
import argparse

from gtcrn_bc import GTCRN_BC, GTCRN
from loss import HybridLoss
from dataset import create_dataloader
from pprint import pprint
# import warnings
# warnings.filterwarnings('ignore')


# ── Default configuration ──────────────────────────────────────────────────

DEFAULT_CONFIG = dict(
    # Model
    model_type="gtcrn_bc",         # "gtcrn_bc" or "gtcrn" (AC-only baseline)

    # Dataset
    repo='verbreb/vibravox_16k_2s_subset',
    snr_range=(0, 20),             # dB range for Gaussian noise on AC

    # Training
    batch_size=8,
    lr=3e-4,
    epochs=50,
    grad_clip=5.0,

    # Data limits (set to None for full dataset)
    max_train_samples=None,        # e.g. 2000 for quick test
    max_val_samples=None,          # e.g. 500 for quick test
    num_workers=0,                 # 0 for Mac, 2-4 for Colab


    # Checkpointing
    save_dir="checkpoints",
    save_every=5,                  # save checkpoint every N epochs

    # Device
    device=None,                   # auto-detect if None
    mode='forehead'

    # FFT
    # n_fft = 512,
    # hop_length = 128,
    # win_length = 512,
    # center = True,
    # pad_mode = "reflect",
    # onesided = True
)


def get_device(requested=None):
    """Auto-detect best device: CUDA > MPS > CPU."""
    if requested is not None:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def validate(model, val_loader, loss_fn, device, model_type):
    """Run validation loop, return average loss."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            ac_noisy, bc, ac_clean, _ = [x.to(device) for x in batch]

            if model_type == "gtcrn_bc":
                pred = model(ac_noisy, bc)
            else:
                pred = model(ac_noisy)

            loss = loss_fn(pred, ac_clean)
            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(n_batches, 1)

def make_pbar(iterable, total=None, desc=None):
    # Colab/TTY can be flaky; these settings are usually stable.
    return tqdm(
        iterable,
        total=total,
        desc=desc,
        dynamic_ncols=True,
        mininterval=0.2,
        maxinterval=1.0,
        smoothing=0.0,
        ascii=True,          # more robust in terminals
        leave=False,         # avoid accumulating bars
    )


def train(config=None):
    """
    Main training function.
    
    Args:
        config: dict overriding DEFAULT_CONFIG values, or None for defaults.
    
    Returns:
        model: trained model
        history: dict with 'train_loss' and 'val_loss' lists
    """
    # Merge config
    cfg = {**DEFAULT_CONFIG}
    if config:
        cfg.update(config)

    device = get_device(cfg["device"])
    print(f"Device: {device}")

    # ── Create model ───────────────────────────────────────────────────
    if cfg["model_type"] == "gtcrn_bc":
        model = GTCRN_BC()
    else:
        model = GTCRN()

    model = model.to(device)
    total, trainable = count_parameters(model)
    print(f"Model: {cfg['model_type']} | Params: {total:,} total, {trainable:,} trainable")

    print('Train config:')
    pprint(cfg)

    # ── Data ───────────────────────────────────────────────────────────
    train_loader = create_dataloader(
        repo=cfg['repo'],
        split='train',
        mode=cfg['mode'],
        batch_size=cfg['batch_size'],
        num_workers=cfg['num_workers'],
        snr_range=cfg['snr_range'],
        spec_config={}
    )

    val_loader = create_dataloader(
        repo=cfg['repo'],
        split='test',
        mode=cfg['mode'],
        batch_size=cfg['batch_size'],
        num_workers=cfg['num_workers'],
        snr_range=cfg['snr_range'],
        spec_config={}
    )

    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # ── Loss & Optimizer ───────────────────────────────────────────────
    loss_fn = HybridLoss(spec_config={}).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # ── Checkpointing ──────────────────────────────────────────────────
    save_dir = Path(cfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Training loop ──────────────────────────────────────────────────
    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()

        pbar = make_pbar(train_loader, total=len(train_loader), desc=f"Epoch {epoch}/{cfg['epochs']}")

        for batch_idx, batch in enumerate(pbar):
            ac_noisy, bc, ac_clean, _ = [x.to(device) for x in batch]
            optimizer.zero_grad()

            if cfg["model_type"] == "gtcrn_bc":
                pred = model(ac_noisy, bc)
            else:
                pred = model(ac_noisy)

            loss = loss_fn(pred, ac_clean)
            loss.backward()

            if cfg["grad_clip"] > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])

            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

            # Update right-side metrics every 10 batches (and on batch 1)
            if (n_batches % 10 == 0) or (n_batches == 1):
                avg_epoch_loss = epoch_loss / n_batches
                pbar.set_postfix({"avg_loss": f"{avg_epoch_loss:.4f}"}, refresh=False)

            # # Print progress every 50 batches
            # if (batch_idx + 1) % 50 == 0:
            #     avg = epoch_loss / n_batches
            #     print(f"  [{epoch}] batch {batch_idx + 1}/{len(train_loader)} "
            #           f"loss: {avg:.4f}")

        # ── Epoch summary ──────────────────────────────────────────────
        train_loss = epoch_loss / max(n_batches, 1)
        val_loss = validate(model, val_loader, loss_fn, device, cfg["model_type"])
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        scheduler.step(val_loss)
        lr_now = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch}/{cfg['epochs']} | "
              f"train: {train_loss:.4f} | val: {val_loss:.4f} | "
              f"lr: {lr_now:.2e} | time: {elapsed:.1f}s")

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_loss": val_loss,
                "config": cfg,
            }, save_dir / "best_model.pt")
            print(f"  ★ New best model saved (val_loss: {val_loss:.4f})")

        # Periodic save
        if cfg["save_every"] > 0 and epoch % cfg["save_every"] == 0:
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_loss": val_loss,
                "config": cfg,
            }, save_dir / f"checkpoint_epoch{epoch}.pt")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Best model saved to: {save_dir / 'best_model.pt'}")

    return model, history

def parse_args():
    parser = argparse.ArgumentParser(description="Train GTCRN-BC")

    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="Number of DataLoader workers")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    cli_config = {}

    if args.batch_size is not None:
        cli_config["batch_size"] = args.batch_size

    if args.num_workers is not None:
        cli_config["num_workers"] = args.num_workers

    train(cli_config)