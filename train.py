import math
import os
import time
import torch
import torch.nn as nn
from pathlib import Path
import argparse
from util import(
    _stft,
    _istft,
    make_pbar,
    get_device,
    count_parameters,
    infinite_loader,
    match_batch,
    random_crop_1d,
    add_noise_snr
)

from gtcrn_bc import GTCRN
from loss import HybridLoss
from dataset import create_dataloader, create_dataloader_noise
from pprint import pprint
from torchmetrics import MetricCollection
from torchmetrics.audio import (
    PerceptualEvaluationSpeechQuality,
    ShortTimeObjectiveIntelligibility,
    ScaleInvariantSignalNoiseRatio,
    DeepNoiseSuppressionMeanOpinionScore
)
import warnings
warnings.filterwarnings('ignore')


# ── Default configuration ──────────────────────────────────────────────────

DEFAULT_CONFIG = dict(
    # Model
    model_type="gtcrn_bc",         # "gtcrn_bc" or "gtcrn" (AC-only baseline)

    # Dataset
    snr_range=(0, 20),             # dB range for Gaussian noise on AC

    # Training
    batch_size=128,
    lr=1e-3,
    epochs=50,
    grad_clip=5.0,

    # Data limits (set to None for full dataset)
    max_train_samples=None,        # e.g. 2000 for quick test
    max_val_samples=None,          # e.g. 500 for quick test
    num_workers=2,                 # 0 for Mac, 2-4 for Colab

    snr_min=-5,
    snr_max=15,

    # Checkpointing
    save_dir="checkpoints",
    save_every=20,                  # save checkpoint every N epochs

    # Device
    device=None,                   # auto-detect if None
    mode='temple',
    pin_memory=False
)

def validate(model, val_loader, noise_iter, cfg, loss_fn, device):
    """Run validation loop, return average loss."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    pbar = make_pbar(val_loader)

    with torch.no_grad():
        for batch in pbar:
            bc = batch['bc'].to(device)                 # [B, T]
            ac_clean = batch['ac_clean'].to(device)     # [B, T]
            B, T = ac_clean.shape
            noise_batch = next(noise_iter)['noise'].to(device)         # [B1, L]
            
            noise_batch = match_batch(noise_batch, B)
            noise_batch = random_crop_1d(noise_batch, T)
            snr_db = torch.empty(B, device=device).uniform_(cfg['snr_min'], cfg['snr_max'])  # [B]
            ac_noisy = add_noise_snr(ac_clean, noise_batch, snr_db)  # [B, T]

            bc = torch.view_as_real(_stft(bc))
            ac_noisy = torch.view_as_real(_stft(ac_noisy))
            ac_clean = torch.view_as_real(_stft(ac_clean))

            pred = model(ac_noisy, bc)

            loss = loss_fn(pred, ac_clean).cpu()
            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(n_batches, 1)

# def eval_model(model, val_loader, noise_iter, cfg, metrics, dnsmos, device):
def eval_model(model, val_loader, noise_iter, cfg, metrics, device):
    metrics.reset()
    # dnsmos.reset()

    pbar = make_pbar(val_loader)

    with torch.no_grad():
        for batch in pbar:
            bc = batch['bc'].to(device)                 # [B, T]
            ac_clean = batch['ac_clean'].to(device)     # [B, T]
            B, T = ac_clean.shape
            noise_batch = next(noise_iter)['noise'].to(device)         # [B1, L]
            
            noise_batch = match_batch(noise_batch, B)
            noise_batch = random_crop_1d(noise_batch, T)
            snr_db = torch.empty(B, device=device).uniform_(cfg['snr_min'], cfg['snr_max'])  # [B]
            ac_noisy = add_noise_snr(ac_clean, noise_batch, snr_db)  # [B, T]

            bc_model = torch.view_as_real(_stft(bc))
            ac_noisy_model = torch.view_as_real(_stft(ac_noisy))

            pred = model(ac_noisy_model, bc_model)
            pred = _istft(pred)

            metrics.update(pred, ac_clean)
            # dnsmos.update(ac_clean)
    return

# def eval_data(val_loader, noise_iter, cfg, metrics, dnsmos, device):
def eval_data(val_loader, noise_iter, cfg, metrics, device):
    metrics.reset()
    # dnsmos.reset()

    pbar = make_pbar(val_loader)

    with torch.no_grad():
        for batch in pbar:
            ac_clean = batch['ac_clean'].to(device)     # [B, T]
            B, T = ac_clean.shape
            noise_batch = next(noise_iter)['noise'].to(device)         # [B1, L]
            
            noise_batch = match_batch(noise_batch, B)
            noise_batch = random_crop_1d(noise_batch, T)
            snr_db = torch.empty(B, device=device).uniform_(cfg['snr_min'], cfg['snr_max'])  # [B]
            ac_noisy = add_noise_snr(ac_clean, noise_batch, snr_db)  # [B, T]

            metrics.update(ac_noisy, ac_clean)
            # dnsmos.update(ac_clean)
    return


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
    total, trainable = count_parameters(model)
    print(f"Model: {cfg['model_type']} | Params: {total:,} total, {trainable:,} trainable")

    print('Train config:')
    pprint(cfg)

    # ── Data ───────────────────────────────────────────────────────────
    train_loader = create_dataloader(
        split='train',
        mode=cfg['mode'],
        batch_size=cfg['batch_size'],
        num_workers=cfg['num_workers'],
        pin_memory=cfg['pin_memory']
    )

    val_loader = create_dataloader(
        split='test',
        mode=cfg['mode'],
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

    # ── Training loop ──────────────────────────────────────────────────
    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")

    metrics = MetricCollection({
        "si_snr": ScaleInvariantSignalNoiseRatio().to(device),
        "pesq": PerceptualEvaluationSpeechQuality(16000, 'wb').to(device),
        "stoi": ShortTimeObjectiveIntelligibility(16000).to(device)
    })
    dnsmos = DeepNoiseSuppressionMeanOpinionScore(16000, False).to(device)

    eval_data(val_loader, noise_iter, cfg, metrics, dnsmos, device)
    print('Eval on data', metrics.compute(), f'DNSMOS: {dnsmos.compute().item()}')
    eval_model(model, val_loader, noise_iter, cfg, metrics, dnsmos, device)
    print('Eval on model', metrics.compute(), f'DNSMOS: {dnsmos.compute().item()}')


    for epoch in range(1, cfg["epochs"] + 1):
        
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()

        pbar = make_pbar(train_loader, total=len(train_loader), desc=f"Epoch {epoch}/{cfg['epochs']}")

        for batch in pbar:
            bc = batch['bc'].to(device)                 # [B, T]
            ac_clean = batch['ac_clean'].to(device)     # [B, T]
            B, T = ac_clean.shape
            noise_batch = next(noise_iter)['noise'].to(device)         # [B1, L]
            
            noise_batch = match_batch(noise_batch, B)
            noise_batch = random_crop_1d(noise_batch, T)
            snr_db = torch.empty(B, device=device).uniform_(cfg['snr_min'], cfg['snr_max'])  # [B]
            ac_noisy = add_noise_snr(ac_clean, noise_batch, snr_db)  # [B, T]

            bc = torch.view_as_real(_stft(bc))
            ac_noisy = torch.view_as_real(_stft(ac_noisy))
            ac_clean = torch.view_as_real(_stft(ac_clean))

            optimizer.zero_grad()

            pred = model(ac_noisy, bc)

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

        # ── Epoch summary ──────────────────────────────────────────────
        train_loss = epoch_loss / max(n_batches, 1)
        
        val_loss = validate(model, val_loader, noise_iter, cfg, loss_fn, device)
        
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        scheduler.step(val_loss)
        lr_now = optimizer.param_groups[0]["lr"]
        
        
        print(f"Epoch {epoch}/{cfg['epochs']} | "
            f"train: {train_loss:.4f} | val: {val_loss:.4f} | "
            f"lr: {lr_now:.2e} | time: {elapsed:.1f}s"
        )

        if epoch % 5 == 0:
                eval_data(val_loader, noise_iter, cfg, metrics, dnsmos, device)
                print('Eval on data', metrics.compute(), f'DNSMOS: {dnsmos.compute().item()}')
                eval_model(model, val_loader, noise_iter, cfg, metrics, dnsmos, device)
                print('Eval on model', metrics.compute(), f'DNSMOS: {dnsmos.compute().item()}')

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
            # print(f"  ★ New best model saved (val_loss: {val_loss:.4f})")

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

    parser.add_argument("--check", type=int, default=None)
    parser.add_argument("--pin_memory", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--snr_min", type=int, default=None)
    parser.add_argument("--snr_max", type=int, default=None)
    parser.add_argument("--mode", type=str, default=None)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    cli_config = {}

    if args.batch_size is not None:
        cli_config["batch_size"] = args.batch_size

    if args.num_workers is not None:
        cli_config["num_workers"] = args.num_workers

    if args.mode is not None:
        cli_config["mode"] = args.mode
    
    if args.pin_memory is not None:
        cli_config["pin_memory"] = (args.pin_memory == 1)

    if args.snr_min is not None:
        cli_config["snr_min"] = args.snr_min

    if args.snr_max is not None:
        cli_config["snr_max"] = args.snr_max

    train(cli_config)