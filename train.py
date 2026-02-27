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
from util import _stft, _istft

from gtcrn_bc import GTCRN
from loss import HybridLoss
from dataset import create_dataloader
from pprint import pprint
import soundfile as sf
from torchmetrics.audio import (
    PerceptualEvaluationSpeechQuality,
    ShortTimeObjectiveIntelligibility,
    ScaleInvariantSignalNoiseRatio,
)
import warnings
warnings.filterwarnings('ignore')


# ── Default configuration ──────────────────────────────────────────────────

DEFAULT_CONFIG = dict(
    # Model
    model_type="gtcrn_bc",         # "gtcrn_bc" or "gtcrn" (AC-only baseline)

    # Dataset
    repo='verbreb/vibravox_16k_2s_subset',
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


    # Checkpointing
    save_dir="checkpoints",
    save_every=20,                  # save checkpoint every N epochs

    # Device
    device=None,                   # auto-detect if None
    mode='forehead',
    pin_memory=False
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

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def validate(model, val_loader, loss_fn, metrics: dict, device):
    """Run validation loop, return average loss."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    pbar = make_pbar(val_loader)
    
    for m in metrics.values():
        m.reset()

    with torch.no_grad():
        for batch in pbar:
            ac_noisy    = torch.view_as_real(_stft(batch['ac_noisy'].to(device)))
            bc          = torch.view_as_real(_stft(batch['bc'].to(device)))
            ac_clean    = torch.view_as_real(_stft(batch['ac_clean'].to(device)))

            pred = model(ac_noisy, bc)

            loss = loss_fn(pred, ac_clean).cpu()
            total_loss += loss.item()
            n_batches += 1
            
            if len(metrics) > 0:
                ac_clean = batch['ac_clean'].to(device)
                ac_pred  = _istft(pred)

                for m in metrics.values():
                    # (pred, target)
                    m.update(ac_pred, ac_clean)

    return total_loss / max(n_batches, 1)

def save_examples(epoch, model, val_examples, device):
    if not os.path.isdir('examples'):
        os.mkdir('examples')

    model.eval()

    ac_noisy    = val_examples['ac_noisy'].cpu()
    bc          = val_examples['bc'].cpu()
    ac_clean    = val_examples['ac_clean'].cpu()
    snr_db      = val_examples['snr_db'].cpu()
    
    ac_noisy_model_in   = torch.view_as_real(_stft(ac_noisy).to(device))
    bc_model_in         = torch.view_as_real(_stft(bc).to(device))

    pred = model(ac_noisy_model_in, bc_model_in)
    pred = torch.view_as_complex(pred)
    pred = _istft(pred)

    for i in range(min(pred.shape[0], 2)):
        f_name = f'examples/ep_{epoch}_b_{i}_AC_clean.wav'
        sf.write(f_name, ac_clean[i].detach().numpy(), samplerate=16000)
        f_name = f'examples/ep_{epoch}_b_{i}_AC_noisy_SNR_{snr_db[i]:.2f}.wav'
        sf.write(f_name, ac_noisy[i].detach().numpy(), samplerate=16000)
        f_name = f'examples/ep_{epoch}_b_{i}_BC.wav'
        sf.write(f_name, bc[i].detach().numpy(), samplerate=16000)
        f_name = f'examples/ep_{epoch}_b_{i}_pred.wav'
        sf.write(f_name, pred[i].detach().numpy(), samplerate=16000)


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
    model = GTCRN()

    model = model.to(device)
    total, trainable = count_parameters(model)
    print(f"Model: {cfg['model_type']} | Params: {total:,} total, {trainable:,} trainable")

    # pesq = PerceptualEvaluationSpeechQuality(16000, 'wb').to(device)
    stoi = ShortTimeObjectiveIntelligibility(16000).to(device)
    si_snr = ScaleInvariantSignalNoiseRatio().to(device)

    metrics = dict(stoi=stoi, si_snr=si_snr)
    # metrics = dict(pesq=pesq, stoi=stoi, si_snr=si_snr)

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
        pin_memory=cfg['pin_memory']
    )

    val_loader = create_dataloader(
        repo=cfg['repo'],
        split='test',
        mode=cfg['mode'],
        batch_size=cfg['batch_size'],
        num_workers=cfg['num_workers'],
        snr_range=cfg['snr_range'],
        pin_memory=cfg['pin_memory']
    )

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

    val_examples = next(iter(val_loader))

    for epoch in range(1, cfg["epochs"] + 1):
        
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()

        pbar = make_pbar(train_loader, total=len(train_loader), desc=f"Epoch {epoch}/{cfg['epochs']}")

        for batch in pbar:
            ac_noisy    = torch.view_as_real(_stft(batch['ac_noisy']).to(device))
            bc          = torch.view_as_real(_stft(batch['bc']).to(device))
            ac_clean    = torch.view_as_real(_stft(batch['ac_clean']).to(device))

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
        
        if epoch % 10 == 0:
            val_loss = validate(model, val_loader, loss_fn, metrics, device)
        else:
            val_loss = validate(model, val_loader, loss_fn, {}, device)
        
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        scheduler.step(val_loss)
        lr_now = optimizer.param_groups[0]["lr"]
        
        print(f"Epoch {epoch}/{cfg['epochs']} | "
              f"train: {train_loss:.4f} | val: {val_loss:.4f} | "
              f"lr: {lr_now:.2e} | time: {elapsed:.1f}s"
        )
        
        if epoch % 10 == 0:
            for k,v in metrics.items():
                print(f'{k}: {v.compute().cpu().item():.2f}', end=', ')
            print()

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
            save_examples(epoch, model, val_examples, device)

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

    if args.snr_min is not None or args.snr_max is not None:
        snr_min_new = args.snr_min if args.snr_min is not None else DEFAULT_CONFIG['snr_range'][0]
        snr_max_new = args.snr_max if args.snr_max is not None else DEFAULT_CONFIG['snr_range'][1]
        cli_config['snr_range'] = (snr_min_new, snr_max_new)

    if args.check is None:
        train(cli_config)
    else:
        print('Check')
        dl = create_dataloader(split='test')
        device = get_device()
        model = GTCRN()
        model = model.to(device)
        loss_fn = HybridLoss().to(device)

        pesq = PerceptualEvaluationSpeechQuality(16000, 'wb')
        stoi = ShortTimeObjectiveIntelligibility(16000)
        si_snr = ScaleInvariantSignalNoiseRatio()

        metrics = dict(
            pesq=pesq,
            stoi=stoi,
            si_snr=si_snr
        )


        # loss, metrics against bc
        print('AC vs BC')
        model.eval()
        total_loss = 0.0
        n_batches = 0
        pbar = make_pbar(dl)
        
        for m in metrics.values():
            m.reset()

        with torch.no_grad():
            for batch in pbar:
                ac_noisy    = torch.view_as_real(_stft(batch['ac_noisy']))
                bc          = torch.view_as_real(_stft(batch['bc']))
                ac_clean    = torch.view_as_real(_stft(batch['ac_clean']))


                loss = loss_fn(bc, ac_clean).cpu()
                total_loss += loss.item()
                n_batches += 1
                
                ac_clean = batch['ac_clean'].cpu()
                bc = batch['bc'].cpu()

                for m in metrics.values():
                    # (pred, target)
                    m.update(bc, ac_clean)

        print(f'loss: {total_loss / max(n_batches, 1)}')
        for k, v in metrics.items():
            print(f'{k}: {v.compute().item():.2f}')


        val_res = validate(model, dl, loss_fn, metrics, device)

        print(val_res)
        for k, v in metrics.items():
            print(f'{k}: {v.compute().item():.2f}')