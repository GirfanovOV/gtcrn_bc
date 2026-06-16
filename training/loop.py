from itertools import islice
from pathlib import Path
from pprint import pprint

import torch
import torch.nn as nn

from dataset import create_dataloader, create_dataloader_noise
from gtcrn_bc import GTCRN
from loss import HybridLoss
from util import get_device, infinite_loader, make_pbar

from .batch import make_noisy_batch, to_model_inputs
from .checkpoint import save_checkpoint
from .config import DATASET_REPOS, DEFAULT_CONFIG
from .diagnostics import component_grad_norms
from .logging import (
    add_loss_components,
    append_grad_log,
    append_loss_log,
    average_loss_components,
    format_loss_components,
    init_grad_log,
    init_loss_components,
    init_loss_log,
)


def limit_batches(loader, max_batches):
    if max_batches is None:
        return loader
    return islice(loader, max_batches)


def limited_len(loader, max_batches):
    if max_batches is None:
        return len(loader)
    return min(len(loader), max_batches)


def prepare_data(cfg):
    train_loader = create_dataloader(
        repo=cfg["dataset_repo"],
        split="train",
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        pin_memory=cfg["pin_memory"],
        audio_length_sec=cfg["audio_length_sec"],
        random_crop=True,
    )

    val_loader = create_dataloader(
        repo=cfg["dataset_repo"],
        split="test",
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        pin_memory=cfg["pin_memory"],
        audio_length_sec=cfg["audio_length_sec"],
        random_crop=False,
    )

    train_noise_loader = create_dataloader_noise(
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        pin_memory=cfg["pin_memory"],
    )
    val_noise_loader = create_dataloader_noise(
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        pin_memory=cfg["pin_memory"],
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
        loss_fn,
        epoch,
        grad_log_path,
    ):
    model.train()
    epoch_loss = 0.0
    component_sums = init_loss_components()
    n_batches = 0

    for batch in pbar:
        batch_idx = n_batches + 1
        grad_log_interval = cfg["grad_log_interval"]
        should_log_grad = (
            grad_log_path is not None
            and grad_log_interval is not None
            and grad_log_interval > 0
            and batch_idx % grad_log_interval == 0
        )
        ac_noisy, bc, ac_clean, lengths = make_noisy_batch(batch, noise_iter, cfg, device)
        ac_noisy, bc, ac_clean = to_model_inputs(ac_noisy, bc, ac_clean)

        optimizer.zero_grad()
        pred = model(ac_noisy, bc)
        if should_log_grad:
            loss, components, grad_components = loss_fn(
                pred,
                ac_clean,
                lengths,
                return_components=True,
                return_grad_components=True,
            )
            grad_norms = component_grad_norms(model, grad_components)
            lr_now = optimizer.param_groups[0]["lr"]
            total_batches = pbar.total or 0
            global_step = (epoch - 1) * total_batches + batch_idx if total_batches else batch_idx
            append_grad_log(grad_log_path, epoch, batch_idx, global_step, lr_now, grad_norms)
        else:
            loss, components = loss_fn(pred, ac_clean, lengths, return_components=True)
        loss.backward()

        if cfg["grad_clip"] > 0:
            nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])

        optimizer.step()
        epoch_loss += loss.item()
        add_loss_components(component_sums, components)
        n_batches += 1

        if (n_batches % 10 == 0) or (n_batches == 1):
            avg_components = average_loss_components(component_sums, n_batches)
            pbar.set_postfix({
                "avg_loss": f"{avg_components['total']:.4f}",
                "sisnr": f"{avg_components['sisnr']:.4f}",
            }, refresh=False)

    avg_components = average_loss_components(component_sums, n_batches)
    return epoch_loss / max(n_batches, 1), avg_components


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
    cfg = {**DEFAULT_CONFIG}
    if config:
        cfg.update(config)
    if cfg["dataset_repo"] not in DATASET_REPOS:
        raise ValueError(
            f"Unknown dataset_repo={cfg['dataset_repo']!r}. "
            f"Choose one of: {', '.join(DATASET_REPOS)}"
        )
    if not 2 <= cfg["audio_length_sec"] <= 8:
        raise ValueError("audio_length_sec must be in the [2, 8] second range")

    device = get_device(cfg["device"])
    print(f"Device: {device}")

    model = GTCRN()
    model = model.to(device)

    print("Train config:")
    pprint(cfg)

    train_loader, val_loader, train_noise_iter, val_noise_loader = prepare_data(cfg)
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    loss_fn = HybridLoss(sisnr_weight=cfg["sisnr_weight"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

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

        train_loss, train_components = train_epoch(
            pbar,
            cfg,
            train_noise_iter,
            device,
            model,
            optimizer,
            loss_fn,
            epoch,
            grad_log_path,
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

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if cfg["save_checkpoints"]:
                save_checkpoint(save_dir / "best_model.pt", epoch, model, optimizer, val_loss, cfg)

        if cfg["save_checkpoints"] and cfg["save_every"] > 0 and epoch % cfg["save_every"] == 0:
            save_checkpoint(save_dir / f"checkpoint_epoch{epoch}.pt", epoch, model, optimizer, val_loss, cfg)

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    if cfg["save_checkpoints"]:
        print(f"Best model saved to: {save_dir / 'best_model.pt'}")
    else:
        print("Checkpoint saving disabled.")

    return model
