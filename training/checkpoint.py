import torch


def save_checkpoint(path, epoch, model, optimizer, val_loss, cfg):
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "val_loss": val_loss,
        "config": cfg,
    }, path)
