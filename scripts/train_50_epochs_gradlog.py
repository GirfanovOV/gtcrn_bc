from training.config import DEFAULT_DATASET_REPO
from training.loop import train


if __name__ == "__main__":
    train({
        "dataset_repo": "verbreb/vibravox_16k_8s_headset_temple_full",
        "epochs": 50,
        "max_train_batches": 100,
        "sisnr_weight": 1.0,
        "grad_log_path": "checkpoints/grad_log_50epochs_100batches.csv",
        "grad_log_interval": 25,
        "save_checkpoints": False,
    })
