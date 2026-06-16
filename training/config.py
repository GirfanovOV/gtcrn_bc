DATASET_REPOS = (
    "verbreb/vibravox_abcs_merge_headset_temple_16k",
    "verbreb/abcs_16k_headset_temple",
    "verbreb/vibravox_16k_8s_headset_temple_full",
)
DEFAULT_DATASET_REPO = "verbreb/vibravox_16k_8s_headset_temple_full"

DEFAULT_CONFIG = dict(
    # Training
    batch_size=64,
    lr=1e-3,
    epochs=100,
    grad_clip=5.0,
    max_train_batches=None,
    max_val_batches=None,
    sisnr_weight=1.0,

    # Data limits (set to None for full dataset)
    dataset_repo=DEFAULT_DATASET_REPO,
    max_train_samples=None,        # e.g. 2000 for quick test
    max_val_samples=None,          # e.g. 500 for quick test
    audio_length_sec=8,
    num_workers=4,                 # 0 for Mac, 2-4 for Colab

    snr_min=-5,
    snr_max=15,
    val_deterministic=True,
    val_snr_db=None,

    # Checkpointing
    save_dir="checkpoints",
    save_every=20,                 # save checkpoint every N epochs
    save_checkpoints=True,
    loss_log_path="checkpoints/loss_log.csv",
    grad_log_path=None,            # optional CSV path for gradient norm logging
    grad_log_interval=25,          # log gradient norms every N train batches

    # Device
    device=None,                   # auto-detect if None
    pin_memory=True,
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
