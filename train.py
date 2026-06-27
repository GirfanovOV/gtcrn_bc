import argparse
import warnings

from training.config import CHECK_CONFIG, DATASET_REPOS, DEFAULT_DATASET_REPO


warnings.filterwarnings("ignore")


def train(config=None):
    from training.loop import train as run_train

    return run_train(config)


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
    parser.add_argument("--val_batch_size", "--val-batch-size", type=int, default=None)
    parser.add_argument("--gradient_accum_steps", "--gradient-accum-steps", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument(
        "--no-train-shuffle",
        "--no_train_shuffle",
        dest="train_shuffle",
        action="store_false",
        default=None,
    )
    parser.add_argument(
        "--no-noise-shuffle",
        "--no_noise_shuffle",
        dest="noise_shuffle",
        action="store_false",
        default=None,
    )
    parser.add_argument("--snr_min", type=int, default=None)
    parser.add_argument("--snr_max", type=int, default=None)
    parser.add_argument("--noise_aware_coeff", "--noise-aware-coeff", type=float, default=None)
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
    parser.add_argument("--audio_length_sec", "--audio-length-sec", type=float, default=None)
    parser.add_argument(
        "--cache_audio_in_memory",
        "--cache-audio-in-memory",
        dest="cache_audio_in_memory",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        "--share_cached_audio",
        "--share-cached-audio",
        dest="share_cached_audio",
        action="store_true",
        default=None,
    )
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_val_batches", type=int, default=None)
    parser.add_argument("--sisnr_weight", "--sisnr-weight", type=float, default=None)
    parser.add_argument("--checkpoint_name", "--checkpoint-name", type=str, default=None)
    parser.add_argument("--grad_log_path", "--grad-log-path", type=str, default=None)
    parser.add_argument("--grad_log_interval", "--grad-log-interval", type=int, default=None)
    parser.add_argument(
        "--log_cuda_memory",
        "--log-cuda-memory",
        dest="log_cuda_memory",
        action="store_true",
        default=None,
    )

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
