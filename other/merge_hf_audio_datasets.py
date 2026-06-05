import argparse
from pathlib import Path

from datasets import Audio, DatasetDict, concatenate_datasets, load_dataset


AUDIO_COLUMNS = ("headset_microphone", "temple_vibration_pickup")
SPLITS = ("train", "test")
TARGET_SAMPLE_RATE = 16000


def load_and_normalize(repo_id: str, sample_rate: int) -> DatasetDict:
    ds = load_dataset(repo_id)

    missing_splits = [split for split in SPLITS if split not in ds]
    if missing_splits:
        raise ValueError(f"{repo_id} is missing splits: {missing_splits}")

    for split in SPLITS:
        missing_columns = [column for column in AUDIO_COLUMNS if column not in ds[split].column_names]
        if missing_columns:
            raise ValueError(f"{repo_id}/{split} is missing columns: {missing_columns}")

        keep_columns = list(AUDIO_COLUMNS)
        drop_columns = [column for column in ds[split].column_names if column not in keep_columns]
        if drop_columns:
            ds[split] = ds[split].remove_columns(drop_columns)

        for column in AUDIO_COLUMNS:
            ds[split] = ds[split].cast_column(
                column,
                Audio(sampling_rate=sample_rate, decode=False),
            )

    return ds


def merge_datasets(source_repos: list[str], sample_rate: int) -> DatasetDict:
    datasets = [load_and_normalize(repo_id, sample_rate) for repo_id in source_repos]

    merged = DatasetDict()
    for split in SPLITS:
        merged[split] = concatenate_datasets([ds[split] for ds in datasets])
        for column in AUDIO_COLUMNS:
            merged[split] = merged[split].cast_column(
                column,
                Audio(sampling_rate=sample_rate),
            )

    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge HF audio datasets with headset_microphone/temple_vibration_pickup columns."
    )
    parser.add_argument(
        "--source-repo",
        action="append",
        required=True,
        help="Source dataset repo id. Pass this argument once per dataset.",
    )
    parser.add_argument(
        "--output-repo",
        default=None,
        help="Destination Hugging Face dataset repo id, e.g. username/merged_headset_temple.",
    )
    parser.add_argument(
        "--target-sample-rate",
        type=int,
        default=TARGET_SAMPLE_RATE,
        help="Sampling rate metadata for the resulting Audio columns.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create/push the output dataset as private.",
    )
    parser.add_argument(
        "--save-to-disk",
        type=Path,
        default=None,
        help="Optional local DatasetDict output directory before pushing.",
    )
    parser.add_argument(
        "--no-push",
        action="store_true",
        help="Build only; do not push to Hugging Face.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_repo is None and not args.no_push:
        raise ValueError("--output-repo is required unless --no-push is passed")

    merged = merge_datasets(args.source_repo, args.target_sample_rate)
    print(merged)

    preview = merged["train"]
    for column in AUDIO_COLUMNS:
        preview = preview.cast_column(
            column,
            Audio(sampling_rate=args.target_sample_rate, decode=False),
        )
    print("First train row with decode=False:", preview[0])

    if args.save_to_disk is not None:
        merged.save_to_disk(args.save_to_disk)
        print(f"Saved merged dataset to {args.save_to_disk}")

    if not args.no_push:
        merged.push_to_hub(args.output_repo, private=args.private)
        print(f"Pushed merged dataset to https://huggingface.co/datasets/{args.output_repo}")


if __name__ == "__main__":
    main()
