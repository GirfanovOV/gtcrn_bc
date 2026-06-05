import argparse
import io
import re
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as F
from datasets import Audio, Dataset, DatasetDict, Features


TARGET_SAMPLE_RATE = 16000
SPLITS = ("train", "test")
RECORD_RE = re.compile(r"^Speaker\d+_C_\d+\.wav$")


def find_split_files(data_dir: Path, split: str) -> list[Path]:
    split_dir = data_dir / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory does not exist: {split_dir}")

    files = []
    for speaker_dir in sorted(split_dir.glob("Speaker*")):
        if not speaker_dir.is_dir():
            continue
        for wav_path in sorted(speaker_dir.glob("*.wav")):
            if RECORD_RE.fullmatch(wav_path.name):
                files.append(wav_path)

    if not files:
        raise RuntimeError(
            f"No matching files found in {split_dir}. "
            "Expected files like Speaker1_C_001.wav inside Speaker* directories."
        )

    return files


def wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, samples, sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def resample_channel(channel: np.ndarray, sample_rate: int, target_sample_rate: int) -> np.ndarray:
    if sample_rate == target_sample_rate:
        return channel.astype(np.float32, copy=False)

    waveform = torch.from_numpy(channel.astype(np.float32, copy=False))
    resampled = F.resample(waveform, sample_rate, target_sample_rate)
    return resampled.numpy()


def make_example(wav_path: Path, target_sample_rate: int) -> dict:
    audio, sample_rate = sf.read(wav_path, dtype="float32", always_2d=True)

    if audio.shape[1] != 2:
        raise ValueError(f"Expected exactly 2 channels in {wav_path}, got {audio.shape[1]}")

    headset = resample_channel(audio[:, 0], sample_rate, target_sample_rate)
    temple = resample_channel(audio[:, 1], sample_rate, target_sample_rate)

    return {
        "headset_microphone": {
            "bytes": wav_bytes(headset, target_sample_rate),
            "path": None,
        },
        "temple_vibration_pickup": {
            "bytes": wav_bytes(temple, target_sample_rate),
            "path": None,
        },
    }


def build_split(files: list[Path], target_sample_rate: int) -> Dataset:
    features = Features(
        {
            "headset_microphone": Audio(sampling_rate=target_sample_rate),
            "temple_vibration_pickup": Audio(sampling_rate=target_sample_rate),
        }
    )

    def examples(wav_files: list[str], sample_rate: int):
        paths = [Path(wav_file) for wav_file in wav_files]
        for wav_path in paths:
            yield make_example(wav_path, sample_rate)

    return Dataset.from_generator(
        examples,
        features=features,
        gen_kwargs={
            "wav_files": [str(wav_path) for wav_path in files],
            "sample_rate": target_sample_rate,
        },
    )


def build_dataset(data_dir: Path, target_sample_rate: int) -> DatasetDict:
    split_files = {split: find_split_files(data_dir, split) for split in SPLITS}
    for split, files in split_files.items():
        print(f"{split}: found {len(files)} C-record wav files")

    return DatasetDict(
        {
            split: build_split(files, target_sample_rate)
            for split, files in split_files.items()
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an ABCS Hugging Face dataset with Vibravox-like audio columns."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/ABCS_dataset"),
        help="Path containing train/test/Speaker*/Speaker*_C_*.wav",
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help="Hugging Face dataset repo id, e.g. username/abcs_16k_headset_temple",
    )
    parser.add_argument(
        "--target-sample-rate",
        type=int,
        default=TARGET_SAMPLE_RATE,
        help="Sample rate stored in the resulting Audio columns.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create/push the dataset repo as private.",
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
    if args.repo_id is None and not args.no_push:
        raise ValueError("--repo-id is required unless --no-push is passed")

    ds = build_dataset(args.data_dir, args.target_sample_rate)

    print(ds)
    preview = ds["train"]
    for column in ("headset_microphone", "temple_vibration_pickup"):
        preview = preview.cast_column(column, Audio(sampling_rate=args.target_sample_rate, decode=False))
    # print("First train row with decode=False:", preview[0])

    if args.save_to_disk is not None:
        ds.save_to_disk(args.save_to_disk)
        print(f"Saved dataset to {args.save_to_disk}")

    if not args.no_push:
        ds.push_to_hub(args.repo_id, private=args.private)
        print(f"Pushed dataset to https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
