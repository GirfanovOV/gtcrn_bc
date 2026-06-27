import argparse
import csv
from pathlib import Path


METRIC_NAMES = (
    "pesq",
    "stoi",
    "sisnr",
    "dnsmos_p808",
    "dnsmos_sig",
    "dnsmos_bak",
    "dnsmos_ovr",
)


CSV_FIELD_NAMES = ("index", "snr_db", *METRIC_NAMES)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate GTCRN-BC quality metrics on the default test split.")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Checkpoint path or name. Names are resolved in cwd, checkpoints/, and models/.",
    )
    parser.add_argument("--batch-size", "--batch_size", type=int, default=8)
    parser.add_argument("--num-workers", "--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-samples", "--max_samples", type=int, default=None)
    parser.add_argument("--max-batches", "--max_batches", type=int, default=None)
    parser.add_argument("--snr-min", "--snr_min", type=float, default=None)
    parser.add_argument("--snr-max", "--snr_max", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--audio-length-sec", "--audio_length_sec", type=float, default=None)
    parser.add_argument("--output-csv", "--output_csv", default=None)
    return parser.parse_args()


def resolve_checkpoint(value):
    raw = Path(value).expanduser()
    candidates = [raw]
    if raw.suffix == "":
        candidates.append(raw.with_suffix(".pt"))

    for base in (Path("checkpoints"), Path("models")):
        candidates.append(base / raw)
        if raw.suffix == "":
            candidates.append((base / raw).with_suffix(".pt"))

    for path in candidates:
        if path.exists():
            return path

    checked = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Checkpoint not found. Checked:\n  {checked}")


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model = GTCRN().to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, checkpoint if isinstance(checkpoint, dict) else {}


def limit_loader(loader, max_batches):
    if max_batches is None:
        yield from loader
        return

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break
        yield batch


def limited_len(loader, max_batches):
    if max_batches is None:
        return len(loader)
    return min(len(loader), max_batches)


def numpy_audio(x):
    return x.detach().cpu().to(torch.float32).numpy()


def make_eval_noisy_batch(batch, noise_iter, cfg, device):
    bc = batch["bc"].to(device)
    ac_clean = batch["ac_clean"].to(device)
    lengths = batch["lengths"].to(device)
    batch_size, length = ac_clean.shape

    noise = next(noise_iter)["noise"].to(device)
    noise = match_batch(noise, batch_size)
    noise = center_crop_1d(noise, length)

    snr_min = cfg["snr_min"]
    snr_max = cfg["snr_max"]
    if snr_min > snr_max:
        raise ValueError(f"snr_min must be <= snr_max, got {snr_min} > {snr_max}")
    if snr_min == snr_max:
        snr_db = torch.full((batch_size,), snr_min, device=device, dtype=ac_clean.dtype)
    else:
        snr_db = torch.empty(batch_size, device=device, dtype=ac_clean.dtype).uniform_(snr_min, snr_max)

    valid = torch.arange(length, device=device).unsqueeze(0) < lengths.unsqueeze(1)
    valid_f = valid.to(ac_clean.dtype)
    valid_count = lengths.clamp_min(1).to(ac_clean.dtype).unsqueeze(1)

    sig_pow = ac_clean.pow(2).mul(valid_f).sum(dim=1, keepdim=True) / valid_count
    noi_pow = noise.pow(2).mul(valid_f).sum(dim=1, keepdim=True) / valid_count + 1e-8
    snr_lin = (10.0 ** (snr_db / 10.0)).view(-1, 1)
    scale = torch.sqrt(sig_pow / (snr_lin * noi_pow + 1e-8))

    ac_noisy = (ac_clean + noise * scale) * valid_f
    return ac_noisy, bc, ac_clean, lengths, snr_db


def sisnr_db(estimate, target):
    estimate = estimate - estimate.mean()
    target = target - target.mean()
    target_energy = np.sum(target * target) + 1e-8
    projection = np.sum(estimate * target) * target / target_energy
    noise = estimate - projection
    return 10.0 * np.log10((np.sum(projection * projection) + 1e-8) / (np.sum(noise * noise) + 1e-8))


def make_pesq_fn():
    from pesq import NoUtterancesError, pesq

    def compute(reference, estimate):
        try:
            return float(pesq(TARGET_SAMPLE_RATE, reference, estimate, "wb"))
        except NoUtterancesError:
            return np.nan

    return compute


def make_stoi_fn():
    from pystoi import stoi

    def compute(reference, estimate):
        return float(stoi(reference, estimate, TARGET_SAMPLE_RATE, extended=False))

    return compute


def make_dnsmos_fn(device):
    from torchmetrics.audio import DeepNoiseSuppressionMeanOpinionScore

    metric = DeepNoiseSuppressionMeanOpinionScore(TARGET_SAMPLE_RATE, False).to(device)
    metric.eval()

    def compute(estimate):
        with torch.inference_mode():
            wav = torch.from_numpy(estimate).to(device=device, dtype=torch.float32)
            scores = metric(wav.unsqueeze(0))

        if isinstance(scores, dict):
            return {
                "dnsmos_p808": float(scores.get("p808_mos", np.nan)),
                "dnsmos_sig": float(scores.get("mos_sig", np.nan)),
                "dnsmos_bak": float(scores.get("mos_bak", np.nan)),
                "dnsmos_ovr": float(scores.get("mos_ovr", np.nan)),
            }

        values = scores.detach().cpu().flatten().numpy().astype(float)
        names = ("dnsmos_p808", "dnsmos_sig", "dnsmos_bak", "dnsmos_ovr")
        return {name: float(values[idx]) if idx < len(values) else np.nan for idx, name in enumerate(names)}

    return compute


def finite_mean(values):
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else np.nan


def write_csv(path, rows):
    if path is None:
        return

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELD_NAMES)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()

    global np, torch
    global TARGET_SAMPLE_RATE, create_dataloader, create_dataloader_noise
    global GTCRN, to_model_inputs, DEFAULT_CONFIG
    global _istft, get_device, infinite_loader, tqdm, match_batch, center_crop_1d

    print("Importing runtime dependencies...", flush=True)
    import numpy as np
    import torch

    from dataset import TARGET_SAMPLE_RATE, create_dataloader, create_dataloader_noise
    from gtcrn_bc import GTCRN
    from training.batch import to_model_inputs
    from training.config import DEFAULT_CONFIG
    from tqdm.auto import tqdm

    from util import _istft, center_crop_1d, get_device, infinite_loader, match_batch

    torch.manual_seed(args.seed)

    cfg = {**DEFAULT_CONFIG}
    cfg.update(
        batch_size=args.batch_size,
        val_batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=False,
        max_val_samples=args.max_samples,
    )
    if args.snr_min is not None:
        cfg["snr_min"] = args.snr_min
    if args.snr_max is not None:
        cfg["snr_max"] = args.snr_max
    if args.audio_length_sec is not None:
        cfg["audio_length_sec"] = args.audio_length_sec

    print("Resolving checkpoint and loading model...", flush=True)
    device = get_device(args.device)
    checkpoint_path = resolve_checkpoint(args.checkpoint)
    model, checkpoint = load_model(checkpoint_path, device)

    print(f"Device: {device}", flush=True)
    print(f"Checkpoint: {checkpoint_path}", flush=True)
    if checkpoint.get("epoch") is not None:
        print(f"Checkpoint epoch: {checkpoint['epoch']}", flush=True)
    print(f"Eval SNR range: [{cfg['snr_min']}, {cfg['snr_max']}] dB", flush=True)
    print(f"Random seed: {args.seed}", flush=True)

    print("Preparing test and noise dataloaders...", flush=True)
    test_loader = create_dataloader(
        repo=cfg["dataset_repo"],
        split="test",
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        pin_memory=cfg["pin_memory"],
        audio_length_sec=cfg["audio_length_sec"],
        random_crop=False,
        cache_in_memory=False,
        shuffle=False,
        max_samples=cfg["max_val_samples"],
    )
    noise_loader = create_dataloader_noise(
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        pin_memory=cfg["pin_memory"],
        shuffle=False,
    )
    noise_iter = infinite_loader(noise_loader)

    print("Loading PESQ, STOI, and DNSMOS metrics...", flush=True)
    pesq_fn = make_pesq_fn()
    stoi_fn = make_stoi_fn()
    dnsmos_fn = make_dnsmos_fn(device)

    rows = []
    totals = {name: [] for name in METRIC_NAMES}
    sample_index = 0
    pbar = tqdm(
        limit_loader(test_loader, args.max_batches),
        total=limited_len(test_loader, args.max_batches),
        desc="Evaluating",
        dynamic_ncols=True,
        mininterval=0.5,
        leave=True,
    )

    with torch.inference_mode():
        for batch in pbar:
            ac_noisy, bc, ac_clean, lengths, batch_snr_db = make_eval_noisy_batch(
                batch,
                noise_iter,
                cfg,
                device,
            )
            ac_noisy_stft, bc_stft = to_model_inputs(ac_noisy, bc)
            pred_stft = model(ac_noisy_stft, bc_stft)
            enhanced = _istft(pred_stft, length=int(lengths.max().item()))

            for idx in range(ac_clean.size(0)):
                length = int(lengths[idx].item())
                reference = numpy_audio(ac_clean[idx, :length])
                estimate = numpy_audio(enhanced[idx, :length])

                row = {
                    "index": sample_index,
                    "snr_db": float(batch_snr_db[idx].detach().cpu().item()),
                    "pesq": pesq_fn(reference, estimate),
                    "stoi": stoi_fn(reference, estimate),
                    "sisnr": sisnr_db(estimate, reference),
                }
                row.update(dnsmos_fn(estimate))

                for name in METRIC_NAMES:
                    totals[name].append(row[name])
                rows.append(row)
                sample_index += 1

            pbar.set_postfix(
                {
                    "pesq": f"{finite_mean(totals['pesq']):.3f}",
                    "stoi": f"{finite_mean(totals['stoi']):.3f}",
                    "sisnr": f"{finite_mean(totals['sisnr']):.2f}",
                },
                refresh=False,
            )

    write_csv(args.output_csv, rows)

    print("\nMetrics:")
    for name in METRIC_NAMES:
        value = finite_mean(totals[name])
        if np.isfinite(value):
            print(f"  {name}: {value:.4f}")
        else:
            print(f"  {name}: n/a")
    print(f"Samples: {sample_index}")
    if args.output_csv is not None:
        print(f"Per-sample CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
