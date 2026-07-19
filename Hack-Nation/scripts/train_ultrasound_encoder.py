"""Train the dual-head ovary/follicle segmentation model on USOVA3D.

Unlike ``scripts/train_ultrasound.py`` -- which evaluates the assembled heuristic
pipeline on synthetic phantoms and fits no weights -- this script trains a real
neural model on the real annotated volumes and writes a loadable checkpoint.

Protocol, driven by ``docs/ULTRASOUND_DATA_AUDIT.md``:

* volume-grouped split, persisted, no slice ever crossing a split boundary;
* validation used for early stopping and checkpoint selection;
* the test split is NOT touched here -- ``evaluate_ultrasound_encoder.py`` reads
  it exactly once;
* patch-based training with foreground-biased sampling, because uniform patches
  are mostly follicle-free.

Usage::

    python scripts/train_ultrasound_encoder.py --config configs/experiments/exp_usova3d_3d_unet.yaml
    python scripts/train_ultrasound_encoder.py --config <cfg> --smoke   # tiny CPU run
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
import yaml  # noqa: E402
from torch import optim  # noqa: E402

from ingestion.ultrasound.usova3d_dataset import (  # noqa: E402
    build_volume_split,
    discover_volumes,
    load_volume_arrays,
    write_split_manifest,
)
from models.ultrasound.dual_head_unet import DualHeadUNet  # noqa: E402
from models.ultrasound.torch_losses import DualHeadLoss  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent


def resolve(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def pad_to(array: np.ndarray, size: tuple[int, ...]) -> np.ndarray:
    """Zero-pad an array up to ``size`` on the trailing edge.

    USOVA3D volumes differ in shape -- in-plane extents run from 91x101 to
    162x208 -- so a fixed patch size does not fit inside every volume. Cropping
    to whatever fits would yield different patch shapes per volume, which cannot
    be stacked into a batch.

    Zero is the right pad value for both channels: intensity 0 is background
    (anechoic/off-field) and mask 0 is "no structure", so padding adds no false
    anatomy and no false supervision.
    """
    pad_width = [(0, max(size[i] - array.shape[i], 0)) for i in range(array.ndim)]
    if not any(after for _, after in pad_width):
        return array
    return np.pad(array, pad_width, mode="constant", constant_values=0)


def sample_patch_2d(
    image: np.ndarray,
    ovary: np.ndarray,
    follicle: np.ndarray,
    size: tuple[int, int],
    rng: np.random.Generator,
    *,
    force_foreground: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Crop a 2D patch from one slice of a volume.

    The slice is chosen first, then the in-plane crop. When foreground is forced
    the slice is drawn from those that actually contain a follicle -- 26% of
    slices contain none, and sampling those uniformly would starve the follicle
    head of positives even more severely than in 3D, since a 2D patch covers far
    less anatomy than a 3D one.
    """
    if force_foreground and follicle.any():
        slice_candidates = np.flatnonzero(follicle.reshape(follicle.shape[0], -1).any(axis=1))
        z = int(slice_candidates[rng.integers(len(slice_candidates))])
    else:
        z = int(rng.integers(image.shape[0]))

    plane, ovary_plane, follicle_plane = image[z], ovary[z], follicle[z]
    shape = plane.shape

    if force_foreground and follicle_plane.any():
        candidates = np.argwhere(follicle_plane > 0)
        centre = candidates[rng.integers(len(candidates))]
        start = [
            int(np.clip(centre[i] - size[i] // 2, 0, max(shape[i] - size[i], 0))) for i in range(2)
        ]
    else:
        start = [int(rng.integers(0, max(shape[i] - size[i], 0) + 1)) for i in range(2)]

    crop = tuple(slice(start[i], start[i] + size[i]) for i in range(2))
    # Pad rather than shrink: volumes differ in size and the batch must be uniform.
    return (
        pad_to(plane[crop], size),
        pad_to(ovary_plane[crop], size),
        pad_to(follicle_plane[crop], size),
    )


def sample_patch(
    image: np.ndarray,
    ovary: np.ndarray,
    follicle: np.ndarray,
    size: tuple[int, int, int],
    rng: np.random.Generator,
    *,
    force_foreground: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Crop a random patch, optionally centred on a follicle voxel.

    Foreground biasing matters more here than usual: follicle voxels are a small
    minority within an already-minority ovary, so uniform crops would leave the
    follicle head training almost entirely on negatives.
    """
    shape = image.shape

    if force_foreground and follicle.any():
        candidates = np.argwhere(follicle > 0)
        centre = candidates[rng.integers(len(candidates))]
        start = [
            int(np.clip(centre[i] - size[i] // 2, 0, max(shape[i] - size[i], 0))) for i in range(3)
        ]
    else:
        start = [int(rng.integers(0, max(shape[i] - size[i], 0) + 1)) for i in range(3)]

    slices = tuple(slice(start[i], start[i] + size[i]) for i in range(3))
    # Pad rather than shrink, for the same reason as the 2D sampler.
    return (
        pad_to(image[slices], size),
        pad_to(ovary[slices], size),
        pad_to(follicle[slices], size),
    )


def augment(
    image: np.ndarray,
    ovary: np.ndarray,
    follicle: np.ndarray,
    config: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Conservative intensity and flip augmentation.

    Only transforms that cannot create or destroy a follicle are applied.
    Rotation/scale/elastic are declared in config but intentionally not applied
    here: they require mask-consistent interpolation, and a nearest-neighbour
    mask resample at this voxel size can erase a small follicle outright.
    """
    if not config.get("enabled", True):
        return image, ovary, follicle

    for axis in config.get("flip_axes", []) or []:
        # Config declares axes for the 3D case (Z, Y, X). For a 2D patch the
        # in-plane axes are 0 and 1, so an axis of 2 refers to nothing and must
        # be skipped rather than raising.
        if axis >= image.ndim:
            continue
        if rng.random() < 0.5:
            image = np.flip(image, axis=axis)
            ovary = np.flip(ovary, axis=axis)
            follicle = np.flip(follicle, axis=axis)

    low, high = config.get("intensity_scale_range", [1.0, 1.0])
    image = image * float(rng.uniform(low, high))

    gamma_low, gamma_high = config.get("gamma_range", [1.0, 1.0])
    gamma = float(rng.uniform(gamma_low, gamma_high))
    image = np.clip(image, 0.0, 1.0) ** gamma

    noise_std = float(config.get("speckle_noise_std", 0.0))
    if noise_std > 0:
        # Multiplicative: ultrasound speckle is multiplicative, not additive.
        image = image * (1.0 + rng.normal(0.0, noise_std, image.shape))

    return (
        np.ascontiguousarray(np.clip(image, 0.0, 1.0), dtype=np.float32),
        np.ascontiguousarray(ovary, dtype=np.float32),
        np.ascontiguousarray(follicle, dtype=np.float32),
    )


@torch.no_grad()
def validate(
    model: DualHeadUNet,
    volumes: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    patch_size: tuple[int, int, int],
    device: torch.device,
    *,
    n_samples: int = 12,
) -> dict[str, float]:
    """Dice on validation volumes, averaged over several evenly-spaced samples.

    An earlier version scored ONE centred patch per volume. That was badly
    optimistic and, worse, nearly uninformative: it reported a follicle Dice of
    0.93 for a checkpoint whose held-out follicle Dice was 0.49, and it selected
    a model at epoch 4 that had barely trained. A single patch of a single slice
    is a sample of size one, and checkpoint selection against it is close to
    picking at random.

    Sampling ``n_samples`` positions spread through each volume keeps validation
    cheap enough to run every epoch while making the number mean something. The
    authoritative measurement is still the full sliding-window pass over the
    untouched test split in ``evaluate_ultrasound_encoder.py``.
    """
    model.eval()
    scores: dict[str, list[float]] = {"ovary": [], "follicle": []}
    is_2d = model.dims == 2

    def _score(x: np.ndarray, ovary_t: np.ndarray, follicle_t: np.ndarray) -> None:
        tensor = torch.from_numpy(x).float()[None, None].to(device)
        outputs = model(tensor)
        for name, target in (("ovary", ovary_t), ("follicle", follicle_t)):
            probs = torch.sigmoid(outputs[name])[0, 0].cpu().numpy()
            prediction = (probs > 0.5).astype(np.float32)
            denominator = prediction.sum() + target.sum()
            # An empty target with an empty prediction is a correct answer, and
            # scoring it 0.0 would drag the mean down for getting it right.
            dice = (
                1.0 if denominator == 0 else float(2.0 * (prediction * target).sum() / denominator)
            )
            scores[name].append(dice)

    for image, ovary, follicle in volumes.values():
        if is_2d:
            # Evenly-spaced slices across the whole volume, NOT the single best
            # follicle slice. Including follicle-free slices is the point: they
            # are 26% of the data and they are where over-segmentation shows up.
            indices = np.linspace(0, image.shape[0] - 1, num=n_samples, dtype=int)
            for z in indices:
                crop = tuple(
                    slice(
                        max((image.shape[i + 1] - patch_size[i]) // 2, 0),
                        max((image.shape[i + 1] - patch_size[i]) // 2, 0) + patch_size[i],
                    )
                    for i in range(2)
                )
                _score(
                    pad_to(image[z][crop], patch_size),
                    pad_to(ovary[z][crop], patch_size),
                    pad_to(follicle[z][crop], patch_size),
                )
        else:
            starts = np.linspace(
                0, max(image.shape[0] - patch_size[0], 0), num=n_samples, dtype=int
            )
            for z0 in starts:
                crop = (
                    slice(int(z0), int(z0) + patch_size[0]),
                    *(
                        slice(
                            max((image.shape[i] - patch_size[i]) // 2, 0),
                            max((image.shape[i] - patch_size[i]) // 2, 0) + patch_size[i],
                        )
                        for i in range(1, 3)
                    ),
                )
                _score(
                    pad_to(image[crop], patch_size),
                    pad_to(ovary[crop], patch_size),
                    pad_to(follicle[crop], patch_size),
                )

    model.train()
    return {
        "val_ovary_dice": float(np.mean(scores["ovary"])),
        "val_follicle_dice": float(np.mean(scores["follicle"])),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output.dir from the config, so a caller can redirect artifacts.",
    )
    parser.add_argument("--smoke", action="store_true", help="tiny CPU run for CI")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(resolve(args.config))

    training_cfg = config["training"]
    seed = int(training_cfg.get("seed", 42))
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # -- data ---------------------------------------------------------------
    volumes = discover_volumes(resolve(config["data"]["root"]))
    manifest_path = resolve(config["data"]["split_manifest"])
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        print(f"split: reusing {manifest_path.name}")
    else:
        manifest = build_volume_split(
            volumes,
            seed=int(config["split"]["seed"]),
            n_validation=int(config["split"]["n_validation"]),
            n_test=int(config["split"]["n_test"]),
        )
        write_split_manifest(
            manifest, manifest_path, created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        )
        print(f"split: wrote {manifest_path.name}")

    by_id = {volume.volume_id: volume for volume in volumes}
    annotator = config["data"].get("annotator", "r1")

    train_ids = list(manifest["train_ids"])
    val_ids = list(manifest["validation_ids"])
    if args.smoke:
        train_ids, val_ids = train_ids[:2], val_ids[:1]

    print(f"loading {len(train_ids)} train + {len(val_ids)} val volumes (annotator {annotator})")
    train_data = {vid: load_volume_arrays(by_id[vid], annotator=annotator) for vid in train_ids}
    val_data = {vid: load_volume_arrays(by_id[vid], annotator=annotator) for vid in val_ids}

    # -- model --------------------------------------------------------------
    model_cfg = dict(config["model"])
    if args.smoke:
        model_cfg["base_channels"] = 8
        model_cfg["depth"] = 2
    model = DualHeadUNet(**model_cfg).to(device)
    print(f"model: {model.n_parameters():,} parameters")

    criterion = DualHeadLoss(**config["loss"])
    optimizer = optim.Adam(
        model.parameters(),
        lr=float(training_cfg["learning_rate"]),
        weight_decay=float(training_cfg.get("weight_decay", 0.0)),
    )

    dims = int(model_cfg.get("dims", 3))
    patch_size = tuple(config["patch"]["size"])
    if len(patch_size) != dims:
        raise ValueError(
            f"model.dims={dims} but patch.size has {len(patch_size)} entries {patch_size}. "
            "The 2D and 3D configs must declare patches of matching rank."
        )
    if args.smoke:
        patch_size = (32, 32) if dims == 2 else (16, 32, 32)
    foreground_fraction = float(config["patch"].get("foreground_fraction", 0.5))
    batch_size = 1 if args.smoke else int(training_cfg["batch_size"])
    epochs = args.epochs or (2 if args.smoke else int(training_cfg["epochs"]))
    steps_per_epoch = 2 if args.smoke else max(len(train_ids) * 4, 8)

    output_dir = resolve(args.output_dir or config["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (output_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    history_path = output_dir / "training_history.jsonl"
    history_path.write_text("")

    monitor = training_cfg.get("monitor", "val_follicle_dice")
    best_score = -np.inf
    patience = int(training_cfg.get("early_stopping_patience", 25))
    since_improvement = 0

    print(f"\ntraining {epochs} epochs x {steps_per_epoch} steps, patch {patch_size}\n")
    for epoch in range(1, epochs + 1):
        epoch_losses: list[float] = []
        for _ in range(steps_per_epoch):
            images, ovaries, follicles = [], [], []
            for _ in range(batch_size):
                vid = train_ids[int(rng.integers(len(train_ids)))]
                image, ovary, follicle = train_data[vid]
                sampler = sample_patch_2d if dims == 2 else sample_patch
                patch = sampler(
                    image,
                    ovary,
                    follicle,
                    patch_size,
                    rng,
                    force_foreground=rng.random() < foreground_fraction,
                )
                patch = augment(*patch, config.get("augmentation", {}), rng)
                images.append(patch[0])
                ovaries.append(patch[1])
                follicles.append(patch[2])

            x = torch.from_numpy(np.stack(images)).float()[:, None].to(device)
            y_ovary = torch.from_numpy(np.stack(ovaries)).float()[:, None].to(device)
            y_follicle = torch.from_numpy(np.stack(follicles)).float()[:, None].to(device)

            optimizer.zero_grad()
            terms = criterion(model(x), y_ovary, y_follicle)
            terms["loss"].backward()
            optimizer.step()
            epoch_losses.append(float(terms["loss"].detach()))

        metrics = validate(model, val_data, patch_size, device)
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(epoch_losses)),
            **metrics,
        }
        with history_path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")

        score = metrics.get(monitor, -np.inf)
        marker = ""
        if score > best_score:
            best_score = score
            since_improvement = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_config": model_cfg,
                    "epoch": epoch,
                    "metrics": metrics,
                    "annotator": annotator,
                    "split_manifest": manifest,
                    "loss_config": config["loss"],
                    "inference_config": config.get("inference", {}),
                    "model_version": "ultrasound-usova3d-3d-v1",
                },
                output_dir / "checkpoint_best.pt",
            )
            marker = "  *best"
        else:
            since_improvement += 1

        print(
            f"epoch {epoch:3d}  loss {record['train_loss']:.4f}  "
            f"ovary {metrics['val_ovary_dice']:.4f}  "
            f"follicle {metrics['val_follicle_dice']:.4f}{marker}"
        )

        if since_improvement >= patience:
            print(f"\nearly stop: no {monitor} improvement in {patience} epochs")
            break

    torch.save(
        {"model_state": model.state_dict(), "model_config": model_cfg},
        output_dir / "checkpoint_last.pt",
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(
            {
                "best_" + monitor: float(best_score),
                "epochs_run": epoch,
                "n_train_volumes": len(train_ids),
                "n_val_volumes": len(val_ids),
                "annotator": annotator,
                "smoke": bool(args.smoke),
                "note": (
                    "Validation metrics only. Held-out TEST metrics come from "
                    "scripts/evaluate_ultrasound_encoder.py and are the reportable numbers."
                ),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nbest {monitor}: {best_score:.4f}")
    print(f"checkpoint -> {output_dir / 'checkpoint_best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
