"""Evaluate the trained segmentation model on the untouched held-out volumes.

This script reads the test split **once**. Nothing here may be used to select a
threshold, a checkpoint or a hyperparameter; doing so would turn the reported
figure into an in-sample number on a 2-volume test set.

Two reporting rules follow directly from the data audit:

**Every metric is reported against both annotators.** Exact follicle-count
agreement between the two experts is 5 of 16 volumes, and mean ovary-volume
disagreement is 16.5%. A single-rater score silently adopts one expert's
idiosyncrasies as truth; reporting both makes the irreducible uncertainty
visible, and the gap between the two is itself the most honest error bar
available.

**Millimetre-denominated metrics are withheld for uncalibrated volumes.** Ten of
sixteen volumes carry placeholder 1.0mm isotropic spacing. Reporting an ovarian
volume in mL for those would attach a confident unit to an unknown scale.

Per-volume metrics are always emitted alongside aggregates: with two test
volumes, a mean is barely a statistic and a single failure is invisible in it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
import yaml  # noqa: E402

from ingestion.ultrasound.usova3d_dataset import (  # noqa: E402
    discover_volumes,
    load_volume_arrays,
)
from models.ultrasound.dual_head_unet import DualHeadUNet  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
_EPS = 1e-9


def resolve(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def dice(prediction: np.ndarray, target: np.ndarray) -> float:
    """Dice, defined as 1.0 when both masks are empty."""
    denominator = prediction.sum() + target.sum()
    if denominator == 0:
        return 1.0
    return float(2.0 * (prediction * target).sum() / denominator)


def iou(prediction: np.ndarray, target: np.ndarray) -> float:
    union = np.logical_or(prediction, target).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(prediction, target).sum() / union)


@torch.no_grad()
def _slicewise_predict_2d(
    model: DualHeadUNet,
    image: np.ndarray,
    patch_size: tuple[int, int],
    device: torch.device,
    *,
    overlap: float = 0.25,
) -> dict[str, np.ndarray]:
    """Apply a 2D model slice-by-slice, tiling within each slice.

    The 2D model is the deployment-oriented one: clinically the input is a single
    B-mode frame. Evaluating it here on every slice of a held-out volume lets the
    2D and 3D models be compared on identical ground truth, which is the whole
    point of training both. Note this is NOT the same as clinical performance on
    a real 2D acquisition -- a resliced plane from a volume differs from a live
    frame in speckle texture and resolution anisotropy.
    """
    accum = {name: np.zeros(image.shape, dtype=np.float32) for name in ("ovary", "follicle")}
    counts = np.zeros(image.shape, dtype=np.float32)
    stride = [max(int(patch_size[i] * (1.0 - overlap)), 1) for i in range(2)]

    for z in range(image.shape[0]):
        plane = image[z]
        starts = [
            sorted(
                {
                    *range(0, max(plane.shape[i] - patch_size[i], 0) + 1, stride[i]),
                    max(plane.shape[i] - patch_size[i], 0),
                }
            )
            for i in range(2)
        ]
        for y in starts[0]:
            for x in starts[1]:
                window = (
                    slice(y, y + patch_size[0]),
                    slice(x, x + patch_size[1]),
                )
                crop = plane[window]
                # Pad short edges so every tile is the trained input size, then
                # discard the padding when writing back.
                padded = np.pad(
                    crop,
                    [(0, max(patch_size[i] - crop.shape[i], 0)) for i in range(2)],
                    mode="constant",
                )
                tensor = torch.from_numpy(padded).float()[None, None].to(device)
                outputs = model(tensor)
                for name in ("ovary", "follicle"):
                    probs = torch.sigmoid(outputs[name])[0, 0].cpu().numpy()
                    accum[name][z][window] += probs[: crop.shape[0], : crop.shape[1]]
                counts[z][window] += 1.0

    counts = np.maximum(counts, _EPS)
    return {name: accum[name] / counts for name in accum}


@torch.no_grad()
def sliding_window_predict(
    model: DualHeadUNet,
    image: np.ndarray,
    patch_size: tuple[int, int, int],
    device: torch.device,
    *,
    overlap: float = 0.25,
) -> dict[str, np.ndarray]:
    """Full-volume inference by averaging overlapping patch predictions.

    Averaging rather than tiling: hard patch boundaries produce visible seams in
    the probability map, and a follicle straddling a boundary gets split into two
    instances by the connected-component step downstream.
    """
    if model.dims == 2:
        return _slicewise_predict_2d(model, image, patch_size, device, overlap=overlap)

    patch_size = tuple(min(patch_size[i], image.shape[i]) for i in range(3))
    stride = [max(int(patch_size[i] * (1.0 - overlap)), 1) for i in range(3)]

    accum = {name: np.zeros(image.shape, dtype=np.float32) for name in ("ovary", "follicle")}
    counts = np.zeros(image.shape, dtype=np.float32)

    starts = [
        sorted(
            {
                *range(0, max(image.shape[i] - patch_size[i], 0) + 1, stride[i]),
                max(image.shape[i] - patch_size[i], 0),
            }
        )
        for i in range(3)
    ]

    for z in starts[0]:
        for y in starts[1]:
            for x in starts[2]:
                window = (
                    slice(z, z + patch_size[0]),
                    slice(y, y + patch_size[1]),
                    slice(x, x + patch_size[2]),
                )
                patch = torch.from_numpy(image[window]).float()[None, None].to(device)
                outputs = model(patch)
                for name in ("ovary", "follicle"):
                    accum[name][window] += torch.sigmoid(outputs[name])[0, 0].cpu().numpy()
                counts[window] += 1.0

    counts = np.maximum(counts, _EPS)
    return {name: accum[name] / counts for name in accum}


def extract_instances(
    follicle_mask: np.ndarray,
    ovary_mask: np.ndarray,
    spacing: tuple[float, float, float],
    *,
    min_diameter_mm: float | None,
) -> list[dict[str, Any]]:
    """Connected-component follicle instances, restricted to the ovary.

    ``min_diameter_mm`` is None for uncalibrated volumes, in which case NO size
    filter is applied and the instance list is explicitly not comparable to any
    published count threshold.
    """
    # Same opening + scale-free floor the encoder applies, so evaluation measures
    # what deployment will actually produce. Duplicating the constants here would
    # let the two drift apart silently.
    from models.ultrasound.trained_encoder import (  # noqa: PLC0415
        MIN_FOLLICLE_VOXELS,
        SPECKLE_OPENING_RADIUS,
        _ball,
    )

    inside = follicle_mask.astype(bool) & ovary_mask.astype(bool)
    inside = ndi.binary_opening(inside, structure=_ball(SPECKLE_OPENING_RADIUS, inside.ndim))
    labelled, n = ndi.label(inside)
    voxel_volume = float(np.prod(spacing))

    instances: list[dict[str, Any]] = []
    for index in range(1, n + 1):
        voxels = int((labelled == index).sum())
        volume_mm3 = voxels * voxel_volume if min_diameter_mm is not None else None
        diameter = (
            float((6.0 * volume_mm3 / np.pi) ** (1.0 / 3.0)) if volume_mm3 is not None else None
        )
        if min_diameter_mm is not None and diameter is not None:
            if diameter < min_diameter_mm:
                continue
        elif voxels < MIN_FOLLICLE_VOXELS:
            # Uncalibrated volume: apply the scale-free floor rather than keeping
            # every speck. This is the fix for the 649-instances-vs-8 result.
            continue
        centroid = ndi.center_of_mass(labelled == index)
        instances.append(
            {
                "instance_id": f"f{index}",
                "voxels": voxels,
                "centroid_voxel": [float(c) for c in centroid],
                "volume_mm3": volume_mm3,
                "equivalent_diameter_mm": diameter,
            }
        )
    return instances


def match_instances(
    predicted: list[dict[str, Any]],
    truth_labels: np.ndarray,
    predicted_mask: np.ndarray,
) -> dict[str, float]:
    """Instance precision/recall/F1 by centroid-in-ground-truth-instance matching.

    A predicted instance counts as a true positive when its centroid falls inside
    a ground-truth instance, and each ground-truth instance may be claimed once.
    Centroid matching rather than IoU: a follicle is small and roughly spherical,
    so a centroid hit is the operationally meaningful "found it", and IoU
    thresholds at this scale swing wildly on one voxel of boundary.
    """
    truth_ids = {int(i) for i in np.unique(truth_labels) if i != 0}
    matched: set[int] = set()
    true_positives = 0

    for instance in predicted:
        centroid = tuple(int(round(c)) for c in instance["centroid_voxel"])
        centroid = tuple(
            min(max(centroid[i], 0), truth_labels.shape[i] - 1) for i in range(len(centroid))
        )
        label = int(truth_labels[centroid])
        if label != 0 and label not in matched:
            matched.add(label)
            true_positives += 1

    n_pred = len(predicted)
    n_true = len(truth_ids)
    precision = true_positives / n_pred if n_pred else (1.0 if n_true == 0 else 0.0)
    recall = true_positives / n_true if n_true else (1.0 if n_pred == 0 else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "instance_precision": float(precision),
        "instance_recall": float(recall),
        "instance_f1": float(f1),
        "n_predicted": float(n_pred),
        "n_truth": float(n_true),
        "count_error": float(n_pred - n_true),
        "false_follicles_outside_ovary": float(
            np.logical_and(predicted_mask.astype(bool), ~truth_labels.astype(bool)).sum()
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", default="checkpoint_best.pt")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    experiment_dir = resolve(args.experiment_dir)

    checkpoint_path = experiment_dir / args.checkpoint
    if not checkpoint_path.exists():
        print(f"ERROR: no checkpoint at {checkpoint_path}", file=sys.stderr)
        return 1

    config = yaml.safe_load((experiment_dir / "config.resolved.yaml").read_text())
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualHeadUNet(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(f"loaded {checkpoint_path.name} (epoch {checkpoint.get('epoch')}) on {device}")

    manifest = checkpoint["split_manifest"]
    test_ids = list(manifest["test_ids"])
    print(f"held-out test volumes: {test_ids}\n")

    volumes = {v.volume_id: v for v in discover_volumes(resolve(config["data"]["root"]))}
    patch_size = tuple(config["patch"]["size"])
    overlap = float(config["patch"].get("sliding_window_overlap", 0.25))
    ovary_threshold = float(config.get("inference", {}).get("ovary_threshold", 0.5))
    follicle_threshold = float(config.get("inference", {}).get("follicle_threshold", 0.5))

    per_volume: list[dict[str, Any]] = []

    for volume_id in test_ids:
        volume = volumes[volume_id]
        image, _, _ = load_volume_arrays(volume, annotator="r1")
        probs = sliding_window_predict(model, image, patch_size, device, overlap=overlap)
        ovary_pred = (probs["ovary"] > ovary_threshold).astype(np.float32)
        follicle_pred = (probs["follicle"] > follicle_threshold).astype(np.float32)

        calibrated = volume.spacing_is_calibrated
        min_diameter = 2.0 if calibrated else None

        for annotator in ("r1", "r2"):
            _, ovary_true, follicle_true = load_volume_arrays(volume, annotator=annotator)

            instances = extract_instances(
                follicle_pred, ovary_pred, volume.spacing_mm, min_diameter_mm=min_diameter
            )
            truth_labels, _ = ndi.label(follicle_true.astype(bool) & ovary_true.astype(bool))

            record: dict[str, Any] = {
                "volume_id": volume_id,
                "annotator": annotator,
                "spacing_calibrated": calibrated,
                "ovary_dice": dice(ovary_pred, ovary_true),
                "ovary_iou": iou(ovary_pred, ovary_true),
                "follicle_dice": dice(follicle_pred, follicle_true),
                "follicle_iou": iou(follicle_pred, follicle_true),
                **match_instances(instances, truth_labels, follicle_pred),
            }

            if calibrated:
                voxel_ml = float(np.prod(volume.spacing_mm)) / 1000.0
                record["ovary_volume_ml_pred"] = float(ovary_pred.sum() * voxel_ml)
                record["ovary_volume_ml_true"] = float(ovary_true.sum() * voxel_ml)
                record["ovary_volume_ml_abs_error"] = abs(
                    record["ovary_volume_ml_pred"] - record["ovary_volume_ml_true"]
                )
            else:
                record["ovary_volume_ml_pred"] = None
                record["ovary_volume_ml_true"] = None
                record["note"] = "Placeholder spacing; millimetre-denominated metrics withheld."

            per_volume.append(record)
            print(
                f"{volume_id} [{annotator}]  ovary Dice {record['ovary_dice']:.4f}  "
                f"follicle Dice {record['follicle_dice']:.4f}  "
                f"count {int(record['n_predicted'])} vs {int(record['n_truth'])}  "
                f"F1 {record['instance_f1']:.3f}"
            )

    def aggregate(annotator: str, key: str) -> float:
        values = [r[key] for r in per_volume if r["annotator"] == annotator and r[key] is not None]
        return float(np.mean(values)) if values else float("nan")

    summary = {
        annotator: {
            "ovary_dice": aggregate(annotator, "ovary_dice"),
            "ovary_iou": aggregate(annotator, "ovary_iou"),
            "follicle_dice": aggregate(annotator, "follicle_dice"),
            "follicle_iou": aggregate(annotator, "follicle_iou"),
            "instance_precision": aggregate(annotator, "instance_precision"),
            "instance_recall": aggregate(annotator, "instance_recall"),
            "instance_f1": aggregate(annotator, "instance_f1"),
            "follicle_count_mae": float(
                np.mean([abs(r["count_error"]) for r in per_volume if r["annotator"] == annotator])
            ),
            "follicle_count_bias": float(
                np.mean([r["count_error"] for r in per_volume if r["annotator"] == annotator])
            ),
        }
        for annotator in ("r1", "r2")
    }

    output = {
        "test_volume_ids": test_ids,
        "n_test_volumes": len(test_ids),
        "summary_by_annotator": summary,
        "per_volume": per_volume,
        "caveats": [
            "Two held-out volumes. These are point estimates with no meaningful "
            "confidence interval; treat them as a smoke-level indication, not a "
            "validated performance claim.",
            "The two annotators agree on the exact follicle count in only 5 of 16 "
            "volumes overall, so the r1/r2 gap is a lower bound on label uncertainty.",
            "Millimetre metrics are withheld for volumes with placeholder spacing.",
            "This is an ovary/follicle morphology model. It is not a PCOS diagnostic "
            "model and has learned no interaction with clinical or temporal data.",
        ],
    }

    destination = experiment_dir / "test_metrics.json"
    destination.write_text(json.dumps(output, indent=2) + "\n")

    print("\n=== held-out summary ===")
    for annotator, metrics in summary.items():
        print(
            f"  [{annotator}] ovary Dice {metrics['ovary_dice']:.4f}  "
            f"follicle Dice {metrics['follicle_dice']:.4f}  "
            f"instance F1 {metrics['instance_f1']:.3f}  "
            f"count MAE {metrics['follicle_count_mae']:.2f}"
        )
    print(f"\nwrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
