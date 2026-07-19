"""USOVA3D volume discovery, subject-level splitting, and loading.

Everything in this module is built around one fact from
``docs/ULTRASOUND_DATA_AUDIT.md``: there are **16 volumes and no patient
identifier**. Two consequences shape the design.

**Grouping is by volume, and slices never cross a split.** Adjacent slices of one
volume are near-duplicates; a slice-level split would put a slice in training and
its neighbour in test and report a Dice that measures memorisation. The split is
therefore computed over volume IDs and slices are derived *afterwards*, from
whichever volumes a split contains.

**The grouping may still be imperfect, and we say so.** Since no patient key
exists, two volumes that are one patient's left and right ovary would land in
different splits undetected. That residual risk is recorded in the manifest
rather than hidden -- it is a property of the data, not a choice we can make
differently.

Both annotators are loaded. ``r1`` is the default target, but ``r2`` is always
available so evaluation can report against both; with exact follicle-count
agreement on only 5 of 16 volumes, a single-rater number overstates certainty.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "USOVA3D_ANNOTATORS",
    "SliceRecord",
    "Usova3dVolume",
    "build_volume_split",
    "discover_volumes",
    "load_slice",
    "load_volume_arrays",
    "write_split_manifest",
]

USOVA3D_ANNOTATORS = ("r1", "r2")

#: Binary masks are stored as {0, 255}; anything above this is foreground.
_MASK_THRESHOLD = 127


@dataclass(frozen=True)
class Usova3dVolume:
    """One annotated ovarian volume."""

    volume_id: str
    split_hint: str
    image_dir: Path
    label_dir: Path
    meta: dict[str, Any]

    @property
    def n_slices(self) -> int:
        return int(self.meta.get("n_slices", 0))

    @property
    def spacing_mm(self) -> tuple[float, float, float]:
        spacing = self.meta.get("spacing") or [1.0, 1.0, 1.0]
        return (float(spacing[0]), float(spacing[1]), float(spacing[2]))

    @property
    def spacing_is_calibrated(self) -> bool:
        """False when spacing is exactly unit isotropic.

        A spacing of exactly (1.0, 1.0, 1.0) is the conventional placeholder for
        "no calibration recorded". Treating it as a measurement would produce
        confident millimetre figures from an unknown scale, so every physical
        quantity derived from such a volume must be flagged.
        """
        return not all(abs(value - 1.0) < 1e-9 for value in self.spacing_mm)

    def slice_paths(self) -> list[Path]:
        return sorted(self.image_dir.glob("*.png"))

    def mask_dir(self, structure: str, annotator: str) -> Path:
        """Directory for ``structure`` ('ovary' | 'follicle') by ``annotator``."""
        return self.label_dir / f"{structure}_{annotator}"

    def instance_dir(self, annotator: str) -> Path:
        """Instance-ID follicle masks, which carry per-follicle integer labels."""
        return self.label_dir / f"follicle_{annotator}_labels"


@dataclass
class SliceRecord:
    """One slice with both structures' masks, as the training loop consumes it."""

    image: np.ndarray
    ovary_mask: np.ndarray
    follicle_mask: np.ndarray
    volume_id: str
    slice_index: int
    spacing_mm: tuple[float, float, float]
    source_path: str
    annotator: str
    spacing_is_calibrated: bool = True
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "image": self.image,
            "ovary_mask": self.ovary_mask,
            "follicle_mask": self.follicle_mask,
            "patient_id": self.volume_id,
            "study_id": self.volume_id,
            "laterality": None,
            "spacing_mm": self.spacing_mm,
            "source_path": self.source_path,
        }


def discover_volumes(root: str | Path) -> list[Usova3dVolume]:
    """Find every annotated volume under a USOVA3D root.

    Args:
        root: Directory containing ``train/``, ``val/`` and ``test/``.

    Returns:
        Volumes sorted by id, regardless of which pre-existing split directory
        they sit in -- the split we apply is computed here, not inherited from
        the directory layout.

    Raises:
        FileNotFoundError: If no volumes are found.
    """
    root = Path(root)
    volumes: list[Usova3dVolume] = []

    for split_dir in ("train", "val", "test"):
        images_root = root / split_dir / "images"
        labels_root = root / split_dir / "labels"
        if not images_root.is_dir():
            continue
        for image_dir in sorted(images_root.iterdir()):
            if not image_dir.is_dir():
                continue
            label_dir = labels_root / image_dir.name
            meta_path = label_dir / "meta.json"
            if not meta_path.exists():
                continue
            volumes.append(
                Usova3dVolume(
                    volume_id=image_dir.name,
                    split_hint=split_dir,
                    image_dir=image_dir,
                    label_dir=label_dir,
                    meta=json.loads(meta_path.read_text()),
                )
            )

    if not volumes:
        raise FileNotFoundError(
            f"No USOVA3D volumes with meta.json found under {root}. Expected "
            "<root>/<split>/images/Vol<N>/ and <root>/<split>/labels/Vol<N>/meta.json."
        )
    return sorted(volumes, key=lambda volume: volume.volume_id)


def dataset_checksum(volumes: list[Usova3dVolume]) -> str:
    """Stable digest of volume identity and geometry.

    Deliberately over the *manifest* (ids, shapes, spacing, slice counts) rather
    than over pixel bytes: hashing 3,419 PNGs on every run is slow, and this
    catches the failure that matters -- a split manifest being applied to a
    differently-constituted dataset.
    """
    digest = hashlib.sha256()
    for volume in sorted(volumes, key=lambda item: item.volume_id):
        digest.update(volume.volume_id.encode())
        digest.update(str(volume.meta.get("shape_zyx")).encode())
        digest.update(str(volume.meta.get("spacing")).encode())
        digest.update(str(volume.n_slices).encode())
    return digest.hexdigest()


def build_volume_split(
    volumes: list[Usova3dVolume],
    *,
    seed: int = 42,
    n_validation: int = 2,
    n_test: int = 2,
) -> dict[str, Any]:
    """Deterministic subject-level split over volume IDs.

    With 16 volumes the split is small enough that *which* volumes land in test
    materially changes the reported metric. The seed is therefore recorded in the
    manifest and the test set must be chosen once and never re-drawn to chase a
    better number.

    Args:
        volumes: All discovered volumes.
        seed: RNG seed, recorded in the manifest.
        n_validation: Volumes held out for early stopping.
        n_test: Volumes held out for final evaluation, touched once.

    Returns:
        A manifest dict.

    Raises:
        ValueError: If there are too few volumes to fill the requested splits.
    """
    ids = sorted(volume.volume_id for volume in volumes)
    if len(ids) < n_validation + n_test + 1:
        raise ValueError(
            f"Need at least {n_validation + n_test + 1} volumes to form a split, got {len(ids)}."
        )

    rng = np.random.default_rng(seed)
    # Cast back to plain str: numpy's str_ subclass serialises as a tagged scalar
    # in some JSON encoders, and the manifest must be readable by anything.
    shuffled = [str(item) for item in rng.permutation(ids)]
    test_ids = sorted(shuffled[:n_test])
    validation_ids = sorted(shuffled[n_test : n_test + n_validation])
    train_ids = sorted(shuffled[n_test + n_validation :])

    calibrated = {volume.volume_id: volume.spacing_is_calibrated for volume in volumes}

    return {
        "seed": seed,
        "grouping_key": "volume_id",
        "train_ids": train_ids,
        "validation_ids": validation_ids,
        "test_ids": test_ids,
        "dataset_checksum": dataset_checksum(volumes),
        "created_at": None,  # stamped by write_split_manifest
        "n_volumes": len(ids),
        "spacing_calibrated_by_volume": calibrated,
        "caveats": [
            "USOVA3D carries no patient identifier, so grouping is by VOLUME. If two "
            "volumes are the left and right ovary of one patient, they may fall in "
            "different splits. This cannot be detected from the data on disk.",
            "Volumes with spacing exactly (1.0, 1.0, 1.0) are treated as UNCALIBRATED; "
            "millimetre-denominated metrics are not reported for them.",
            "The test split must be evaluated once. Selecting hyperparameters against "
            "it would make the reported metric an optimistic in-sample number.",
        ],
    }


def write_split_manifest(manifest: dict[str, Any], path: str | Path, *, created_at: str) -> Path:
    """Persist a split manifest.

    Args:
        manifest: From :func:`build_volume_split`.
        path: Destination JSON path.
        created_at: ISO timestamp, passed in rather than read from the clock so
            the function stays deterministic and testable.
    """
    payload = {**manifest, "created_at": created_at}
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n")
    return destination


def _read_png(path: Path) -> np.ndarray:
    from PIL import Image  # noqa: PLC0415 - keeps pillow optional at import time

    return np.asarray(Image.open(path))


def load_slice(
    volume: Usova3dVolume,
    index: int,
    *,
    annotator: str = "r1",
    normalize: bool = True,
) -> SliceRecord:
    """Load one slice with its ovary and follicle masks.

    Args:
        volume: The source volume.
        index: Slice index.
        annotator: ``"r1"`` or ``"r2"``.
        normalize: Scale intensities to [0, 1] using the slice's own range.

    Returns:
        A :class:`SliceRecord`.

    Raises:
        ValueError: If an image and its mask disagree on shape, or values are
            not finite. A silent shape mismatch would train the model against
            misaligned supervision.
    """
    if annotator not in USOVA3D_ANNOTATORS:
        raise ValueError(f"Unknown annotator '{annotator}'; expected one of {USOVA3D_ANNOTATORS}.")

    paths = volume.slice_paths()
    if not 0 <= index < len(paths):
        raise IndexError(
            f"{volume.volume_id} has {len(paths)} slices; index {index} is out of range."
        )

    image_path = paths[index]
    name = image_path.name
    image = _read_png(image_path).astype(np.float32)

    warnings: list[str] = []
    masks: dict[str, np.ndarray] = {}
    for structure in ("ovary", "follicle"):
        mask_path = volume.mask_dir(structure, annotator) / name
        if not mask_path.exists():
            warnings.append(f"missing {structure} mask for {volume.volume_id}/{name}")
            masks[structure] = np.zeros(image.shape, dtype=np.float32)
            continue
        mask = _read_png(mask_path)
        if mask.shape != image.shape:
            raise ValueError(
                f"{volume.volume_id}/{name}: {structure} mask shape {mask.shape} does not "
                f"match image shape {image.shape}."
            )
        masks[structure] = (mask > _MASK_THRESHOLD).astype(np.float32)

    if not np.isfinite(image).all():
        raise ValueError(f"{image_path} contains non-finite values.")

    if normalize:
        spread = float(image.max() - image.min())
        image = (image - image.min()) / spread if spread > 1e-8 else np.zeros_like(image)

    if not volume.spacing_is_calibrated:
        warnings.append(
            f"{volume.volume_id} has placeholder spacing (1.0 mm isotropic); physical "
            "measurements from it are not reliable."
        )

    return SliceRecord(
        image=image,
        ovary_mask=masks["ovary"],
        follicle_mask=masks["follicle"],
        volume_id=volume.volume_id,
        slice_index=index,
        spacing_mm=volume.spacing_mm,
        source_path=str(image_path),
        annotator=annotator,
        spacing_is_calibrated=volume.spacing_is_calibrated,
        warnings=warnings,
    )


def load_volume_arrays(
    volume: Usova3dVolume,
    *,
    annotator: str = "r1",
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack a whole volume into ``(image, ovary, follicle)`` of shape ``(Z, Y, X)``."""
    records = [
        load_slice(volume, index, annotator=annotator, normalize=normalize)
        for index in range(len(volume.slice_paths()))
    ]
    if not records:
        raise ValueError(f"{volume.volume_id} has no slices.")
    return (
        np.stack([record.image for record in records]),
        np.stack([record.ovary_mask for record in records]),
        np.stack([record.follicle_mask for record in records]),
    )
