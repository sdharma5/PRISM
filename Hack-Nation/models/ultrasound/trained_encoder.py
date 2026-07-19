"""Lifecycle interface for the trained ovarian ultrasound encoder.

Implements ``fit`` / ``predict`` / ``evaluate`` / ``save`` / ``load`` /
``export_token`` over the dual-head U-Net, and is the object the orchestrator
injects in a production run.

**The heuristic is never a silent substitute.** ``models.ultrasound.encoder``
falls back to ``ThresholdSegmenter2D`` whenever torch is missing, which is right
for CI and wrong for inference: a caller who asked for the trained model and
received a scipy heuristic would get plausible numbers from a component that
learned nothing. :meth:`load` therefore raises on a missing checkpoint rather
than degrading, and selecting the heuristic requires naming it explicitly.

**Measurement semantics are enforced, not documented.** A 3D volume may populate
``follicle_count_per_ovary``; a single 2D frame may populate only
``follicle_count_per_section``; a cine loop may populate
``estimated_unique_follicle_count_from_cine`` and only once tracking exists.
These are different quantities against different clinical thresholds, and
:meth:`export_token` refuses to cross them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage as ndi

from schemas.modality_token import ModalityToken

__all__ = ["TrainedUltrasoundEncoder", "UltrasoundQualityGate"]

_CHECKPOINT_NAME = "checkpoint_best.pt"
_EPS = 1e-9

#: 2023 International Evidence-based Guideline: antral follicle counting is
#: defined over follicles of 2-10 mm. Below 2 mm a B-mode blob is not a
#: resolvable follicle. Applied in millimetres so the filter is scanner-
#: independent -- and therefore inapplicable when spacing is unknown.
MIN_FOLLICLE_DIAMETER_MM = 2.0
LARGE_STRUCTURE_DIAMETER_MM = 25.0

#: Scale-free speckle floor, applied even when spacing is unknown.
#:
#: The mm filter above is the clinically correct one, but it needs calibration.
#: When spacing is a placeholder, disabling size filtering ENTIRELY is far worse
#: than filtering approximately: on a held-out uncalibrated volume that produced
#: 649 "follicles" against 8 annotated, because every speckle speck survived as
#: its own connected component. A structure smaller than this many voxels is not
#: a resolvable follicle under any plausible calibration, so it is excluded and
#: the resulting count is still flagged as non-comparable to published thresholds.
MIN_FOLLICLE_VOXELS = 27  # ~3x3x3

#: Radius of the binary opening applied before labelling. Removes isolated
#: speckle and thin bridges that would otherwise merge or multiply instances.
SPECKLE_OPENING_RADIUS = 1


def _ball(radius: int, ndim: int) -> np.ndarray:
    """Binary structuring element approximating a disc/sphere of ``radius``."""
    radius = max(int(radius), 1)
    axes = np.ogrid[tuple(slice(-radius, radius + 1) for _ in range(ndim))]
    squared = np.zeros((1,) * ndim, dtype=float)
    for axis in axes:
        squared = squared + np.asarray(axis, dtype=float) ** 2
    return np.asarray(squared <= radius**2 + 1e-9)


@dataclass
class UltrasoundQualityGate:
    """Deterministic QC. Decides whether measurement is attempted at all.

    Deterministic rather than learned because there are no image-quality labels
    in USOVA3D; a learned QC head would be fitting noise and calling it quality.
    """

    min_ovary_voxel_fraction: float = 0.002
    max_ovary_voxel_fraction: float = 0.60
    min_mean_confidence: float = 0.30

    def assess(
        self,
        ovary_mask: np.ndarray,
        ovary_probs: np.ndarray,
        *,
        spacing_calibrated: bool,
    ) -> dict[str, Any]:
        """Return the gate's findings and whether measurement may proceed."""
        fraction = float(ovary_mask.mean()) if ovary_mask.size else 0.0
        confidence = float(ovary_probs[ovary_mask.astype(bool)].mean()) if ovary_mask.any() else 0.0

        reasons: list[str] = []
        if fraction < self.min_ovary_voxel_fraction:
            reasons.append(
                f"ovary occupies {fraction:.4f} of the volume, below the "
                f"{self.min_ovary_voxel_fraction} floor: no ovary reliably detected"
            )
        if fraction > self.max_ovary_voxel_fraction:
            # Without a ceiling, a structureless volume gets split in half and the
            # larger half measured as an ovary.
            reasons.append(
                f"ovary occupies {fraction:.4f} of the volume, above the "
                f"{self.max_ovary_voxel_fraction} ceiling: implausible segmentation"
            )
        if confidence < self.min_mean_confidence:
            reasons.append(f"mean ovary confidence {confidence:.3f} is below threshold")

        return {
            "ovary_detected": bool(ovary_mask.any()),
            "ovary_voxel_fraction": fraction,
            "mean_ovary_confidence": confidence,
            "pixel_spacing_available": spacing_calibrated,
            "measurement_feasible": not reasons,
            "failure_reasons": reasons,
        }


class TrainedUltrasoundEncoder:
    """The trained segmentation model, wrapped for the inference pipeline."""

    modality = "ovarian_ultrasound"

    def __init__(
        self,
        *,
        model: Any | None = None,
        model_config: dict[str, Any] | None = None,
        model_version: str = "ultrasound-usova3d-3d-v1",
        source_dataset: str = "USOVA3D",
        ovary_threshold: float = 0.5,
        follicle_threshold: float = 0.5,
        patch_size: tuple[int, int, int] = (32, 96, 96),
        quality_gate: UltrasoundQualityGate | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.model_config = model_config or {}
        self.model_version = model_version
        self.source_dataset = source_dataset
        self.ovary_threshold = ovary_threshold
        self.follicle_threshold = follicle_threshold
        self.patch_size = patch_size
        self.quality_gate = quality_gate or UltrasoundQualityGate()
        self.metrics = metrics or {}

    # -- lifecycle ---------------------------------------------------------

    def fit(self, *args: Any, **kwargs: Any) -> TrainedUltrasoundEncoder:
        """Training lives in ``scripts/train_ultrasound_encoder.py``.

        Deliberately not implemented here: training needs a split manifest,
        augmentation, checkpointing and early stopping, and burying that inside
        an inference-time object would make it tempting to fit on whatever data
        happens to be at hand -- including a test volume.
        """
        raise NotImplementedError(
            "Use scripts/train_ultrasound_encoder.py --config "
            "configs/experiments/exp_usova3d_3d_unet.yaml. Training requires a "
            "persisted volume-grouped split, which this object does not manage."
        )

    @classmethod
    def load(cls, path: str | Path, **overrides: Any) -> TrainedUltrasoundEncoder:
        """Load a trained checkpoint.

        Args:
            path: Checkpoint file, or an experiment directory containing one.
            **overrides: Constructor overrides.

        Raises:
            FileNotFoundError: If no checkpoint exists. This is intentionally a
                hard error -- see the module docstring on silent fallback.
        """
        import torch  # noqa: PLC0415

        from models.ultrasound.dual_head_unet import DualHeadUNet  # noqa: PLC0415

        candidate = Path(path)
        if candidate.is_dir():
            candidate = candidate / _CHECKPOINT_NAME
        if not candidate.exists():
            raise FileNotFoundError(
                f"No trained ultrasound checkpoint at {candidate}. Train one with "
                "scripts/train_ultrasound_encoder.py. The heuristic segmenter is NOT "
                "an automatic substitute; select it explicitly if that is what you want."
            )

        checkpoint = torch.load(candidate, map_location="cpu", weights_only=False)
        model = DualHeadUNet(**checkpoint["model_config"])
        model.load_state_dict(checkpoint["model_state"])
        model.eval()

        inference_cfg = checkpoint.get("inference_config", {}) or {}
        return cls(
            model=model,
            model_config=checkpoint["model_config"],
            model_version=checkpoint.get("model_version", "ultrasound-usova3d-3d-v1"),
            ovary_threshold=float(inference_cfg.get("ovary_threshold", 0.5)),
            follicle_threshold=float(inference_cfg.get("follicle_threshold", 0.5)),
            metrics=checkpoint.get("metrics", {}),
            **overrides,
        )

    def save(self, directory: str | Path) -> Path:
        """Persist weights plus everything needed to reproduce inference."""
        import torch  # noqa: PLC0415

        if self.model is None:
            raise RuntimeError("No model to save.")

        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "model_config": self.model_config,
                "model_version": self.model_version,
                "inference_config": {
                    "ovary_threshold": self.ovary_threshold,
                    "follicle_threshold": self.follicle_threshold,
                },
                "metrics": self.metrics,
            },
            destination / _CHECKPOINT_NAME,
        )
        (destination / "encoder_manifest.json").write_text(
            json.dumps(
                {
                    "model_version": self.model_version,
                    "source_dataset": self.source_dataset,
                    "model_config": self.model_config,
                    "thresholds": {
                        "ovary": self.ovary_threshold,
                        "follicle": self.follicle_threshold,
                    },
                    "label_definitions": {
                        "ovary": "ovarian stroma including follicle lumina",
                        "follicle": "anechoic follicle inside the ovary",
                    },
                    "min_follicle_diameter_mm": MIN_FOLLICLE_DIAMETER_MM,
                    "metrics": self.metrics,
                },
                indent=2,
            )
            + "\n"
        )
        return destination / _CHECKPOINT_NAME

    # -- prediction --------------------------------------------------------

    def predict(
        self, volume: np.ndarray, *, spacing_mm: tuple[float, float, float] | None = None
    ) -> dict[str, Any]:
        """Segment one volume or frame.

        Returns:
            Probability maps, binary masks and the QC assessment.
        """
        import torch  # noqa: PLC0415

        if self.model is None:
            raise RuntimeError("Encoder has no model. Use TrainedUltrasoundEncoder.load().")

        array = np.asarray(volume, dtype=np.float32)
        spread = float(array.max() - array.min())
        array = (array - array.min()) / spread if spread > 1e-8 else np.zeros_like(array)

        is_2d = array.ndim == 2
        tensor_input = array[None] if is_2d else array

        with torch.no_grad():
            x = torch.from_numpy(tensor_input).float()[None, None]
            outputs = self.model(x)
            probs = {
                name: torch.sigmoid(outputs[name])[0, 0].numpy() for name in ("ovary", "follicle")
            }

        if is_2d:
            probs = {name: value[0] for name, value in probs.items()}

        ovary_mask = (probs["ovary"] > self.ovary_threshold).astype(np.uint8)
        follicle_mask = (probs["follicle"] > self.follicle_threshold).astype(np.uint8)
        # Anatomical constraint applied at inference too, not only in the loss.
        follicle_mask = follicle_mask * ovary_mask

        calibrated = spacing_mm is not None and not all(
            abs(value - 1.0) < 1e-9 for value in spacing_mm
        )
        quality = self.quality_gate.assess(
            ovary_mask, probs["ovary"], spacing_calibrated=calibrated
        )

        return {
            "probs": probs,
            "ovary_mask": ovary_mask,
            "follicle_mask": follicle_mask,
            "quality": quality,
            "spacing_mm": spacing_mm,
            "spacing_calibrated": calibrated,
            "acquisition_mode": "single_frame" if is_2d else "volume_3d",
        }

    def extract_instances(
        self,
        follicle_mask: np.ndarray,
        ovary_mask: np.ndarray,
        spacing_mm: tuple[float, float, float] | None,
        follicle_probs: np.ndarray | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Connected-component instances.

        Returns ``(retained, all_raw)``. Raw instances are kept even when
        excluded from the count, so an excluded structure remains inspectable
        rather than silently vanishing.
        """
        inside = follicle_mask.astype(bool) & ovary_mask.astype(bool)
        # Opening BEFORE labelling: a single-voxel speckle bridge either creates a
        # spurious instance or fuses two real ones, and connected components has
        # no way to recover from either afterwards.
        inside = ndi.binary_opening(inside, structure=_ball(SPECKLE_OPENING_RADIUS, inside.ndim))
        labelled, n = ndi.label(inside)

        calibrated = spacing_mm is not None and not all(
            abs(value - 1.0) < 1e-9 for value in spacing_mm
        )
        voxel_volume = float(np.prod(spacing_mm)) if calibrated else None

        raw: list[dict[str, Any]] = []
        retained: list[dict[str, Any]] = []

        for index in range(1, n + 1):
            component = labelled == index
            voxels = int(component.sum())
            volume_mm3 = voxels * voxel_volume if voxel_volume is not None else None
            diameter = (
                float((6.0 * volume_mm3 / np.pi) ** (1.0 / 3.0)) if volume_mm3 is not None else None
            )
            centroid = ndi.center_of_mass(component)
            confidence = (
                float(follicle_probs[component].mean()) if follicle_probs is not None else 0.0
            )

            instance = {
                "instance_id": f"f{index}",
                "centroid_voxel": tuple(float(c) for c in centroid),
                "centroid_mm": (
                    tuple(float(c) * s for c, s in zip(centroid, spacing_mm, strict=True))
                    if calibrated
                    else None
                ),
                "volume_mm3": volume_mm3,
                "equivalent_diameter_mm": diameter,
                "voxels": voxels,
                "confidence": confidence,
                "excluded_reason": None,
            }
            raw.append(instance)

            if diameter is None:
                # No calibration, so the mm filter cannot be applied. Fall back to
                # the scale-free voxel floor rather than retaining everything --
                # "unknown scale" is not "no structure is too small". The count is
                # still flagged as non-comparable on the token.
                if voxels < MIN_FOLLICLE_VOXELS:
                    instance["excluded_reason"] = (
                        f"below {MIN_FOLLICLE_VOXELS} voxels (scale-free speckle floor; "
                        "spacing unknown so the mm filter could not be applied)"
                    )
                else:
                    retained.append(instance)
            elif diameter < MIN_FOLLICLE_DIAMETER_MM:
                instance["excluded_reason"] = f"below {MIN_FOLLICLE_DIAMETER_MM} mm"
            elif diameter > LARGE_STRUCTURE_DIAMETER_MM:
                instance["excluded_reason"] = (
                    f"above {LARGE_STRUCTURE_DIAMETER_MM} mm: flagged large structure, "
                    "excluded from the follicle count and given no pathology name"
                )
            else:
                retained.append(instance)

        return retained, raw

    # -- token export ------------------------------------------------------

    def export_token(self, payload: Any, *, patient_id: str) -> ModalityToken:
        """Encode one ultrasound study into a :class:`ModalityToken`.

        Args:
            payload: An array, or an object exposing ``pixels`` and ``metadata``.
            patient_id: Patient identifier.

        Raises:
            ValueError: For an unsupported acquisition rank.
        """
        volume, spacing_mm, laterality = _unpack(payload)
        result = self.predict(volume, spacing_mm=spacing_mm)

        quality = result["quality"]
        mode = result["acquisition_mode"]
        warnings: list[str] = ["Research model; clinician confirmation required"]
        missing: list[str] = []

        structured: dict[str, Any] = {
            "input_type": "3d_volume" if mode == "volume_3d" else "2d_frame",
            "laterality": laterality,
            "acquisition_mode": mode,
            "measurement_feasible": quality["measurement_feasible"],
            "ovary_detected": quality["ovary_detected"],
            "ovary_voxel_fraction": quality["ovary_voxel_fraction"],
            "large_structure_flag": False,
        }

        if not quality["measurement_feasible"]:
            # Abstain rather than invent. Every measurement field is null and the
            # reasons travel with the token.
            structured.update(
                {
                    "follicle_count_per_ovary": None,
                    "follicle_count_per_section": None,
                    "estimated_unique_follicle_count_from_cine": None,
                    "ovary_volume_ml": None,
                    "mean_follicle_diameter_mm": None,
                }
            )
            warnings.extend(quality["failure_reasons"])
            warnings.append("Quality gate failed: no measurements are reported.")
            missing = ["follicle_count", "ovary_volume_ml"]
            return ModalityToken(
                patient_id=patient_id,
                modality="ovarian_ultrasound",
                structured_features=structured,
                quality_score=0.0,
                confidence_score=0.0,
                model_version=self.model_version,
                source_dataset=self.source_dataset,
                missing_fields=missing,
                warnings=warnings,
            )

        retained, raw = self.extract_instances(
            result["follicle_mask"],
            result["ovary_mask"],
            spacing_mm,
            result["probs"]["follicle"],
        )
        calibrated = result["spacing_calibrated"]

        # THE measurement-semantics guard: a single frame is a cross-section and
        # may never populate a per-ovary count, which is what the PCOM threshold
        # is defined against.
        count = len(retained)
        if mode == "volume_3d":
            structured["follicle_count_per_ovary"] = count
            structured["follicle_count_per_section"] = None
        else:
            structured["follicle_count_per_ovary"] = None
            structured["follicle_count_per_section"] = count
            warnings.append(
                "Single frame: only a per-section count is reported. It is NOT an "
                "antral follicle count and must not be compared to a per-ovary threshold."
            )
        structured["estimated_unique_follicle_count_from_cine"] = None

        diameters = [
            item["equivalent_diameter_mm"]
            for item in retained
            if item["equivalent_diameter_mm"] is not None
        ]
        structured["mean_follicle_diameter_mm"] = float(np.mean(diameters)) if diameters else None
        structured["large_structure_flag"] = any(
            item["excluded_reason"] and "above" in item["excluded_reason"] for item in raw
        )
        structured["n_raw_instances"] = len(raw)

        if calibrated and mode == "volume_3d":
            voxel_ml = float(np.prod(spacing_mm)) / 1000.0
            structured["ovary_volume_ml"] = float(result["ovary_mask"].sum() * voxel_ml)
        else:
            structured["ovary_volume_ml"] = None
            if mode == "volume_3d":
                missing.append("ovary_volume_ml")

        if not calibrated:
            missing.append("spacing_mm")
            warnings.append(
                "Pixel spacing is unknown or a placeholder: the "
                f"{MIN_FOLLICLE_DIAMETER_MM} mm minimum-size filter could NOT be applied, "
                "so this count includes structures below follicle size and is not "
                "comparable to any published threshold."
            )

        confidence = float(quality["mean_ovary_confidence"])
        return ModalityToken(
            patient_id=patient_id,
            modality="ovarian_ultrasound",
            structured_features=structured,
            quality_score=round(min(max(quality["ovary_voxel_fraction"] * 5.0, 0.0), 1.0), 4),
            confidence_score=round(min(max(confidence, 0.0), 1.0), 4),
            model_version=self.model_version,
            source_dataset=self.source_dataset,
            missing_fields=sorted(set(missing)),
            warnings=warnings,
        )


def _unpack(payload: Any) -> tuple[np.ndarray, tuple[float, float, float] | None, str | None]:
    """Accept an array, an UltrasoundInput, or a list of them."""
    if isinstance(payload, list):
        if not payload:
            raise ValueError("Empty ultrasound input list.")
        payload = payload[0]

    if isinstance(payload, np.ndarray):
        return payload, None, None

    pixels = getattr(payload, "pixels", None)
    metadata = getattr(payload, "metadata", None)
    if pixels is None:
        raise TypeError(
            f"Unsupported ultrasound payload {type(payload).__name__}; expected an array "
            "or an object with .pixels and .metadata."
        )
    spacing = getattr(metadata, "spacing_mm", None) if metadata is not None else None
    laterality = getattr(metadata, "laterality", None) if metadata is not None else None
    return np.asarray(pixels), spacing, laterality
