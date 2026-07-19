#!/usr/bin/env python
"""Run the ultrasound morphology pipeline end to end and score it.

With no dataset present this runs on synthetic phantoms, which is the only mode
CI uses. That is not a limitation to work around — the phantoms carry exact
ground-truth per-section counts, unique follicle counts, frame spans, diameters
and ovarian volumes, so this script reports genuine absolute error rather than
"it completed".

**2D is the primary pathway and is evaluated first.** The three acquisition
classes are scored separately and their count metrics are never pooled:

* ``single_frame`` -> per-section count MAE, ovary area error
* ``cine_loop``    -> unique-track count MAE, tracking fragmentation and merge
                      rates, per-section MAE on the representative frame
* ``volume_3d``    -> true per-ovary count MAE, ovarian volume error

A per-section count and a per-ovary count are different physical quantities on
different supports. Averaging their errors would produce a number describing
nothing, and would hide the most informative failure: a model that reads frames
well but tracks badly.

It also runs the quality gate against deliberately unmeasurable studies (noise
frames and volumes, and studies with the spacing withheld) to check the pipeline
abstains rather than producing a confident number.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running as a script from a checkout without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from typing import Any

import numpy as np
import yaml

from evaluation.ultrasound import (
    diameter_error,
    evaluate_segmentation,
    match_tracks_to_truth,
    ovarian_volume_absolute_error,
    per_section_count_mae,
    quality_gate_sensitivity,
    quality_gate_unsafe_acceptance_rate,
    tracking_fragmentation_and_merge,
    unique_track_count_mae,
)
from ingestion.ultrasound.loader import load_ultrasound
from models.ultrasound.encoder import UltrasoundEncoder
from schemas.model_output import ExperimentResult, FoldMetrics
from scripts._cli import (
    add_deprecated_alias,
    add_standard_arguments,
    make_parser,
    resolve_output_dir,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT = REPO_ROOT / "configs" / "experiments" / "exp_ultrasound.yaml"


def load_yaml(path: Path) -> dict[str, Any]:
    with Path(path).open() as fh:
        return yaml.safe_load(fh) or {}


def _resolve(path_like: str) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else REPO_ROOT / path


def _load(image: np.ndarray, *, mode: str, spacing, index: int, prefix: str):
    """Load one synthetic study with its acquisition mode declared."""
    return load_ultrasound(
        image,
        patient_id=f"{prefix}{index:03d}",
        study_id=f"{prefix}{index:03d}",
        spacing_mm=spacing,
        laterality="left" if index % 2 == 0 else "right",
        route="transvaginal",
        source_dataset="synthetic_phantom",
        acquisition_mode=mode,
    )


# ---------------------------------------------------------------------------
# Stage 1 / 2 surrogate: 2D single-frame cohort (the primary pathway)
# ---------------------------------------------------------------------------


def run_single_frame_cohort(
    encoder: UltrasoundEncoder, *, n_studies: int, seed: int, settings: dict[str, Any]
) -> dict[str, Any]:
    """Score the single-frame path: per-section counts and cross-sectional area."""
    from tests.fixtures.synthetic_ultrasound import make_phantom_2d  # noqa: PLC0415

    frame_cfg = settings.get("frame", {}) or {}
    spacing_2d = tuple(frame_cfg.get("pixel_spacing_mm", (0.35, 0.35)))
    spacing = (spacing_2d[0], spacing_2d[0], spacing_2d[1])

    pred_counts: list[int | None] = []
    true_counts: list[int] = []
    area_errors: list[float] = []
    diameter_scores: list[dict[str, float]] = []
    per_study: list[dict[str, float]] = []
    tokens: list[dict[str, Any]] = []

    for index in range(n_studies):
        phantom = make_phantom_2d(
            shape=tuple(frame_cfg.get("shape", (128, 128))),  # type: ignore[arg-type]
            pixel_spacing_mm=spacing_2d,  # type: ignore[arg-type]
            semi_axes_mm=tuple(frame_cfg.get("semi_axes_mm", (15.0, 12.0))),  # type: ignore[arg-type]
            seed=seed * 100 + index,
        )
        loaded = _load(
            phantom.frame, mode="single_frame", spacing=spacing, index=index, prefix="FRAME"
        )
        encoding = encoder.encode(loaded.array, loaded.metadata, acquisition_mode="single_frame")
        morphology = encoding.morphology

        pred_counts.append(morphology.follicle_number_per_section)
        true_counts.append(phantom.true_follicle_number_per_section)

        if morphology.ovary_area_mm2 is not None and phantom.true_ovary_area_mm2 > 0:
            area_errors.append(
                abs(morphology.ovary_area_mm2 - phantom.true_ovary_area_mm2)
                / phantom.true_ovary_area_mm2
            )
        if morphology.follicle_diameters_mm:
            diameter_scores.append(
                diameter_error(morphology.follicle_diameters_mm, phantom.true_diameters_mm)
            )
        segmentation = encoding.frame_segmentations[0]
        per_study.append(
            evaluate_segmentation(
                pred_ovary_region=segmentation.ovary_region_mask,
                pred_follicle=segmentation.follicle_mask,
                true_ovary_region=phantom.ovary_mask,
                true_follicle=phantom.follicle_mask,
            )
        )
        tokens.append(encoding.token.model_dump(mode="json"))

    metrics: dict[str, float] = {
        f"frame_{k}": float(np.mean([s[k] for s in per_study])) for k in per_study[0]
    }
    metrics.update(per_section_count_mae(pred_counts, true_counts))
    if area_errors:
        metrics["ovary_area_mape"] = float(np.mean(area_errors))
    if diameter_scores:
        for key in diameter_scores[0]:
            metrics[f"frame_{key}"] = float(np.mean([s[key] for s in diameter_scores]))
    return {"metrics": metrics, "tokens": tokens, "n": n_studies}


# ---------------------------------------------------------------------------
# Stage 3 surrogate: cine cohort (tracking)
# ---------------------------------------------------------------------------


def run_cine_cohort(
    encoder: UltrasoundEncoder, *, n_studies: int, seed: int, settings: dict[str, Any]
) -> dict[str, Any]:
    """Score the cine path: unique-track counts, fragmentation and merging."""
    from tests.fixtures.synthetic_ultrasound import make_cine_phantom  # noqa: PLC0415

    cine_cfg = settings.get("cine", {}) or {}
    spacing_2d = tuple(cine_cfg.get("pixel_spacing_mm", (0.35, 0.35)))
    spacing = (spacing_2d[0], spacing_2d[0], spacing_2d[1])

    pred_unique: list[int | None] = []
    true_unique: list[int] = []
    pred_sections: list[int | None] = []
    true_sections: list[int] = []
    coverages: list[float] = []
    confidences: list[float] = []
    tracking_scores: list[dict[str, float]] = []
    tokens: list[dict[str, Any]] = []

    for index in range(n_studies):
        # Every other loop drops frames, so coverage degradation is exercised
        # rather than only ever scored on a perfect sweep.
        unusable = (2, 7, 11) if index % 2 else ()
        phantom = make_cine_phantom(
            n_frames=int(cine_cfg.get("n_frames", 16)),
            shape=tuple(cine_cfg.get("shape", (128, 128))),  # type: ignore[arg-type]
            pixel_spacing_mm=spacing_2d,  # type: ignore[arg-type]
            ovary_semi_axes_mm=tuple(cine_cfg.get("ovary_semi_axes_mm", (15.0, 12.0))),  # type: ignore[arg-type]
            unusable_frames=unusable,
            seed=seed * 100 + index,
        )
        loaded = _load(
            phantom.frames, mode="cine_loop", spacing=spacing, index=index, prefix="CINE"
        )
        encoding = encoder.encode(loaded.array, loaded.metadata, acquisition_mode="cine_loop")
        morphology = encoding.morphology
        tracking = encoding.tracking

        pred_unique.append(morphology.estimated_follicle_number_per_ovary)
        true_unique.append(phantom.true_unique_follicle_count)

        # The representative per-section count, scored against the truth for the
        # frame it was taken from — never against the unique count.
        cine = encoding.cine_morphology
        if cine is not None and cine.representative_frame_index is not None:
            pred_sections.append(morphology.follicle_number_per_section)
            true_sections.append(phantom.true_per_section_counts[cine.representative_frame_index])

        if tracking is not None:
            coverages.append(tracking.tracking_coverage)
            confidences.append(tracking.confidence)
            observations = {
                t.track_id: {o.frame_index: o.centroid_mm for o in t.observations}
                for t in tracking.tracks
            }
            tracking_scores.append(
                tracking_fragmentation_and_merge(
                    match_tracks_to_truth(observations, phantom.true_observations_mm()),
                    n_true_follicles=phantom.true_unique_follicle_count,
                    n_predicted_tracks=tracking.estimated_unique_count,
                )
            )
        tokens.append(encoding.token.model_dump(mode="json"))

    metrics: dict[str, float] = {}
    metrics.update(unique_track_count_mae(pred_unique, true_unique))
    if pred_sections:
        # Namespaced: the cine loop's representative per-section count is scored
        # on a different cohort from the single-frame one, so sharing a metric
        # key would let one silently overwrite the other.
        metrics.update(
            {
                f"cine_representative_{k}": v
                for k, v in per_section_count_mae(pred_sections, true_sections).items()
            }
        )
    if coverages:
        metrics["mean_tracking_coverage"] = float(np.mean(coverages))
        metrics["mean_tracking_confidence"] = float(np.mean(confidences))
    if tracking_scores:
        for key in tracking_scores[0]:
            metrics[key] = float(np.nanmean([s[key] for s in tracking_scores]))
    return {"metrics": metrics, "tokens": tokens, "n": n_studies}


# ---------------------------------------------------------------------------
# Optional enhanced mode: 3D volume cohort
# ---------------------------------------------------------------------------


def run_volume_cohort(
    encoder: UltrasoundEncoder, *, n_studies: int, seed: int, settings: dict[str, Any]
) -> dict[str, Any]:
    """Score the optional volumetric path: true per-ovary counts and volumes."""
    from tests.fixtures.synthetic_ultrasound import make_phantom  # noqa: PLC0415

    volume_cfg = settings.get("volume", {}) or {}
    spacing = tuple(volume_cfg.get("spacing_mm", (1.0, 0.6, 0.6)))

    per_study: list[dict[str, float]] = []
    pred_counts: list[int | None] = []
    true_counts: list[int] = []
    pred_volumes: list[float | None] = []
    true_volumes: list[float] = []
    diameter_scores: list[dict[str, float]] = []
    tokens: list[dict[str, Any]] = []

    for index in range(n_studies):
        phantom = make_phantom(
            shape=tuple(volume_cfg.get("shape", (48, 64, 64))),  # type: ignore[arg-type]
            spacing=spacing,  # type: ignore[arg-type]
            seed=seed * 100 + index,
        )
        loaded = _load(
            phantom.volume, mode="volume_3d", spacing=phantom.spacing, index=index, prefix="VOL"
        )
        encoding = encoder.encode(loaded.array, loaded.metadata, acquisition_mode="volume_3d")
        morphology = encoding.morphology

        per_study.append(
            evaluate_segmentation(
                pred_ovary_region=encoding.segmentation.ovary_region_mask,
                pred_follicle=encoding.segmentation.follicle_mask,
                true_ovary_region=phantom.ovary_mask,
                true_follicle=phantom.follicle_mask,
            )
        )
        pred_counts.append(morphology.follicle_number_per_ovary)
        true_counts.append(phantom.true_count)
        pred_volumes.append(morphology.ovary_volume_ml)
        true_volumes.append(phantom.true_ovary_volume_ml)
        if morphology.follicle_diameters_mm:
            diameter_scores.append(
                diameter_error(morphology.follicle_diameters_mm, phantom.true_diameters_mm)
            )
        tokens.append(encoding.token.model_dump(mode="json"))

    metrics: dict[str, float] = {
        f"volume_{k}": float(np.mean([s[k] for s in per_study])) for k in per_study[0]
    }
    errors = [
        abs(float(p) - float(t))
        for p, t in zip(pred_counts, true_counts, strict=True)
        if p is not None
    ]
    metrics["per_ovary_count_mae"] = float(np.mean(errors)) if errors else float("nan")
    metrics["per_ovary_count_exact_match"] = (
        float(np.mean([e == 0.0 for e in errors])) if errors else float("nan")
    )
    metrics.update(ovarian_volume_absolute_error(pred_volumes, true_volumes))
    if diameter_scores:
        for key in diameter_scores[0]:
            metrics[f"volume_{key}"] = float(np.mean([s[key] for s in diameter_scores]))
    return {"metrics": metrics, "tokens": tokens, "n": n_studies}


# ---------------------------------------------------------------------------
# Quality gate
# ---------------------------------------------------------------------------


def run_quality_gate_check(encoder: UltrasoundEncoder, *, seed: int) -> dict[str, float]:
    """Check both gates accept measurable studies and refuse unmeasurable ones.

    The 2D and 3D gates are checked together because the safety property must
    hold for the pipeline as a whole: a study routed to the wrong gate that then
    passes is exactly the failure the unsafe-acceptance rate exists to catch.
    """
    from tests.fixtures.synthetic_ultrasound import (  # noqa: PLC0415
        make_cine_phantom,
        make_phantom,
        make_phantom_2d,
        make_poor_quality_frame,
        make_poor_quality_volume,
    )

    assessments = []
    truly_measurable: list[bool] = []

    def _assess(image, mode, spacing, prefix, index, measurable):
        loaded = _load(image, mode=mode, spacing=spacing, index=index, prefix=prefix)
        assessments.append(
            encoder.encode(loaded.array, loaded.metadata, acquisition_mode=mode).quality
        )
        truly_measurable.append(measurable)

    spacing_2d = (0.35, 0.35, 0.35)
    for index in range(3):
        _assess(
            make_phantom_2d(seed=seed * 50 + index).frame,
            "single_frame",
            spacing_2d,
            "GOODF",
            index,
            True,
        )
    for index in range(2):
        _assess(
            make_cine_phantom(seed=seed * 50 + index).frames,
            "cine_loop",
            spacing_2d,
            "GOODC",
            index,
            True,
        )
    for index in range(2):
        phantom = make_phantom(seed=seed * 50 + index)
        _assess(phantom.volume, "volume_3d", phantom.spacing, "GOODV", index, True)

    # Structureless noise: no ovary at all, in either dimensionality.
    for index in range(2):
        _assess(
            make_poor_quality_frame(seed=seed * 50 + 90 + index),
            "single_frame",
            spacing_2d,
            "NOISEF",
            index,
            False,
        )
        _assess(
            make_poor_quality_volume(seed=seed * 50 + 90 + index),
            "volume_3d",
            (1.0, 0.6, 0.6),
            "NOISEV",
            index,
            False,
        )

    # Perfectly good images whose spacing is unknown: no physical measurement.
    _assess(make_phantom_2d(seed=seed * 50 + 99).frame, "single_frame", None, "NOSPACEF", 0, False)
    _assess(make_phantom(seed=seed * 50 + 99).volume, "volume_3d", None, "NOSPACEV", 0, False)

    return {
        "quality_gate_sensitivity": quality_gate_sensitivity(assessments, truly_measurable),
        "quality_gate_unsafe_acceptance_rate": quality_gate_unsafe_acceptance_rate(
            assessments, truly_measurable
        ),
        "n_gate_studies": float(len(assessments)),
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser. Exposed so the CLI contract test can inspect it."""
    parser = make_parser(description=__doc__)
    add_standard_arguments(parser, config_default=DEFAULT_EXPERIMENT, quiet=False)
    # '--experiment' was this script's original name for '--config'.
    add_deprecated_alias(parser, "--experiment", dest="config", replacement="--config", type=Path)
    parser.add_argument("--n-studies", type=int, default=6, help="Phantom studies to generate.")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["single_frame", "cine_loop", "volume_3d"],
        choices=["single_frame", "cine_loop", "volume_3d"],
        help="Acquisition pathways to evaluate. 2D ones are the primary pathways.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    args = build_parser().parse_args(argv)

    experiment = load_yaml(args.config)
    data_config = load_yaml(_resolve(experiment.get("data", "configs/data/ultrasound.yaml")))
    model_config = load_yaml(
        _resolve(experiment.get("model", "configs/models/ultrasound_segmentation.yaml"))
    )

    segmenter = (model_config.get("segmenter_2d", {}) or {}).get("kind", "auto")
    instance_cfg = model_config.get("follicle_instances", {}) or {}
    settings = data_config.get("synthetic_fallback", {}) or {}
    seeds = [int(s) for s in experiment.get("seeds", [0])]
    if args.seed is not None:
        seeds = [int(args.seed)]
    experiment_id = args.experiment_id or str(experiment.get("experiment_id", "exp_ultrasound"))
    output_dir = resolve_output_dir(experiment, args.output_dir, experiment_id=experiment_id)

    encoder = UltrasoundEncoder(
        segmenter_kind=str(segmenter),
        min_diameter_mm=instance_cfg.get("min_diameter_mm"),
        large_structure_diameter_mm=instance_cfg.get("large_structure_diameter_mm"),
    )

    runners = {
        "single_frame": run_single_frame_cohort,
        "cine_loop": run_cine_cohort,
        "volume_3d": run_volume_cohort,
    }

    # Report the staged strategy this evaluation stands in for, so the artifact
    # records what was NOT trained here: no real 2D scans exist in CI.
    stages = [s.get("name") for s in (model_config.get("training", {}) or {}).get("stages", [])]
    if stages:
        print("Staged training strategy declared in the model config:")
        for index, name in enumerate(stages, start=1):
            print(f"  stage {index}: {name}")
        print("  (this script evaluates the assembled pipeline on phantoms; no weights are fit)\n")

    fold_metrics: list[FoldMetrics] = []
    all_tokens: list[dict[str, Any]] = []
    for fold, seed in enumerate(seeds):
        combined: dict[str, float] = {}
        n_test = 0
        for mode in args.modes:
            if mode not in runners:
                raise SystemExit(f"Unknown mode '{mode}'; expected one of {sorted(runners)}.")
            result = runners[mode](encoder, n_studies=args.n_studies, seed=seed, settings=settings)
            combined.update(result["metrics"])
            all_tokens.extend(result["tokens"])
            n_test += int(result["n"])
        combined.update(run_quality_gate_check(encoder, seed=seed))

        fold_metrics.append(
            FoldMetrics(fold=fold, seed=seed, n_train=0, n_test=n_test, metrics=combined)
        )
        print(f"seed {seed}:")
        for key in sorted(combined):
            print(f"  {key:<42} {combined[key]:.4f}")

    keys = sorted({k for f in fold_metrics for k in f.metrics})
    aggregate = {
        k: float(np.nanmean([f.metrics[k] for f in fold_metrics if k in f.metrics])) for k in keys
    }
    aggregate_std = {
        k: float(np.nanstd([f.metrics[k] for f in fold_metrics if k in f.metrics])) for k in keys
    }

    experiment_result = ExperimentResult(
        experiment_id=experiment_id,
        dataset_version=str(data_config.get("dataset_version", "unversioned")),
        git_commit="unknown",
        model="ovarian_ultrasound_encoder",
        target="ovarian_morphology",
        split_strategy="synthetic_phantom_no_split",
        seeds=seeds,
        fold_metrics=fold_metrics,
        aggregate_metrics=aggregate,
        aggregate_metrics_std=aggregate_std,
        limitations=[str(x) for x in experiment.get("limitations", [])],
    )
    experiment_result.write_json(output_dir / "metrics.json")
    (output_dir / "tokens.json").write_text(json.dumps(all_tokens, indent=2) + "\n")
    (output_dir / "model_card.json").write_text(
        json.dumps(encoder.export_model_card_metadata().model_dump(mode="json"), indent=2) + "\n"
    )
    print(f"\nWrote {output_dir}/metrics.json, tokens.json, model_card.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
