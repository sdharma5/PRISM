#!/usr/bin/env python
"""Train and evaluate the dynamic hormonal-state model end to end.

Runs on the synthetic longitudinal cohort when mcPHASES is not present, which is
the normal case since mcPHASES requires credentialed access and is never
committed.

The split is **grouped by participant**, always. A random day-level split would
place consecutive days of the same person on both sides of the split and report
interpolation within a known body as generalisation to a new one. The manifest
written here asserts disjointness before any metric is computed.

Also runs the missing-modality ablation, because the headline number of a
multimodal model is measured at a data richness most people never have; the
degradation table is the honest version of the result.
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

from evaluation.temporal import (
    evaluate_temporal,
    format_ablation_table,
    hormone_metrics,
    interval_coverage,
    missing_modality_ablation,
    peak_timing_errors,
)
from models.temporal.losses import LossWeights
from models.temporal.state_model import TemporalStateModel, grouped_participant_split
from schemas.model_output import ExperimentResult, FoldMetrics
from schemas.patient import SplitManifest
from schemas.temporal import ParticipantDay
from scripts._cli import (
    add_deprecated_alias,
    add_standard_arguments,
    make_parser,
    resolve_output_dir,
)
from scripts._experiment_io import resolve_data_root

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT = REPO_ROOT / "configs" / "experiments" / "exp_dynamic_state.yaml"

CHANNEL_GROUPS: dict[str, str] = {
    "lh": "hormone",
    "e3g": "hormone",
    "pdg": "hormone",
    "resting_heart_rate": "wearable",
    "wrist_temperature": "wearable",
    "hrv_rmssd": "wearable",
    "mean_glucose": "cgm",
}


def load_yaml(path: Path) -> dict[str, Any]:
    with Path(path).open() as fh:
        return yaml.safe_load(fh) or {}


def _resolve(path_like: str) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else REPO_ROOT / path


def load_days(
    data_config: dict[str, Any],
    data_root: str | Path | None = None,
) -> tuple[list[ParticipantDay], str]:
    """Load real participant-days, or synthesise a cohort when absent.

    A configured-but-unloadable dataset is a hard error. The previous version
    caught every exception and fell back to synthetic data, which meant a typo
    in a path — or an import that did not resolve — produced a full run against
    planted geometry that was labelled as if it had been asked for. Only the
    *absence* of a configured root is a legitimate reason to synthesise.
    """
    root = resolve_data_root(data_config, data_root)
    if root is not None and root.exists():
        from ingestion.mcphases.loader import load_participant_days  # noqa: PLC0415

        return list(load_participant_days(root)), str(root)

    if data_root is not None:
        # An explicit --data-root that does not exist is a mistake, not a cue to
        # quietly switch to synthetic data.
        raise FileNotFoundError(
            f"--data-root '{root}' does not exist.\n"
            "mcPHASES requires credentialed PhysioNet access and is never committed. "
            "Omit --data-root to run on the synthetic longitudinal cohort instead."
        )

    from tests.fixtures.synthetic_cycles import generate_cohort  # noqa: PLC0415

    settings = data_config.get("synthetic_fallback", {}) or {}
    cohort = generate_cohort(
        n_participants=int(settings.get("n_participants", 14)),
        n_days=int(settings.get("n_days", 90)),
        seed=int(settings.get("seed", 0)),
    )
    return cohort.days, "synthetic_cycles"


def _day_key(day: ParticipantDay) -> str:
    """Label a participant-day exactly as ``TemporalStateModel`` labels its output.

    Must stay in step with ``models/temporal/state_model.py``, which sets
    ``as_of_date = day.calendar_date or f"study_day_{day.study_day}"``.
    """
    return str(day.calendar_date) if day.calendar_date else f"study_day_{day.study_day}"


def hormone_evaluation(model: TemporalStateModel, days: list[ParticipantDay]) -> dict[str, float]:
    """Score hormone reconstruction on observed test values only."""
    outputs = model.predict(days)
    if not outputs:
        return {}
    # Key the truth table the same way the model labels its outputs, rather than
    # parsing the label back into a study day. TemporalStateModel emits
    # `calendar_date or f"study_day_{study_day}"`, so real participant-days are
    # keyed by an ISO date and synthetic ones by "study_day_N". Parsing assumed
    # the synthetic form and raised ValueError on every real dataset — a path no
    # test reached, because the synthetic fixture carries no calendar dates.
    truth = {(d.participant_id, _day_key(d)): d for d in days}
    metrics: dict[str, float] = {}
    by_participant_pred: dict[str, list[float]] = {}
    by_participant_true: dict[str, list[float]] = {}

    for channel in ("lh", "e3g", "pdg"):
        predicted: list[float] = []
        actual: list[float] = []
        sigmas: list[float] = []
        residual = model.heads.hormone.residual_std.get(channel, float("nan"))
        for output in outputs:
            source = truth.get((output.patient_id, str(output.as_of_date)))
            if source is None or not source.is_observed.get(channel):
                continue
            value = source.values.get(channel)
            if value is None:
                continue
            prediction = float(output.hormone_predictions.get(channel, np.nan))
            predicted.append(prediction)
            actual.append(float(value))
            sigmas.append(residual)
            if channel == "lh":
                by_participant_pred.setdefault(output.patient_id, []).append(prediction)
                by_participant_true.setdefault(output.patient_id, []).append(float(value))
        scores = hormone_metrics(predicted, actual)
        for key, value in scores.items():
            metrics[f"{channel}_{key}"] = value
        coverage = interval_coverage(predicted, actual, sigmas)
        metrics[f"{channel}_interval_coverage"] = coverage["coverage"]

    peaks = peak_timing_errors(by_participant_pred, by_participant_true)
    metrics.update({f"lh_{k}": v for k, v in peaks.items()})
    return metrics


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser. Exposed so the CLI contract test can inspect it."""
    parser = make_parser(description=__doc__)
    add_standard_arguments(parser, config_default=DEFAULT_EXPERIMENT, quiet=False)
    # '--experiment' was this script's original name for '--config'.
    add_deprecated_alias(parser, "--experiment", dest="config", replacement="--config", type=Path)
    parser.add_argument(
        "--skip-ablation",
        action="store_true",
        help="Skip the missing-modality ablation table.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    args = build_parser().parse_args(argv)

    experiment = load_yaml(args.config)
    data_config = load_yaml(_resolve(experiment.get("data", "configs/data/mcphases.yaml")))
    model_config = load_yaml(_resolve(experiment.get("model", "configs/models/temporal_gru.yaml")))

    encoder_cfg = model_config.get("encoder", {}) or {}
    weights_cfg = model_config.get("loss_weights", {}) or {}
    split_cfg = experiment.get("split", {}) or {}
    seeds = [int(s) for s in experiment.get("seeds", [0])]
    if args.seed is not None:
        seeds = [int(args.seed)]
    experiment_id = args.experiment_id or str(experiment.get("experiment_id", "exp_dynamic_state"))
    output_dir = resolve_output_dir(experiment, args.output_dir, experiment_id=experiment_id)

    try:
        days, source = load_days(data_config, args.data_root)
    except FileNotFoundError as exc:
        # Actionable message, not a traceback: the user mistyped a path or has
        # not obtained the dataset, and neither is a programming error.
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    participants = sorted({d.participant_id for d in days})
    print(f"Loaded {len(days)} participant-days from {len(participants)} participants ({source}).")

    fold_metrics: list[FoldMetrics] = []
    folds: list[dict[str, list[str]]] = []
    last_model: TemporalStateModel | None = None
    last_test: list[ParticipantDay] = []

    for fold, seed in enumerate(seeds):
        groups = [d.participant_id for d in days]
        train_index, test_index = grouped_participant_split(
            groups, test_fraction=float(split_cfg.get("test_fraction", 0.3)), seed=seed
        )
        train_days = [days[i] for i in train_index]
        test_days = [days[i] for i in test_index]
        train_ids = sorted({d.participant_id for d in train_days})
        test_ids = sorted({d.participant_id for d in test_days})

        overlap = set(train_ids) & set(test_ids)
        if overlap:
            raise RuntimeError(
                f"Participant leakage across the split: {sorted(overlap)}. "
                "Days from one participant must never appear in both train and test."
            )
        folds.append({"train": train_ids, "test": test_ids})

        model = TemporalStateModel(
            lookback_days=int(encoder_cfg.get("lookback_days", 21)),
            hidden_size=int(encoder_cfg.get("hidden_size", 32)),
            encoder_kind=str(encoder_cfg.get("kind", "gru")),
            backend=str(encoder_cfg.get("backend", "auto")),
            use_decay=bool(encoder_cfg.get("use_decay", True)),
            decay_gamma=float(encoder_cfg.get("decay_gamma", 0.35)),
            loss_weights=LossWeights(
                hormone=float(weights_cfg.get("hormone", 1.0)),
                cycle=float(weights_cfg.get("cycle", 1.0)),
                symptom=float(weights_cfg.get("symptom", 0.5)),
                masked=float(weights_cfg.get("masked", 0.5)),
            ),
            channel_groups=CHANNEL_GROUPS,
            seed=seed,
        ).fit(train_days)

        outputs = model.predict(test_days)
        metrics = evaluate_temporal(outputs, test_days)
        metrics.update(hormone_evaluation(model, test_days))
        if model.report:
            metrics.update({f"train_loss_{k}": v for k, v in model.report.losses.items()})

        fold_metrics.append(
            FoldMetrics(
                fold=fold,
                seed=seed,
                n_train=len(train_days),
                n_test=len(test_days),
                metrics={k: float(v) for k, v in metrics.items() if np.isfinite(v)},
            )
        )
        last_model, last_test = model, test_days
        print(
            f"fold {fold} (seed {seed}): {len(train_ids)} train / {len(test_ids)} test participants"
            f" | balanced_acc={metrics.get('balanced_accuracy', float('nan')):.3f}"
            f" macro_f1={metrics.get('macro_f1', float('nan')):.3f}"
            f" lh_mae={metrics.get('lh_mae', float('nan')):.3f}"
        )

    manifest = SplitManifest(
        manifest_id=f"{experiment_id}_splits",
        dataset_id=str(data_config.get("dataset_id", "synthetic_cycles")),
        dataset_version=str(data_config.get("dataset_version", "unversioned")),
        strategy="grouped_kfold",
        n_splits=len(seeds),
        seeds=seeds,
        folds=folds,
    )
    manifest.assert_disjoint()
    (output_dir / "split_manifest.json").write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2) + "\n"
    )

    keys = sorted({k for f in fold_metrics for k in f.metrics})
    aggregate = {
        k: float(np.nanmean([f.metrics[k] for f in fold_metrics if k in f.metrics])) for k in keys
    }
    aggregate_std = {
        k: float(np.nanstd([f.metrics[k] for f in fold_metrics if k in f.metrics])) for k in keys
    }

    if not args.skip_ablation and last_model is not None:
        print("\nMissing-modality ablation:")
        rows = missing_modality_ablation(
            last_test,
            last_model.predict,
            conditions=tuple(experiment.get("evaluation", {}).get("ablations", ["full"])),
        )
        print(format_ablation_table(rows))
        (output_dir / "ablation.json").write_text(
            json.dumps(
                [
                    {
                        "condition": r.condition,
                        "metrics": r.metrics,
                        "degradation": r.degradation,
                        "mean_input_coverage": r.mean_input_coverage,
                    }
                    for r in rows
                ],
                indent=2,
            )
            + "\n"
        )

    if last_model is not None:
        outputs = last_model.predict(last_test)
        if outputs:
            token = last_model.to_token(outputs[-1], source_dataset=source)
            token.write_json(output_dir / "temporal_state_token.json")
        (output_dir / "model_card.json").write_text(
            json.dumps(last_model.export_model_card_metadata().model_dump(mode="json"), indent=2)
            + "\n"
        )

    ExperimentResult(
        experiment_id=experiment_id,
        dataset_version=str(data_config.get("dataset_version", "unversioned")),
        git_commit="unknown",
        model="longitudinal_hormonal_state_model",
        target="current_hormonal_state",
        split_strategy="grouped_kfold_by_participant",
        seeds=seeds,
        fold_metrics=fold_metrics,
        aggregate_metrics=aggregate,
        aggregate_metrics_std=aggregate_std,
        split_manifest=str(output_dir / "split_manifest.json"),
        limitations=[str(x) for x in experiment.get("limitations", [])],
    ).write_json(output_dir / "metrics.json")

    print(f"\nWrote {output_dir}/metrics.json, split_manifest.json, model_card.json")
    for key in ("balanced_accuracy", "macro_f1", "calibration_error", "macro_auprc", "lh_mae"):
        if key in aggregate:
            print(f"  {key:<24} {aggregate[key]:.4f} +/- {aggregate_std[key]:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
