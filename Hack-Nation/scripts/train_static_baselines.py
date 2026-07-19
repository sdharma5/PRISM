#!/usr/bin/env python
"""Train and compare static-feature models under one shared set of patient folds.

Every model in the config sees identical folds and identical fold-local
preprocessing, so the comparison between them is about the models rather than
about who got the friendlier split. Runs end-to-end on a synthetic fixture cohort
when no real dataset is configured.

Research artifact: outputs are phenotype profiles, not diagnoses.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.reports import (  # noqa: E402
    build_experiment_result,
    write_experiment_readme,
    write_predictions,
)
from evaluation.subgroup import subgroup_report  # noqa: E402
from features.feature_manifest import build_feature_manifest, describe_pipeline  # noqa: E402
from features.static_features import build_static_features  # noqa: E402
from models.base import BasePrismModel  # noqa: E402
from scripts._cli import (  # noqa: E402
    add_deprecated_alias,
    add_standard_arguments,
    make_parser,
    resolve_output_root,
    resolve_seed,
)
from scripts._experiment_io import resolve_data_path  # noqa: E402
from training.callbacks import JsonlLoggingCallback, TimingCallback  # noqa: E402
from training.checkpoints import save_fold_checkpoints  # noqa: E402
from training.engine import (  # noqa: E402
    PreprocessingSpec,
    infer_preprocessing_spec,
    run_cross_validation,
)
from training.seeding import seed_everything  # noqa: E402
from training.splits import (  # noqa: E402
    build_split_manifest,
    reserve_holdout_patients,
    save_split_manifest,
)
from training.tracking import (  # noqa: E402
    FEATURE_MANIFEST_FILENAME,
    METRICS_FILENAME,
    README_FILENAME,
    SPLIT_MANIFEST_FILENAME,
    ExperimentTracker,
    load_config,
    resolve_experiment_dir,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser. Exposed so the CLI contract test can inspect it."""
    parser = make_parser(description=__doc__)
    add_standard_arguments(
        parser,
        output_help_suffix=(
            " For this script it is the ROOT under which the run directory is created."
        ),
    )
    # '--output-root' was this script's original name for '--output-dir'.
    add_deprecated_alias(
        parser, "--output-root", dest="output_dir", replacement="--output-dir", type=Path
    )
    parser.add_argument(
        "--no-timestamp",
        action="store_true",
        help="Write into <root>/<experiment_id> instead of a timestamped directory.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def load_cohort(
    config: dict[str, Any],
    data_root: str | Path | None = None,
) -> tuple[pd.DataFrame, str]:
    """Load the configured dataset, or fall back to the synthetic fixture cohort."""
    data_cfg = config.get("data", {})
    path = resolve_data_path(config, data_root) if (data_cfg.get("path") or data_root) else None
    if path:
        if not path.exists():
            raise FileNotFoundError(
                f"Configured dataset '{path}' does not exist.\n"
                "Set PRISM_DATA_ROOT (and export it), pass --data-root, or set `data.path` "
                "in the config. Remove all three to run on the synthetic fixture cohort."
            )
        if path.suffix == ".parquet":
            return pd.read_parquet(path), str(data_cfg.get("dataset_version", "unknown"))
        return pd.read_csv(path), str(data_cfg.get("dataset_version", "unknown"))

    from tests.fixtures.synthetic_tabular import make_synthetic_cohort

    synthetic = data_cfg.get("synthetic", {}) or {}
    df = make_synthetic_cohort(
        n=int(synthetic.get("n", 300)),
        seed=int(synthetic.get("seed", 0)),
        missing_rate=float(synthetic.get("missing_rate", 0.2)),
    )
    return df, "synthetic-fixture-v1"


def resolve_model_class(dotted_path: str) -> type[BasePrismModel]:
    """Import a model class from its ``module.ClassName`` path."""
    module_name, _, class_name = dotted_path.rpartition(".")
    if not module_name:
        raise ValueError(f"'{dotted_path}' is not a fully qualified class path.")
    cls = getattr(importlib.import_module(module_name), class_name)
    if not issubclass(cls, BasePrismModel):
        raise TypeError(f"{dotted_path} is not a BasePrismModel subclass.")
    return cls


def model_specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Accept either a ``models:`` list or a single ``model:`` block."""
    if config.get("models"):
        return list(config["models"])
    if config.get("model"):
        return [config["model"]]
    raise ValueError("Config declares neither 'models' nor 'model'.")


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    config = load_config(args.config)

    experiment_id = args.experiment_id or config.get("experiment_id", args.config.stem)
    output_root = resolve_output_root(config, args.output_dir)
    timestamped = not args.no_timestamp and bool(config.get("output", {}).get("timestamped", True))
    experiment_dir = resolve_experiment_dir(output_root, experiment_id, timestamped=timestamped)

    seed = resolve_seed(config, args.seed)
    seed_everything(seed)

    tracker = ExperimentTracker(
        experiment_dir,
        experiment_id=experiment_id,
        resolved_config=config,
        echo=not args.quiet,
    ).start()

    df, dataset_version = load_cohort(config, args.data_root)
    data_cfg = config.get("data", {})
    id_column = data_cfg.get("id_column", "patient_id")
    label_column = data_cfg.get("label_column", "pmos_binary")
    group_column = data_cfg.get("group_column")

    tracker.log("cohort_loaded", n_rows=len(df), n_columns=df.shape[1], source=dataset_version)

    # -- Features ----------------------------------------------------------
    feature_cfg = config.get("features", {})
    matrix = build_static_features(
        df,
        label_column=label_column,
        id_column=id_column,
        include_groups=feature_cfg.get("include_groups"),
        add_missingness_indicators=bool(feature_cfg.get("add_missingness_indicators", True)),
        per_status_indicators=bool(feature_cfg.get("per_status_indicators", False)),
        min_observed_fraction=float(feature_cfg.get("min_observed_fraction", 0.0)),
    )
    if matrix.y is None:
        raise ValueError(f"Label column '{label_column}' is absent from the cohort.")

    tracker.log(
        "features_built",
        n_features=matrix.X.shape[1],
        groups={g: len(cols) for g, cols in matrix.feature_groups.items()},
        derived=matrix.derived_columns,
    )

    # -- Splits ------------------------------------------------------------
    split_cfg = config.get("split", {})
    strategy = split_cfg.get("strategy", "repeated_stratified_kfold")
    holdout_ids = reserve_holdout_patients(
        matrix.patient_ids,
        matrix.y,
        fraction=float(split_cfg.get("holdout_fraction", 0.0)),
        seed=seed,
    )

    split_kwargs: dict[str, Any] = {
        "manifest_id": f"{experiment_id}_splits",
        "dataset_id": data_cfg.get("dataset_id", "unknown"),
        "dataset_version": dataset_version,
        "holdout_ids": holdout_ids,
    }
    if strategy in {"repeated_stratified_kfold", "grouped_kfold"}:
        split_kwargs["n_splits"] = int(split_cfg.get("n_splits", 5))
    if strategy == "repeated_stratified_kfold":
        split_kwargs["n_repeats"] = int(split_cfg.get("n_repeats", 1))
    if strategy != "holdout":
        split_kwargs["seeds"] = [int(s) for s in split_cfg.get("seeds", [seed])]
    else:
        split_kwargs["seed"] = seed
        split_kwargs["test_size"] = float(split_cfg.get("test_size", 0.2))

    groups = df[group_column].tolist() if group_column and group_column in df.columns else None
    manifest = build_split_manifest(
        strategy, matrix.patient_ids.tolist(), matrix.y.tolist(), groups, **split_kwargs
    )
    save_split_manifest(manifest, tracker.path(SPLIT_MANIFEST_FILENAME))
    tracker.log("splits_built", strategy=strategy, n_folds=len(manifest.folds))

    # -- Cross-validation, one model at a time over identical folds --------
    preprocessing_cfg = config.get("preprocessing", {})
    spec: PreprocessingSpec = infer_preprocessing_spec(
        matrix.X,
        numeric_impute_strategy=preprocessing_cfg.get("numeric_impute_strategy", "median"),
        scale=bool(preprocessing_cfg.get("scale", True)),
        add_indicator=bool(preprocessing_cfg.get("add_indicator", True)),
    )

    eval_cfg = config.get("evaluation", {})
    timing = TimingCallback()
    callbacks = [JsonlLoggingCallback(tracker.logger), timing]

    all_predictions: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    primary_result = None
    last_pipeline = None

    for spec_entry in model_specs(config):
        model_name = spec_entry.get("name") or spec_entry["class"].rpartition(".")[2]
        model_cls = resolve_model_class(spec_entry["class"])
        params = dict(spec_entry.get("params", {}) or {})

        tracker.log("model_start", model=model_name, params=params)

        def factory(
            fold_seed: int, cls: type[BasePrismModel] = model_cls, kwargs: dict[str, Any] = params
        ) -> BasePrismModel:
            return cls(random_state=fold_seed, **kwargs) if _accepts_seed(cls) else cls(**kwargs)

        cv = run_cross_validation(
            matrix.X,
            matrix.y,
            matrix.patient_ids.tolist(),
            manifest,
            factory,
            spec=spec,
            base_seed=seed,
            keep_models=bool(config.get("output", {}).get("save_checkpoints", False)),
            callbacks=callbacks,
        )
        last_pipeline = cv.last_pipeline or last_pipeline

        predictions = cv.predictions.copy()
        if not predictions.empty:
            predictions.insert(0, "model", model_name)
            all_predictions.append(predictions)

        result = build_experiment_result(
            experiment_id=f"{experiment_id}:{model_name}",
            dataset_version=dataset_version,
            git_commit=tracker.git_commit,
            model=model_name,
            target=label_column,
            split_strategy=strategy,
            seeds=manifest.seeds,
            fold_records=cv.fold_records(),
            predictions=cv.predictions,
            feature_manifest=FEATURE_MANIFEST_FILENAME,
            split_manifest=SPLIT_MANIFEST_FILENAME,
            n_bins=int(eval_cfg.get("calibration_bins", 10)),
        )
        result.write_json(tracker.path(f"metrics_{model_name}.json"))

        summary_rows.append(
            {
                "model": model_name,
                "n_folds": len(result.fold_metrics),
                **{k: v for k, v in result.aggregate_metrics.items()},
            }
        )
        tracker.log(
            "model_end",
            model=model_name,
            auroc=result.aggregate_metrics.get("auroc"),
            auprc=result.aggregate_metrics.get("auprc"),
            balanced_accuracy=result.aggregate_metrics.get("balanced_accuracy"),
        )

        # The first non-baseline model is the headline result written to metrics.json.
        if primary_result is None and not model_name.startswith("baseline"):
            primary_result = result
        if bool(config.get("output", {}).get("save_checkpoints", False)):
            save_fold_checkpoints(tracker.dir, cv.fold_results)

    if primary_result is None:
        raise RuntimeError("No non-baseline model produced a result.")

    # -- Manifests and reports --------------------------------------------
    feature_manifest = build_feature_manifest(
        matrix,
        manifest_id=f"{experiment_id}_features",
        dataset_id=data_cfg.get("dataset_id"),
        dataset_version=dataset_version,
        label_column=label_column,
        transforms=describe_pipeline(last_pipeline) if last_pipeline is not None else [],
        notes=[
            "All imputation and scaling statistics were fitted inside each training fold.",
            "Missing values are represented explicitly and are never zero-filled without an "
            "accompanying indicator column.",
        ],
    )
    feature_manifest.write_json(tracker.path(FEATURE_MANIFEST_FILENAME))

    combined = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    predictions_format = config.get("output", {}).get("predictions_format", "csv")
    if not combined.empty:
        write_predictions(combined, tracker.path(f"predictions.{predictions_format}"))

    subgroup_summary: dict[str, Any] = {}
    subgroup_columns = list(eval_cfg.get("subgroup_columns", []) or [])
    primary_predictions = pd.DataFrame()
    if not combined.empty:
        primary_predictions = combined[combined["model"] == primary_result.model]
    if subgroup_columns and not primary_predictions.empty:
        merged = primary_predictions.merge(
            df[[id_column, *[c for c in subgroup_columns if c in df.columns]]],
            left_on="patient_id",
            right_on=id_column,
            how="left",
        )
        subgroup_summary = subgroup_report(
            merged["y_true"].to_numpy(),
            merged["y_prob"].to_numpy(),
            merged,
            subgroup_columns,
            threshold=float(eval_cfg.get("threshold", 0.5)),
        )
        tracker.write_json("subgroup_metrics.json", subgroup_summary)

    primary_result.write_json(tracker.path(METRICS_FILENAME))

    comparison_md = _comparison_table(summary_rows)

    write_experiment_readme(
        tracker.path(README_FILENAME),
        primary_result,
        config_summary={
            "experiment_id": experiment_id,
            "dataset_id": data_cfg.get("dataset_id"),
            "n_patients": int(matrix.patient_ids.nunique()),
            "n_features": int(matrix.X.shape[1]),
            "split_strategy": strategy,
            "seed": seed,
        },
        subgroup_summary=subgroup_summary,
        extra_sections={
            "Model comparison": comparison_md
            + "\n\nBaselines are included so that any headline number can be read relative to "
            "a majority-class predictor and a single-variable rule.",
        },
    )

    model_card = None
    tracker.write_json(
        "model_card.json",
        (model_card or _primary_model_card(config, primary_result, dataset_version)),
    )

    missing = tracker.verify_artifacts()
    if missing:
        raise RuntimeError(f"Experiment finished with missing artifacts: {missing}")

    tracker.finish(
        experiment_dir=str(tracker.dir),
        total_seconds=round(timing.total_seconds, 2),
        models=[row["model"] for row in summary_rows],
    )
    print(f"\nArtifacts written to: {tracker.dir}")
    return tracker.dir


def _comparison_table(rows: list[dict[str, Any]]) -> str:
    """Markdown comparison of every model in the run, written without extra deps."""
    if not rows:
        return "_no model comparison available_"
    columns = ["model", "n_folds", "auroc", "auprc", "balanced_accuracy", "brier"]
    columns = [c for c in columns if any(c in row for row in rows)]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        cells = []
        for column in columns:
            value = row.get(column)
            cells.append(f"{value:.3f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _accepts_seed(cls: type[BasePrismModel]) -> bool:
    """Whether the model constructor takes a ``random_state``."""
    import inspect

    return "random_state" in inspect.signature(cls.__init__).parameters


def _primary_model_card(
    config: dict[str, Any],
    result: Any,
    dataset_version: str,
) -> dict[str, Any]:
    """Model-card metadata for the headline model of this experiment."""
    spec = next(
        (s for s in model_specs(config) if s.get("name") == result.model),
        model_specs(config)[0],
    )
    cls = resolve_model_class(spec["class"])
    model = cls(**dict(spec.get("params", {}) or {}))
    card = model.export_model_card_metadata(
        training_datasets=[dataset_version],
        evaluation_datasets=[dataset_version],
        metrics={
            k: v
            for k, v in result.aggregate_metrics.items()
            if k in {"auroc", "auprc", "balanced_accuracy", "brier", "ece"}
        },
    )
    return card.model_dump(mode="json")


if __name__ == "__main__":
    main()
