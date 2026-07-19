#!/usr/bin/env python
"""Train the masked tabular autoencoder and export static-clinical phenotype tokens.

The autoencoder is unsupervised: it never sees the label. It is still evaluated
under patient-level folds, because the honest question is whether the learned
structure transfers to patients the model has not seen — and the comparator is
always mean imputation.

Research artifact: the exported tokens describe phenotype profiles, not diagnoses.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.classification import aggregate_fold_metrics  # noqa: E402
from evaluation.reports import write_experiment_readme  # noqa: E402
from features.feature_manifest import build_feature_manifest  # noqa: E402
from features.static_features import build_static_features, value_columns_of  # noqa: E402
from models.phenotype.domain_scorer import StaticClinicalTokenizer  # noqa: E402
from models.tabular.masked_autoencoder import MaskedTabularAutoencoder  # noqa: E402
from schemas.model_output import ExperimentResult, FoldMetrics  # noqa: E402
from scripts._cli import (  # noqa: E402
    add_deprecated_alias,
    add_standard_arguments,
    make_parser,
    resolve_output_root,
    resolve_seed,
)
from scripts.train_static_baselines import load_cohort  # noqa: E402
from training.seeding import derive_seed, seed_everything  # noqa: E402
from training.splits import (  # noqa: E402
    build_split_manifest,
    fold_row_indices,
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
    parser.add_argument("--no-timestamp", action="store_true", help="Write to an un-stamped dir.")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


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

    feature_cfg = config.get("features", {})
    matrix = build_static_features(
        df,
        label_column=label_column,
        id_column=id_column,
        include_groups=feature_cfg.get("include_groups"),
        # The autoencoder reconstructs *values*; indicator columns are perfectly
        # predictable from the mask channel and would flatter the reconstruction.
        add_missingness_indicators=False,
        min_observed_fraction=float(feature_cfg.get("min_observed_fraction", 0.0)),
    )
    X = matrix.X[value_columns_of(matrix.X)]
    tracker.log("features_built", n_rows=len(X), n_features=X.shape[1])

    # -- Folds -------------------------------------------------------------
    split_cfg = config.get("split", {})
    strategy = split_cfg.get("strategy", "repeated_stratified_kfold")
    stratify_on = matrix.y.tolist() if matrix.y is not None else None
    manifest = build_split_manifest(
        strategy,
        matrix.patient_ids.tolist(),
        stratify_on,
        None,
        manifest_id=f"{experiment_id}_splits",
        dataset_id=data_cfg.get("dataset_id", "unknown"),
        dataset_version=dataset_version,
        n_splits=int(split_cfg.get("n_splits", 4)),
        n_repeats=int(split_cfg.get("n_repeats", 1)),
        seeds=[int(s) for s in split_cfg.get("seeds", [seed])],
    )
    save_split_manifest(manifest, tracker.path(SPLIT_MANIFEST_FILENAME))

    model_params = dict((config.get("model", {}) or {}).get("params", {}) or {})
    if "mask_rate_range" in model_params:
        model_params["mask_rate_range"] = tuple(model_params["mask_rate_range"])

    ids = matrix.patient_ids.tolist()
    fold_records: list[dict[str, Any]] = []
    embedding_rows: list[pd.DataFrame] = []

    for fold_index, fold in enumerate(manifest.folds):
        fold_seed = derive_seed(seed, "fold", fold_index)
        seed_everything(fold_seed)
        train_idx, test_idx = fold_row_indices(fold, ids)
        if train_idx.size == 0 or test_idx.size == 0:
            continue

        # Column statistics live inside the model and are fitted on train rows only.
        model = MaskedTabularAutoencoder(random_state=fold_seed, **model_params)
        model.fit(X.iloc[train_idx])
        metrics = model.evaluate(X.iloc[test_idx], seed=fold_seed)

        fold_records.append(
            {
                "fold": fold_index,
                "seed": fold_seed,
                "n_train": int(train_idx.size),
                "n_test": int(test_idx.size),
                "metrics": metrics,
            }
        )
        tracker.log("fold_end", fold=fold_index, metrics=metrics)

        latent = model.embed(X.iloc[test_idx])
        frame = pd.DataFrame(latent, columns=[f"z{i}" for i in range(latent.shape[1])])
        frame.insert(0, "patient_id", [ids[i] for i in test_idx])
        frame.insert(1, "fold", fold_index)
        embedding_rows.append(frame)

    if not fold_records:
        raise RuntimeError("No fold produced a result; check the split configuration.")

    mean, std = aggregate_fold_metrics([r["metrics"] for r in fold_records])
    beats = mean.get("beats_mean_imputation", 0.0)

    result = ExperimentResult(
        experiment_id=experiment_id,
        dataset_version=dataset_version,
        git_commit=tracker.git_commit,
        model="tabular_masked_autoencoder",
        target="masked_value_reconstruction",
        split_strategy=strategy,
        seeds=manifest.seeds,
        fold_metrics=[
            FoldMetrics(
                fold=r["fold"],
                seed=r["seed"],
                n_train=r["n_train"],
                n_test=r["n_test"],
                metrics={k: float(v) for k, v in r["metrics"].items()},
            )
            for r in fold_records
        ],
        aggregate_metrics={k: v for k, v in mean.items() if np.isfinite(v)},
        aggregate_metrics_std={k: v for k, v in std.items() if np.isfinite(v)},
        feature_manifest=FEATURE_MANIFEST_FILENAME,
        split_manifest=SPLIT_MANIFEST_FILENAME,
        limitations=[
            "Unsupervised reconstruction quality does not imply the embedding is clinically "
            "meaningful; it only shows the model captured covariance between variables.",
            "The mean-imputation comparator is a floor, not a strong baseline.",
            "Reconstructed values are model estimates and must never be recorded as if they "
            "had been measured.",
            f"Beat mean imputation on {beats:.0%} of folds.",
        ],
    )
    result.write_json(tracker.path(METRICS_FILENAME))

    if embedding_rows:
        pd.concat(embedding_rows, ignore_index=True).to_csv(
            tracker.path("embeddings.csv"), index=False
        )

    # -- Phenotype tokens --------------------------------------------------
    phenotype_cfg = config.get("phenotype", {}) or {}
    token_summary: dict[str, Any] = {}
    if phenotype_cfg.get("export_tokens", True):
        tokenizer = StaticClinicalTokenizer(
            source_dataset=data_cfg.get("dataset_id"),
            include_embedding=bool(phenotype_cfg.get("include_embedding", True)),
            id_column=id_column,
        )
        # Tokens are exported for inspection, so the tokenizer is fitted on the
        # full cohort here. Any *evaluation* of them must refit inside a fold.
        tokenizer.fit(df, **model_params)
        tokens = tokenizer.transform(df)

        token_dir = tracker.dir / "tokens"
        limit = int(phenotype_cfg.get("max_tokens_written", 25))
        for token in tokens[:limit]:
            token.write_json(token_dir / f"{token.patient_id}.json")

        qualified = sum(
            1 for t in tokens if any("evidence_qualifier" in k for k in t.structured_features)
        )
        token_summary = {
            "n_tokens": len(tokens),
            "n_written": min(limit, len(tokens)),
            "mean_quality_score": float(np.mean([t.quality_score for t in tokens])),
            "mean_confidence_score": float(np.mean([t.confidence_score for t in tokens])),
            "n_with_symptom_only_qualifier": qualified,
        }
        tracker.write_json("token_summary.json", token_summary)
        tracker.log("tokens_exported", **token_summary)

    feature_manifest = build_feature_manifest(
        matrix,
        manifest_id=f"{experiment_id}_features",
        dataset_id=data_cfg.get("dataset_id"),
        dataset_version=dataset_version,
        label_column=None,
        transforms=[
            {
                "step": "masked_autoencoder_standardization",
                "class": "MaskedTabularAutoencoder",
                "note": "Per-column mean/std computed from observed entries of the training "
                "fold only.",
            }
        ],
        domain_scoring=(tokenizer.manifest() if phenotype_cfg.get("export_tokens", True) else {}),
        notes=[
            "The autoencoder is unsupervised and never sees the label.",
            "Masked cells enter the network as 0 in standardized space *with* a mask channel, "
            "so absence is never confused with a measured zero.",
        ],
    )
    feature_manifest.write_json(tracker.path(FEATURE_MANIFEST_FILENAME))

    def _metric(key: str) -> float:
        return float(mean.get(key, float("nan")))

    write_experiment_readme(
        tracker.path(README_FILENAME),
        result,
        config_summary={
            "experiment_id": experiment_id,
            "n_patients": int(matrix.patient_ids.nunique()),
            "n_features": int(X.shape[1]),
            "latent_dim": model_params.get("latent_dim", 16),
            "seed": seed,
        },
        extra_sections={
            "Masked reconstruction": (
                f"- masked reconstruction MSE: {_metric('masked_reconstruction_mse'):.4f}\n"
                f"- mean-imputation MSE: {_metric('mean_imputation_mse'):.4f}\n"
                f"- relative improvement: {_metric('mse_improvement_over_mean'):.1%}\n"
            ),
            "Exported tokens": (
                "\n".join(f"- {k}: {v}" for k, v in token_summary.items()) or "_no tokens exported_"
            ),
        },
    )

    missing = tracker.verify_artifacts()
    if missing:
        raise RuntimeError(f"Experiment finished with missing artifacts: {missing}")

    tracker.finish(experiment_dir=str(tracker.dir), n_folds=len(fold_records))
    print(f"\nArtifacts written to: {tracker.dir}")
    return tracker.dir


if __name__ == "__main__":
    main()
