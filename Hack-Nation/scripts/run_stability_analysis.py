#!/usr/bin/env python
"""Stress-test a discovered grouping and report how much of it survives.

    python scripts/run_stability_analysis.py --config configs/experiments/exp_subtype_stability.yaml

Runs bootstrap resampling, leave-one-feature-out and leave-one-modality-out
ablation, and measurement-noise / scaling / imputation perturbation over the
grouping selected by the same config, then writes per-participant stability
reports plus a cohort-level summary. Falls back to synthetic data when no real
dataset is available, and marks the artifacts accordingly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _cli import add_deprecated_alias, add_standard_arguments, make_parser  # noqa: E402
from _experiment_io import (  # noqa: E402
    build_representations,
    load_cohort,
    load_config,
    resolve_artifact_dir,
    write_benchmark_csv,
    write_json,
)
from discover_phenotypes import build_adapter_config  # noqa: E402

from models.adapters.pcos.adapter import PcosAdapter  # noqa: E402
from models.adapters.pcos.output_schema import NON_DIAGNOSTIC_STATEMENT  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser. Exposed so the CLI contract test can inspect it."""
    parser = make_parser(description=__doc__)
    add_standard_arguments(parser, experiment_id=False, quiet=False)
    # '--artifact-dir' was this script's original name for '--output-dir'.
    add_deprecated_alias(
        parser, "--artifact-dir", dest="output_dir", replacement="--output-dir", type=Path
    )
    parser.add_argument(
        "--limit-participants",
        type=int,
        default=0,
        help="Profile only the first N participants (0 profiles all).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    config = load_config(args.config)
    if args.seed is not None:
        config.setdefault("clustering", {})["seeds"] = [int(args.seed)]
    artifact_dir = resolve_artifact_dir(config, args.output_dir)
    cohort = load_cohort(config, args.data_root)
    representations = build_representations(cohort, config)

    adapter = PcosAdapter(build_adapter_config(config)).fit(representations, cohort.subset_ids)
    d = adapter.discovery
    assert d is not None

    write_benchmark_csv(d.benchmarks, artifact_dir / "clustering_benchmark.csv")

    ids = list(d.data.participant_ids)
    if args.limit_participants > 0:
        ids = ids[: args.limit_participants]

    profiles = []
    reports = []
    for patient_id in ids:
        standardized = {
            str(c): float(cohort.standardized.loc[patient_id, c])
            for c in cohort.standardized.columns
        }
        phenotype, stability, _ = adapter.profile(patient_id, None, standardized)
        profiles.append(phenotype.model_dump(mode="json"))
        reports.append(stability.model_dump(mode="json"))

    noise = [p for p in d.perturbations if p.scenario.startswith("assay_noise")]
    scaling = [p for p in d.perturbations if p.scenario.startswith("scaling_")]
    imputation = [p for p in d.perturbations if p.scenario.startswith("imputation_")]

    summary = {
        "experiment_id": config.get("experiment_id", "unnamed"),
        "data_source": cohort.source,
        "notes": cohort.notes,
        "selected_k": d.selection.k,
        "selected_representation": d.selection.representation,
        "selected_algorithm": d.selection.algorithm,
        "k_selection_warnings": d.selection.warnings,
        "bootstrap": {
            "n_effective_resamples": d.bootstrap.n_effective_resamples,
            "mean_jaccard": d.bootstrap.mean_jaccard,
            "per_cluster_jaccard": d.bootstrap.per_cluster_jaccard,
            "mean_ari": d.bootstrap.mean_ari,
            "mean_assignment_entropy": float(
                np.mean(list(d.bootstrap.assignment_entropy.values()))
            ),
        },
        "ablation": {
            "flip_rate_by_feature": d.ablation.flip_rate,
            "highest_fragility_feature": d.ablation.highest_fragility_feature,
            "highest_fragility_flip_rate": d.ablation.highest_fragility_flip_rate,
            "flip_rate_by_modality": d.ablation.modality_flip_rate,
        },
        "perturbation": {
            "mean_assay_noise_flip_rate": (
                float(np.mean([p.flip_rate for p in noise])) if noise else None
            ),
            "mean_assay_noise_js_divergence": (
                float(np.mean([p.mean_js_divergence for p in noise])) if noise else None
            ),
            "scaling_flip_rate": {p.scenario: p.flip_rate for p in scaling},
            "imputation_flip_rate": {p.scenario: p.flip_rate for p in imputation},
            "by_scenario": [
                {
                    "scenario": p.scenario,
                    "flip_rate": p.flip_rate,
                    "mean_js_divergence": p.mean_js_divergence,
                    "warnings": p.warnings,
                }
                for p in d.perturbations
            ],
        },
        "calibration": {
            "temperature": d.calibration.temperature,
            "ece_before": d.calibration.ece_before,
            "ece_after": d.calibration.ece_after,
            "mean_confidence_after": d.calibration.mean_confidence_after,
            "mean_bootstrap_agreement": d.calibration.mean_agreement,
            "warnings": d.calibration.warnings,
        },
        "abstention": {
            "n_profiled": len(reports),
            "n_abstained": sum(1 for r in reports if r["abstain"]),
            "reason_counts": _reason_counts(reports),
        },
        "non_diagnostic_statement": NON_DIAGNOSTIC_STATEMENT,
    }

    write_json({**summary, "reports": reports}, artifact_dir / "stability_report.json")
    write_json(
        {
            "experiment_id": config.get("experiment_id", "unnamed"),
            "data_source": cohort.source,
            "notes": cohort.notes,
            "non_diagnostic_statement": NON_DIAGNOSTIC_STATEMENT,
            "profiles": profiles,
        },
        artifact_dir / "phenotype_profile.json",
    )

    print(f"[run_stability_analysis] data source: {cohort.source}")
    for note in cohort.notes:
        print(f"[run_stability_analysis] NOTE: {note}")
    print(
        f"[run_stability_analysis] K={d.selection.k}, "
        f"mean bootstrap Jaccard={d.bootstrap.mean_jaccard:.3f}, "
        f"most fragile feature={d.ablation.highest_fragility_feature}"
    )
    print(
        f"[run_stability_analysis] abstained on "
        f"{summary['abstention']['n_abstained']}/{len(reports)} participants"
    )
    print(f"[run_stability_analysis] artifacts written to {artifact_dir}")
    return 0


def _reason_counts(reports: list[dict]) -> dict[str, int]:
    """Count how often each abstention rule fired across the cohort."""
    counts: dict[str, int] = {}
    for report in reports:
        for reason in report.get("abstain_reasons", []):
            code = reason.split(":", 1)[0]
            counts[code] = counts.get(code, 0) + 1
    return counts


if __name__ == "__main__":
    raise SystemExit(main())
