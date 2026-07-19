#!/usr/bin/env python
"""Discover exploratory phenotype profiles and write the benchmark evidence.

    python scripts/discover_phenotypes.py --config configs/experiments/exp_subtype_stability.yaml

Runs the (representation x algorithm x K) sweep on the configured PMOS-positive
training subset, selects K on measured evidence, characterizes the resulting
groups in hedged language, and writes every artifact needed to audit the choice —
including the full benchmark table, so a reader can see the configurations that
lost as well as the one that won. Falls back to synthetic data when no real
dataset is configured or present, and says so loudly in the artifacts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _cli import add_deprecated_alias, add_standard_arguments, make_parser  # noqa: E402
from _experiment_io import (  # noqa: E402
    build_representations,
    dataset_settings,
    load_cohort,
    load_config,
    resolve_artifact_dir,
    write_benchmark_csv,
    write_json,
)

from models.adapters.pmos.adapter import PmosAdapter, PmosAdapterConfig  # noqa: E402
from models.adapters.pmos.output_schema import NON_DIAGNOSTIC_STATEMENT  # noqa: E402


def build_adapter_config(config: dict) -> PmosAdapterConfig:
    """Translate the YAML into the adapter's inspectable config object."""
    clustering = config.get("clustering", {})
    stability = config.get("stability", {})
    return PmosAdapterConfig(
        algorithms=tuple(clustering.get("algorithms", ("kmeans", "gaussian_mixture"))),
        k_values=tuple(int(k) for k in clustering.get("k_values", (2, 3, 4, 5, 6))),
        seeds=tuple(int(s) for s in clustering.get("seeds", (0, 1, 2))),
        n_bootstrap=int(stability.get("n_bootstrap", 20)),
        consensus_resamples=int(clustering.get("consensus_resamples", 30)),
        n_noise_replicates=int(stability.get("n_noise_replicates", 5)),
        modality_of=dict(config.get("modalities", {})),
        source_dataset=str(dataset_settings(config).get("path") or "synthetic"),
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser. Exposed so the CLI contract test can inspect it."""
    parser = make_parser(description=__doc__)
    add_standard_arguments(parser, experiment_id=False, quiet=False)
    # '--artifact-dir' was this script's original name for '--output-dir'.
    add_deprecated_alias(
        parser, "--artifact-dir", dest="output_dir", replacement="--output-dir", type=Path
    )
    parser.add_argument(
        "--limit-profiles",
        type=int,
        default=25,
        help="How many per-participant profiles to write (0 writes all).",
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

    adapter = PmosAdapter(build_adapter_config(config)).fit(representations, cohort.subset_ids)
    discovery = adapter.discovery
    assert discovery is not None

    write_benchmark_csv(discovery.benchmarks, artifact_dir / "clustering_benchmark.csv")

    write_json(
        {
            "experiment_id": config.get("experiment_id", "unnamed"),
            "data_source": cohort.source,
            "notes": cohort.notes,
            "n_clustered_participants": len(discovery.data.participant_ids),
            "selected_k": discovery.selection.k,
            "selected_representation": discovery.selection.representation,
            "selected_algorithm": discovery.selection.algorithm,
            "k_selection_rationale": discovery.selection.rationale,
            "k_selection_warnings": discovery.selection.warnings,
            "ranked_configurations": [
                {"representation": r, "algorithm": a, "k": k, "score": s}
                for r, a, k, s in discovery.selection.ranked[:20]
            ],
            "profiles": {
                name: {
                    "n_members": char.n_members,
                    "description": char.description,
                    "elevated": char.elevated,
                    "reduced": char.reduced,
                    "enrichment": char.enrichment,
                    "resembles_research_profile": (
                        discovery.prototype_names[name].profile_name
                        if name in discovery.prototype_names
                        else None
                    ),
                    "prototype_similarity": (
                        discovery.prototype_names[name].similarity
                        if name in discovery.prototype_names
                        else None
                    ),
                }
                for name, char in discovery.characterizations.items()
            },
            "cohort_mean_bootstrap_jaccard": discovery.bootstrap.mean_jaccard,
            "calibration_temperature": discovery.calibration.temperature,
            "non_diagnostic_statement": NON_DIAGNOSTIC_STATEMENT,
        },
        artifact_dir / "discovery_summary.json",
    )

    ids = discovery.data.participant_ids
    if args.limit_profiles > 0:
        ids = ids[: args.limit_profiles]

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
    write_json(
        {
            "experiment_id": config.get("experiment_id", "unnamed"),
            "data_source": cohort.source,
            "cohort_mean_bootstrap_jaccard": discovery.bootstrap.mean_jaccard,
            "non_diagnostic_statement": NON_DIAGNOSTIC_STATEMENT,
            "reports": reports,
        },
        artifact_dir / "stability_report.json",
    )

    abstained = sum(1 for r in reports if r["abstain"])
    print(f"[discover_phenotypes] data source: {cohort.source}")
    for note in cohort.notes:
        print(f"[discover_phenotypes] NOTE: {note}")
    print(
        f"[discover_phenotypes] selected K={discovery.selection.k} "
        f"({discovery.selection.representation} / {discovery.selection.algorithm}) "
        f"over {len(discovery.benchmarks)} configurations"
    )
    print(f"[discover_phenotypes] abstained on {abstained}/{len(reports)} profiled participants")
    print(f"[discover_phenotypes] artifacts written to {artifact_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
