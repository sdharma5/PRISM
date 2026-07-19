"""Shared config loading, data-path resolution and representation building.

Kept out of the package tree on purpose: this is orchestration glue for the
``scripts/`` entry points, not library code that anything else should import.

Data-path resolution
--------------------
:func:`resolve_data_root` and :func:`resolve_data_path` are the single place
where a dataset location is decided. Both follow the precedence documented in
``scripts/_cli.py``::

    explicit CLI flag  >  $PRISM_DATA_ROOT  >  config file value  >  built-in default

Canonical config keys
---------------------
``data.root``
    Directory holding the dataset. The canonical key; prefer it everywhere.
``data.path``
    Optional path to a specific file, relative to ``data.root`` when relative.

Deprecated config keys
----------------------
These are still read, for backward compatibility with configs written before
the keys were unified. They are resolved only after the canonical keys miss,
and new configs must not use them:

- ``dataset.path``  — used by the Step-5 clustering configs. Use ``data.path``.
- ``root``          — top-level, used by the Step-8/9 data configs. Use ``data.root``.

``configs/`` has been migrated to the canonical keys; the legacy readers remain
for user-authored configs living outside this repository.
"""

from __future__ import annotations

import csv
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.adapters.pmos.phenotype_heads import compute_domain_scores  # noqa: E402
from models.phenotype.clustering import ClusteringInput  # noqa: E402
from schemas.phenotype import ClusteringBenchmark  # noqa: E402
from scripts._cli import DATA_ROOT_ENV, env_path, resolve_output_dir, resolve_path  # noqa: E402

__all__ = [
    "CohortBundle",
    "build_representations",
    "dataset_settings",
    "load_config",
    "load_cohort",
    "resolve_artifact_dir",
    "resolve_data_path",
    "resolve_data_root",
    "write_benchmark_csv",
    "write_json",
]

#: Dotted config keys read for the dataset directory, canonical first.
DATA_ROOT_KEYS: tuple[str, ...] = ("data.root", "root")
#: Dotted config keys read for a specific dataset file, canonical first.
DATA_PATH_KEYS: tuple[str, ...] = ("data.path", "dataset.path")


def _dotted(config: dict[str, Any], dotted: str) -> Any:
    """Read ``a.b.c`` out of a nested mapping, returning None on any miss."""
    value: Any = config
    for part in dotted.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
        if value is None:
            return None
    return value


def dataset_settings(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Merge the canonical ``data:`` block over the deprecated ``dataset:`` block.

    Both spellings are accepted so that configs written before the keys were
    unified keep working; ``data:`` wins on conflict because it is canonical.
    """
    config = config or {}
    merged: dict[str, Any] = {}
    for block in ("dataset", "data"):
        value = config.get(block)
        if isinstance(value, dict):
            merged.update(value)
    return merged


def resolve_data_root(
    config: dict[str, Any] | None = None,
    data_root: str | Path | None = None,
) -> Path | None:
    """Resolve the dataset root directory, or None when nothing is configured.

    Precedence: ``data_root`` (the CLI flag) > ``$PRISM_DATA_ROOT`` > the config
    keys in :data:`DATA_ROOT_KEYS`. Returning None is meaningful and is not an
    error: it means "no real dataset is configured", which is the normal state
    of a fresh clone and is what makes the synthetic fallbacks trigger.
    """
    if data_root is not None:
        return resolve_path(data_root)
    from_env = env_path(DATA_ROOT_ENV)
    if from_env is not None:
        return from_env
    for key in DATA_ROOT_KEYS:
        value = _dotted(config or {}, key)
        if value:
            return resolve_path(str(value))
    return None


def resolve_data_path(
    config: dict[str, Any] | None = None,
    data_root: str | Path | None = None,
    key: str | None = None,
) -> Path | None:
    """Resolve the dataset path for a config, or None when unconfigured.

    Args:
        config: The loaded config mapping.
        data_root: Value of ``--data-root``, if the caller passed one.
        key: A specific dotted config key to read instead of the standard list.

    Returns:
        An absolute path, or None when neither a file nor a root is configured.

    A relative ``data.path`` is resolved against the resolved data root when one
    exists, and against the repository root otherwise. That is what lets a
    single ``--data-root`` flag relocate an entire config's worth of paths
    without editing the config.
    """
    config = config or {}
    root = resolve_data_root(config, data_root)

    keys = (key,) if key else DATA_PATH_KEYS
    for dotted in keys:
        value = _dotted(config, dotted) if dotted else None
        if value:
            candidate = Path(str(value)).expanduser()
            if candidate.is_absolute():
                return candidate
            return resolve_path(candidate, base=root)

    return root


def load_config(path: str | Path) -> dict[str, Any]:
    """Read a YAML experiment config. Every analyst choice must live here."""
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping at the top level.")
    return data


def resolve_artifact_dir(config: dict[str, Any], override: str | Path | None = None) -> Path:
    """Create and return the experiment's artifact directory.

    Thin wrapper over :func:`scripts._cli.resolve_output_dir` so that the Step-5
    scripts share the one precedence rule with every other entry point. The
    legacy ``output.artifact_dir`` key is still honoured.
    """
    return resolve_output_dir(
        config,
        override,
        config_keys=("output.artifact_dir", "output.root"),
    )


@dataclass
class CohortBundle:
    """A standardized feature frame plus the explicit clustering subset."""

    frame: pd.DataFrame
    standardized: pd.DataFrame
    subset_ids: list[str]
    source: str
    notes: list[str]


def _standardize(frame: pd.DataFrame) -> pd.DataFrame:
    """Cohort z-scores, then median-fill. Missingness is recorded, not hidden.

    Standardizing before imputing means an imputed value lands at the cohort
    centre of its own variable rather than dragging the participant toward the
    centre of whichever variable happened to have the largest raw scale.
    """
    mean = frame.mean(numeric_only=True)
    sd = frame.std(numeric_only=True).replace(0.0, 1.0)
    z = (frame - mean) / sd
    return z.fillna(z.median()).fillna(0.0)


def load_cohort(config: dict[str, Any], data_root: str | Path | None = None) -> CohortBundle:
    """Load the real dataset if configured and present; otherwise synthesize one.

    Falling back to synthetic data is what lets these scripts run end to end in
    CI and on a fresh clone. The fallback is announced in the returned ``notes``
    and written into the artifacts, so no synthetic run can be mistaken for a
    result on real data.
    """
    dataset = dataset_settings(config)
    notes: list[str] = []
    features: list[str] | None = dataset.get("features")
    candidate = resolve_data_path(config, data_root)

    if candidate is not None:
        if candidate.exists() and candidate.is_file():
            frame = pd.read_csv(candidate)
            id_column = dataset.get("id_column", "patient_id")
            if id_column in frame.columns:
                frame = frame.set_index(id_column)
            label_column = dataset.get("label_column")
            subset_ids = list(frame.index.astype(str))
            if dataset.get("pmos_positive_only", True) and label_column in frame.columns:
                subset_ids = [str(i) for i in frame.index[frame[label_column].astype(float) == 1.0]]
                notes.append(
                    f"Clustering restricted to {len(subset_ids)} participants positive on "
                    f"'{label_column}'."
                )
            numeric = frame.select_dtypes("number")
            if features:
                numeric = numeric[[c for c in features if c in numeric.columns]]
            numeric.index = numeric.index.astype(str)
            return CohortBundle(numeric, _standardize(numeric), subset_ids, str(candidate), notes)
        notes.append(f"Configured dataset '{candidate}' not found; falling back to synthetic.")

    from tests.fixtures.synthetic_clusters import make_synthetic_cluster_frame

    spec = dataset.get("synthetic", {})
    frame, truth = make_synthetic_cluster_frame(
        n_per_group=int(spec.get("n_per_group", 40)),
        noise=float(spec.get("noise", 0.35)),
        missing_rate=float(spec.get("missing_rate", 0.0)),
        seed=int(spec.get("seed", 0)),
    )
    notes.append(
        "SYNTHETIC DATA: no real dataset was available. These artifacts describe planted "
        f"geometry ({len(set(truth))} groups), not biology, and are not a scientific result."
    )
    return CohortBundle(
        frame, _standardize(frame), list(frame.index.astype(str)), "synthetic", notes
    )


def build_representations(
    cohort: CohortBundle,
    config: dict[str, Any],
) -> list[ClusteringInput]:
    """Build every configured representation of the same participants.

    Representations are the first of the three sweep axes. We build the ones we
    can and skip the rest with a note rather than failing: a missing autoencoder
    embedding should cost us one row of the benchmark, not the whole run.
    """
    wanted = list(config.get("clustering", {}).get("representations", ["raw_standardized"]))
    ids = list(cohort.standardized.index.astype(str))
    out: list[ClusteringInput] = []

    if "raw_standardized" in wanted:
        out.append(
            ClusteringInput(
                label="raw_standardized",
                matrix=cohort.standardized.to_numpy(dtype=float),
                participant_ids=ids,
                feature_names=list(cohort.standardized.columns),
            )
        )

    if "domain_scores" in wanted:
        rows: list[list[float]] = []
        domain_names: list[str] = []
        for pid in ids:
            values = {
                str(c): float(cohort.standardized.loc[pid, c]) for c in cohort.standardized.columns
            }
            scores = compute_domain_scores(values)
            domain_names = list(scores)
            rows.append([s.score if s.score is not None else 0.0 for s in scores.values()])
        matrix = np.asarray(rows, dtype=float)
        if matrix.size and matrix.shape[1] > 1 and np.isfinite(matrix).all():
            out.append(
                ClusteringInput(
                    label="domain_scores",
                    matrix=matrix,
                    participant_ids=ids,
                    feature_names=domain_names,
                )
            )

    subset = config.get("clustering", {}).get("feature_subset", {})
    if "feature_subset" in wanted and subset.get("features"):
        columns = [c for c in subset["features"] if c in cohort.standardized.columns]
        if len(columns) >= 2:
            out.append(
                ClusteringInput(
                    label=str(subset.get("name", "feature_subset")),
                    matrix=cohort.standardized[columns].to_numpy(dtype=float),
                    participant_ids=ids,
                    feature_names=columns,
                )
            )

    embedding_path = config.get("clustering", {}).get("embedding_path")
    if "autoencoder_embedding" in wanted and embedding_path:
        candidate = Path(embedding_path)
        if not candidate.is_absolute():
            candidate = REPO_ROOT / candidate
        if candidate.exists():
            embedding = np.load(candidate)
            if embedding.shape[0] == len(ids):
                out.append(
                    ClusteringInput(
                        label="autoencoder_embedding",
                        matrix=np.asarray(embedding, dtype=float),
                        participant_ids=ids,
                        feature_names=[f"z{i}" for i in range(embedding.shape[1])],
                    )
                )

    if not out:
        raise ValueError("No representation could be built from the configuration.")
    return out


def write_benchmark_csv(benchmarks: Sequence[ClusteringBenchmark], path: Path) -> Path:
    """Write the full (representation, algorithm, K) benchmark table."""
    fields = [
        "representation",
        "algorithm",
        "k",
        "seed",
        "n_samples",
        "silhouette",
        "calinski_harabasz",
        "davies_bouldin",
        "mean_bootstrap_jaccard",
        "mean_ari_across_seeds",
        "mean_nmi_across_seeds",
        "warnings",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for benchmark in benchmarks:
            row = benchmark.model_dump()
            row["warnings"] = "; ".join(row.get("warnings") or [])
            writer.writerow({k: row.get(k) for k in fields})
    return path


def write_json(payload: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    return path
