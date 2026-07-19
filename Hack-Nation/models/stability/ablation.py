"""Leave-one-feature-out and leave-one-modality-out re-clustering.

Scientific WHY
--------------
Suppose a discovered grouping is really "the participants who had an AMH assay
ordered". Dropping AMH would then destroy the partition, and that is a finding
about ascertainment, not about phenotype. Ablation makes that failure mode
measurable: we re-run the clustering with one feature (or one whole modality)
removed and count how many participants change group.

Two levels matter for different reasons:

* **Leave-one-feature-out** finds the single variable a grouping is hostage to.
  ``highest_fragility_feature`` is exactly the variable a reviewer should be told
  about first, and a per-patient flip under a single-variable removal is one of
  the blueprint's abstention triggers.
* **Leave-one-modality-out** (labs / anthropometry / symptom report / imaging)
  tests whether the structure survives losing an entire acquisition channel —
  which is the realistic missing-data pattern in multimodal hormonal-health data,
  where a participant either got an ultrasound or did not.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from evaluation.stability import subtype_flip_rate
from models.phenotype.clustering import ClusteringInput, fit_base_partition
from models.stability.bootstrap import align_labels

__all__ = [
    "AblationResult",
    "leave_one_feature_out",
    "leave_one_modality_out",
    "run_ablation",
]


@dataclass
class AblationResult:
    """Flip rates from removing features or modalities, one entry per removal."""

    #: removed feature/modality -> fraction of participants that changed group
    flip_rate: dict[str, float] = field(default_factory=dict)
    #: participant_id -> list of removals under which that participant flipped
    per_participant_flips: dict[str, list[str]] = field(default_factory=dict)
    highest_fragility_feature: str | None = None
    highest_fragility_flip_rate: float = 0.0
    modality_flip_rate: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def fragility_for(self, patient_id: str) -> float:
        """Fraction of attempted removals under which this participant flipped."""
        if not self.flip_rate:
            return 0.0
        return len(self.per_participant_flips.get(patient_id, [])) / len(self.flip_rate)


def _refit_and_compare(
    X: np.ndarray,
    keep_columns: list[int],
    reference_labels: np.ndarray,
    algorithm: str,
    k: int,
    seed: int,
) -> np.ndarray | None:
    """Re-cluster on a column subset and return labels aligned to the reference."""
    if not keep_columns:
        return None
    sub = X[:, keep_columns]
    base = "kmeans" if algorithm == "consensus" else algorithm
    try:
        labels = np.asarray(fit_base_partition(sub, base, k, seed))
    except (ValueError, np.linalg.LinAlgError):
        return None
    mapping = align_labels(reference_labels, labels)
    return np.asarray([mapping.get(int(label), -1) for label in labels])


def leave_one_feature_out(
    data: ClusteringInput,
    reference_labels: np.ndarray,
    algorithm: str,
    k: int,
    seed: int = 0,
) -> AblationResult:
    """Re-cluster once per feature with that feature removed."""
    X = np.asarray(data.matrix, dtype=float)
    names = list(data.feature_names) or [f"f{i}" for i in range(X.shape[1])]
    reference_labels = np.asarray(reference_labels)

    result = AblationResult()
    if X.shape[1] < 2:
        result.warnings.append("fewer than 2 features: leave-one-feature-out is undefined")
        return result

    for j, name in enumerate(names):
        keep = [c for c in range(X.shape[1]) if c != j]
        aligned = _refit_and_compare(X, keep, reference_labels, algorithm, k, seed)
        if aligned is None:
            result.warnings.append(f"re-clustering failed without '{name}'")
            continue
        result.flip_rate[name] = subtype_flip_rate(
            [int(v) for v in reference_labels], [int(v) for v in aligned]
        )
        for i, pid in enumerate(data.participant_ids):
            if int(aligned[i]) != int(reference_labels[i]):
                result.per_participant_flips.setdefault(pid, []).append(name)

    if result.flip_rate:
        worst = max(result.flip_rate.items(), key=lambda kv: kv[1])
        result.highest_fragility_feature = worst[0]
        result.highest_fragility_flip_rate = float(worst[1])
    return result


def leave_one_modality_out(
    data: ClusteringInput,
    reference_labels: np.ndarray,
    modality_of: Mapping[str, str],
    algorithm: str,
    k: int,
    seed: int = 0,
) -> dict[str, float]:
    """Re-cluster once per modality with all of that modality's features removed.

    ``modality_of`` maps feature name -> modality name. Features with no mapping
    are treated as modality ``"unmapped"`` rather than silently dropped, so a
    registry gap shows up in the output instead of biasing the result.
    """
    X = np.asarray(data.matrix, dtype=float)
    names = list(data.feature_names) or [f"f{i}" for i in range(X.shape[1])]
    reference_labels = np.asarray(reference_labels)

    modalities: dict[str, list[int]] = {}
    for j, name in enumerate(names):
        modalities.setdefault(modality_of.get(name, "unmapped"), []).append(j)

    out: dict[str, float] = {}
    for modality, columns in modalities.items():
        keep = [c for c in range(X.shape[1]) if c not in set(columns)]
        aligned = _refit_and_compare(X, keep, reference_labels, algorithm, k, seed)
        if aligned is None:
            continue
        out[modality] = subtype_flip_rate(
            [int(v) for v in reference_labels], [int(v) for v in aligned]
        )
    return out


def run_ablation(
    data: ClusteringInput,
    reference_labels: np.ndarray,
    algorithm: str,
    k: int,
    modality_of: Mapping[str, str] | None = None,
    seed: int = 0,
    modalities_to_test: Sequence[str] | None = None,
) -> AblationResult:
    """Full ablation sweep: every feature, then every modality."""
    result = leave_one_feature_out(data, reference_labels, algorithm, k, seed)
    if modality_of:
        modality_rates = leave_one_modality_out(
            data, reference_labels, modality_of, algorithm, k, seed
        )
        if modalities_to_test is not None:
            wanted = set(modalities_to_test)
            modality_rates = {m: v for m, v in modality_rates.items() if m in wanted}
        result.modality_flip_rate = modality_rates
        if not modality_rates:
            result.warnings.append(
                "leave-one-modality-out produced no results: the modality map does not "
                "cover the clustered columns, or removing a modality left no features. "
                "This check was not performed rather than passed."
            )
    return result
