"""Measurement-noise, scaling and imputation perturbations of a clustering.

Scientific WHY
--------------
Every number we cluster on is an estimate with an error bar. A testosterone
result of 2.4 nmol/L from an immunoassay could plausibly have come back as 2.1 or
2.7 from the same tube. If a participant's assigned group changes between those
two values, the assignment is reporting assay noise, not phenotype.

We perturb along three axes that are all analyst choices or measurement facts,
never biology:

1. **Assay noise** — each variable is jittered by its documented analytical
   coefficient of variation (CV). See :data:`ASSAY_CV` for values and sources.
2. **Alternative scaling** — standard vs robust (median/IQR) vs min-max. Robust
   scaling changes cluster geometry substantially whenever the cohort has
   outliers, which hormonal panels always do.
3. **Alternative imputation** — mean vs median vs KNN. Imputation choice is one
   of the least-reported and most consequential degrees of freedom in this
   literature.

Outputs are the **subtype flip rate** and the **Jensen-Shannon divergence**
between the original and perturbed membership distributions: the first says
whether the headline label moved, the second catches the case where the label
survived but the confidence behind it did not.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from evaluation.stability import jensen_shannon_divergence, subtype_flip_rate
from models.phenotype.clustering import ClusteringInput, fit_base_partition
from models.stability.bootstrap import align_labels

__all__ = [
    "ASSAY_CV",
    "DEFAULT_CV",
    "PerturbationResult",
    "apply_assay_noise",
    "impute_matrix",
    "run_perturbations",
    "scale_matrix",
]

#: Analytical (within-laboratory) coefficients of variation, as fractions.
#:
#: These are order-of-magnitude values typical of routine clinical chemistry and
#: immunoassay platforms, compiled for *sensitivity analysis only* — they are not
#: a specification of any particular laboratory's performance, and any real
#: deployment must substitute its own verified assay CVs.
#:
#: Sources of the ranges these are drawn from: manufacturer package-insert
#: precision studies for automated immunoassays, published biological-variation
#: databases (e.g. the EFLM Biological Variation Database), and NHANES laboratory
#: procedure manuals for the anthropometric and chemistry panels.
ASSAY_CV: dict[str, float] = {
    # Steroid hormones: immunoassay imprecision is high at female concentrations.
    "total_testosterone": 0.12,
    "free_testosterone": 0.15,
    "dheas": 0.08,
    "shbg": 0.07,
    "estradiol": 0.10,
    "progesterone": 0.10,
    # Gonadotropins and AMH.
    "luteinizing_hormone": 0.08,
    "follicle_stimulating_hormone": 0.07,
    "lh_fsh_ratio": 0.11,
    "anti_mullerian_hormone": 0.10,
    # Glycemic and lipid chemistry: automated, comparatively precise.
    "fasting_glucose": 0.03,
    "fasting_insulin": 0.08,
    "homa_ir": 0.09,
    "hdl_cholesterol": 0.04,
    "ldl_cholesterol": 0.05,
    "triglycerides": 0.05,
    # Anthropometry and vitals: observer/technique variation, not assay.
    "bmi": 0.02,
    "weight": 0.01,
    "height": 0.01,
    "waist_circumference": 0.03,
    "hip_circumference": 0.03,
    "waist_hip_ratio": 0.04,
    "systolic_blood_pressure": 0.05,
    "diastolic_blood_pressure": 0.05,
    # Imaging counts: inter-observer variation dominates.
    "follicle_number_per_ovary": 0.15,
    "follicle_count_left": 0.15,
    "follicle_count_right": 0.15,
    "ovary_volume_ml": 0.12,
    # Semi-quantitative clinical scoring.
    "ferriman_gallwey_score": 0.15,
}

#: Used for any variable with no entry above. Deliberately non-zero: an unknown
#: assay is not a perfect assay.
DEFAULT_CV: float = 0.05


@dataclass
class PerturbationResult:
    """Flip rate and distributional drift for one perturbation scenario."""

    scenario: str
    flip_rate: float
    mean_js_divergence: float
    per_participant_flipped: dict[str, bool] = field(default_factory=dict)
    per_participant_js: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def apply_assay_noise(
    X: np.ndarray,
    feature_names: Sequence[str],
    rng: np.random.Generator,
    cv_by_feature: Mapping[str, float] | None = None,
    scale_is_standardized: bool = True,
) -> np.ndarray:
    """Add multiplicative-CV-equivalent Gaussian noise to each column.

    When the matrix has already been standardized (the usual case for a
    clustering representation), a *relative* CV cannot be applied multiplicatively
    — z-scores cross zero. We therefore convert the CV into an additive standard
    deviation in z-units by multiplying it by the column's own standard deviation,
    which preserves the relative ordering of noisy and precise variables.
    """
    cv_map = dict(ASSAY_CV if cv_by_feature is None else cv_by_feature)
    X = np.asarray(X, dtype=float)
    out = X.copy()
    for j, name in enumerate(feature_names):
        cv = float(cv_map.get(name, DEFAULT_CV))
        if scale_is_standardized:
            sigma = cv * float(np.nanstd(X[:, j]) or 1.0)
            out[:, j] = X[:, j] + rng.normal(0.0, sigma, size=X.shape[0])
        else:
            out[:, j] = X[:, j] * (1.0 + rng.normal(0.0, cv, size=X.shape[0]))
    return out


def scale_matrix(X: np.ndarray, strategy: str = "standard") -> np.ndarray:
    """Re-scale a matrix with an alternative, equally defensible strategy."""
    if strategy == "standard":
        return np.asarray(StandardScaler().fit_transform(X), dtype=float)
    if strategy == "robust":
        return np.asarray(RobustScaler().fit_transform(X), dtype=float)
    if strategy == "minmax":
        return np.asarray(MinMaxScaler().fit_transform(X), dtype=float)
    if strategy == "none":
        return np.asarray(X, dtype=float)
    raise ValueError(f"Unknown scaling strategy '{strategy}'.")


def impute_matrix(X: np.ndarray, strategy: str = "median", n_neighbors: int = 5) -> np.ndarray:
    """Fill missing values with an alternative, equally defensible strategy."""
    X = np.asarray(X, dtype=float)
    if not np.isnan(X).any():
        return X
    if strategy == "knn":
        imputer: SimpleImputer | KNNImputer = KNNImputer(
            n_neighbors=min(n_neighbors, max(1, X.shape[0] - 1))
        )
    elif strategy in {"mean", "median", "most_frequent"}:
        imputer = SimpleImputer(strategy=strategy)
    elif strategy == "zero":
        return np.nan_to_num(X, nan=0.0)
    else:
        raise ValueError(f"Unknown imputation strategy '{strategy}'.")
    return np.asarray(imputer.fit_transform(X), dtype=float)


def _membership_from_distance(X: np.ndarray, centers: np.ndarray, temperature: float) -> np.ndarray:
    """Softmax over negative squared distance to each centre."""
    d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    logits = -d2 / max(temperature, 1e-6)
    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    return weights / weights.sum(axis=1, keepdims=True)


def _centers(X: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return np.vstack([X[labels == label].mean(axis=0) for label in np.unique(labels)])


def _score_scenario(
    scenario: str,
    data: ClusteringInput,
    X_perturbed: np.ndarray,
    reference_labels: np.ndarray,
    algorithm: str,
    k: int,
    seed: int,
    temperature: float,
) -> PerturbationResult:
    """Re-cluster a perturbed matrix and compare labels and memberships."""
    base = "kmeans" if algorithm == "consensus" else algorithm
    X0 = np.asarray(data.matrix, dtype=float)
    try:
        labels = np.asarray(fit_base_partition(X_perturbed, base, k, seed))
    except (ValueError, np.linalg.LinAlgError) as exc:
        return PerturbationResult(scenario, 0.0, 0.0, warnings=[f"re-clustering failed: {exc}"])

    mapping = align_labels(reference_labels, labels)
    aligned = np.asarray([mapping.get(int(label), -1) for label in labels])

    p_before = _membership_from_distance(X0, _centers(X0, reference_labels), temperature)
    p_after = _membership_from_distance(X_perturbed, _centers(X_perturbed, labels), temperature)
    order = [
        list(np.unique(labels)).index(c) if c in list(np.unique(labels)) else None
        for c in np.unique(reference_labels)
    ]
    p_after_aligned = np.zeros_like(p_before)
    for target, source in enumerate(order):
        if source is not None and source < p_after.shape[1]:
            p_after_aligned[:, target] = p_after[:, source]

    flipped: dict[str, bool] = {}
    js: dict[str, float] = {}
    for i, pid in enumerate(data.participant_ids):
        flipped[pid] = int(aligned[i]) != int(reference_labels[i])
        js[pid] = jensen_shannon_divergence(p_before[i], p_after_aligned[i])

    return PerturbationResult(
        scenario=scenario,
        flip_rate=subtype_flip_rate([int(v) for v in reference_labels], [int(v) for v in aligned]),
        mean_js_divergence=float(np.mean(list(js.values()))) if js else 0.0,
        per_participant_flipped=flipped,
        per_participant_js=js,
    )


def run_perturbations(
    data: ClusteringInput,
    reference_labels: np.ndarray,
    algorithm: str,
    k: int,
    n_noise_replicates: int = 10,
    scaling_strategies: Sequence[str] = ("robust", "minmax"),
    imputation_strategies: Sequence[str] = ("mean", "median", "knn"),
    cv_by_feature: Mapping[str, float] | None = None,
    seed: int = 0,
    temperature: float = 1.0,
) -> list[PerturbationResult]:
    """Run every perturbation scenario and return one result per scenario.

    Noise replicates are reported individually rather than averaged into one
    number so that the *spread* of flip rates across replicates is visible; a
    scenario list is easier to aggregate later than an average is to decompose.
    """
    X = np.asarray(data.matrix, dtype=float)
    names = list(data.feature_names) or [f"f{i}" for i in range(X.shape[1])]
    reference_labels = np.asarray(reference_labels)
    rng = np.random.default_rng(seed)
    results: list[PerturbationResult] = []

    for r in range(n_noise_replicates):
        noisy = apply_assay_noise(X, names, rng, cv_by_feature)
        results.append(
            _score_scenario(
                f"assay_noise_{r}",
                data,
                noisy,
                reference_labels,
                algorithm,
                k,
                seed + r,
                temperature,
            )
        )

    for strategy in scaling_strategies:
        try:
            rescaled = scale_matrix(X, strategy)
        except ValueError as exc:
            results.append(PerturbationResult(f"scaling_{strategy}", 0.0, 0.0, warnings=[str(exc)]))
            continue
        results.append(
            _score_scenario(
                f"scaling_{strategy}",
                data,
                rescaled,
                reference_labels,
                algorithm,
                k,
                seed,
                temperature,
            )
        )

    for strategy in imputation_strategies:
        try:
            imputed = impute_matrix(X, strategy)
        except ValueError as exc:
            results.append(
                PerturbationResult(f"imputation_{strategy}", 0.0, 0.0, warnings=[str(exc)])
            )
            continue
        results.append(
            _score_scenario(
                f"imputation_{strategy}",
                data,
                imputed,
                reference_labels,
                algorithm,
                k,
                seed,
                temperature,
            )
        )

    return results
