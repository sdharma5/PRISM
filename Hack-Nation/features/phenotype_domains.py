"""Transparent, coverage-aware composite scores for each phenotype domain.

The score is a documented weighted mean of z-scores, not a learned model:

    s_d = sum_j (w_j * z_j * m_j) / sum_j (w_j * m_j)

where ``m_j`` is 1 only when feature *j* is genuinely observed. Every weight,
sign and reporting threshold comes from ``registry/phenotype_domains.yaml`` so
the definition can be audited and changed without touching code.

These are research phenotype profiles. They are not diagnoses and carry no
clinical decision authority.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from features.missingness import resolve_status
from registry.loader import load_phenotype_domains
from schemas.phenotype import DomainScore

#: Evidence classes that represent patient/clinician *report* rather than assay.
REPORT_EVIDENCE_CLASS = "report"

_MIN_STD = 1e-9


@dataclass(frozen=True)
class DomainFeature:
    """One weighted term of a domain composite."""

    code: str
    weight: float
    direction: int
    evidence_class: str


@dataclass(frozen=True)
class DomainSpec:
    """A whole domain as declared in the registry."""

    name: str
    label: str
    description: str
    min_coverage_to_report: float
    features: tuple[DomainFeature, ...]
    symptom_only_qualifier: str | None = None
    #: What kind of evidence this domain is made of, declared in the registry:
    #: "symptoms", "biochemical", "imaging", or "mixed" when unstated. A reader
    #: must be able to tell a cutaneous sign from a measured androgen level
    #: without inspecting the feature list.
    evidence_source: str = "mixed"

    @property
    def total_weight(self) -> float:
        return float(sum(f.weight for f in self.features))


def load_domain_specs(registry: dict[str, Any] | None = None) -> dict[str, DomainSpec]:
    """Parse ``registry/phenotype_domains.yaml`` into typed specs."""
    raw = registry if registry is not None else load_phenotype_domains()
    specs: dict[str, DomainSpec] = {}
    for name, body in (raw.get("domains") or {}).items():
        features = tuple(
            DomainFeature(
                code=str(item["code"]),
                weight=float(item.get("weight", 1.0)),
                direction=int(item.get("direction", 1)),
                evidence_class=str(item.get("evidence_class", "unspecified")),
            )
            for item in body.get("features", [])
        )
        specs[name] = DomainSpec(
            name=name,
            label=str(body.get("label", name)),
            description=str(body.get("description", "")).strip(),
            min_coverage_to_report=float(body.get("min_coverage_to_report", 0.0)),
            features=features,
            symptom_only_qualifier=body.get("symptom_only_qualifier"),
            evidence_source=str(body.get("evidence_source", "mixed")),
        )
    return specs


@dataclass
class DomainReferenceStats:
    """Per-code mean/std used to z-score. Fitted on training data only."""

    means: dict[str, float] = field(default_factory=dict)
    stds: dict[str, float] = field(default_factory=dict)
    n_observed: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, dict[str, float]]:
        return {
            "means": dict(self.means),
            "stds": dict(self.stds),
            "n_observed": {k: float(v) for k, v in self.n_observed.items()},
        }


class PhenotypeDomainScorer:
    """Fit reference statistics on a cohort, then score patients against them.

    ``fit`` must be called on training rows only; z-scores computed against
    test-fold statistics would leak the evaluation distribution into the score.
    """

    version: str = "1.0.0"

    def __init__(self, registry: dict[str, Any] | None = None) -> None:
        raw = registry if registry is not None else load_phenotype_domains()
        self.registry_version: str = str(raw.get("version", "unknown"))
        self.specs: dict[str, DomainSpec] = load_domain_specs(raw)
        self.stats: DomainReferenceStats | None = None

    # -- Fitting -----------------------------------------------------------

    @property
    def required_codes(self) -> list[str]:
        codes = {f.code for spec in self.specs.values() for f in spec.features}
        return sorted(codes)

    def fit(self, df: pd.DataFrame) -> PhenotypeDomainScorer:
        """Compute per-code mean and std from observed values only."""
        means: dict[str, float] = {}
        stds: dict[str, float] = {}
        counts: dict[str, int] = {}
        for code in self.required_codes:
            if code not in df.columns:
                continue
            values = pd.to_numeric(df[code], errors="coerce")
            values = values[resolve_status(df, code) == "observed"].dropna()
            counts[code] = int(values.size)
            if values.size < 2:
                continue
            std = float(values.std(ddof=0))
            means[code] = float(values.mean())
            stds[code] = std if std > _MIN_STD else 1.0
        self.stats = DomainReferenceStats(means=means, stds=stds, n_observed=counts)
        return self

    def _require_stats(self) -> DomainReferenceStats:
        if self.stats is None:
            raise RuntimeError("PhenotypeDomainScorer.fit() must be called before scoring.")
        return self.stats

    # -- Scoring -----------------------------------------------------------

    def score_row(
        self, row: pd.Series, df_row: pd.DataFrame | None = None
    ) -> dict[str, DomainScore]:
        """Score one patient across all domains."""
        frame = df_row if df_row is not None else row.to_frame().T
        scored = self.score_frame(frame)
        return {domain: scores[0] for domain, scores in scored.items()}

    def score_frame(self, df: pd.DataFrame) -> dict[str, list[DomainScore]]:
        """Score every row of ``df``, returning one ``DomainScore`` list per domain."""
        stats = self._require_stats()
        n = len(df)
        results: dict[str, list[DomainScore]] = {}

        for name, spec in self.specs.items():
            usable = [f for f in spec.features if f.code in df.columns and f.code in stats.means]
            unusable = [f.code for f in spec.features if f not in usable]

            weighted_sum = np.zeros(n, dtype=float)
            weight_mass = np.zeros(n, dtype=float)
            observed_flags: dict[str, np.ndarray] = {}

            for feature in usable:
                values = pd.to_numeric(df[feature.code], errors="coerce")
                observed = (resolve_status(df, feature.code) == "observed").to_numpy() & (
                    values.notna().to_numpy()
                )
                observed_flags[feature.code] = observed
                z = (values.to_numpy(dtype=float) - stats.means[feature.code]) / stats.stds[
                    feature.code
                ]
                z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
                contribution = feature.weight * feature.direction * z * observed
                weighted_sum += contribution
                weight_mass += feature.weight * observed

            total_weight = spec.total_weight or 1.0
            coverage = weight_mass / total_weight
            with np.errstate(invalid="ignore", divide="ignore"):
                raw_score = np.where(
                    weight_mass > 0,
                    weighted_sum / np.maximum(weight_mass, _MIN_STD),
                    np.nan,
                )

            scores: list[DomainScore] = []
            for i in range(n):
                observed_codes = sorted(
                    code for code, flags in observed_flags.items() if bool(flags[i])
                )
                missing_codes = sorted(set(f.code for f in spec.features) - set(observed_codes))
                cov = float(min(max(coverage[i], 0.0), 1.0))
                warnings: list[str] = []
                qualifier: str | None = None

                if unusable:
                    warnings.append(
                        f"{len(unusable)} declared feature(s) had no reference statistics and "
                        "were excluded from the composite."
                    )

                observed_classes = {
                    f.evidence_class for f in spec.features if f.code in set(observed_codes)
                }
                if observed_codes and observed_classes == {REPORT_EVIDENCE_CLASS}:
                    qualifier = spec.symptom_only_qualifier
                    warnings.append(
                        f"Domain '{name}' was scored from self-reported symptoms only; it must "
                        "not be read as biochemical or imaging evidence."
                    )

                reportable = cov >= spec.min_coverage_to_report and np.isfinite(raw_score[i])
                if not reportable:
                    warnings.append(
                        f"Coverage {cov:.2f} is below the registry threshold "
                        f"{spec.min_coverage_to_report:.2f}; no score is reported."
                    )

                scores.append(
                    DomainScore(
                        domain=name,
                        score=float(raw_score[i]) if reportable else None,
                        coverage=cov,
                        observed_features=observed_codes,
                        missing_features=missing_codes,
                        evidence_qualifier=qualifier,
                        evidence_source=spec.evidence_source,
                        warnings=warnings,
                    )
                )
            results[name] = scores
        return results

    def score_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Domain scores as a numeric matrix (``NaN`` where not reportable)."""
        scored = self.score_frame(df)
        data = {
            f"domain_{name}": [s.score if s.score is not None else np.nan for s in scores]
            for name, scores in scored.items()
        }
        data.update(
            {
                f"domain_{name}_coverage": [s.coverage for s in scores]
                for name, scores in scored.items()
            }
        )
        return pd.DataFrame(data, index=df.index)

    def manifest(self) -> dict[str, Any]:
        """Everything needed to reproduce these scores, for the feature manifest."""
        stats = self.stats.to_dict() if self.stats is not None else {}
        return {
            "scorer_version": self.version,
            "registry_version": self.registry_version,
            "formula": "s_d = sum_j(w_j * direction_j * z_j * m_j) / sum_j(w_j * m_j)",
            "coverage_definition": "sum_j(w_j * m_j) / sum_j(w_j) over declared features",
            "domains": {
                name: {
                    "label": spec.label,
                    "min_coverage_to_report": spec.min_coverage_to_report,
                    "symptom_only_qualifier": spec.symptom_only_qualifier,
                    "evidence_source": spec.evidence_source,
                    "features": [
                        {
                            "code": f.code,
                            "weight": f.weight,
                            "direction": f.direction,
                            "evidence_class": f.evidence_class,
                        }
                        for f in spec.features
                    ],
                }
                for name, spec in self.specs.items()
            },
            "reference_statistics": stats,
        }


__all__ = [
    "DomainFeature",
    "DomainReferenceStats",
    "DomainSpec",
    "PhenotypeDomainScorer",
    "load_domain_specs",
]
