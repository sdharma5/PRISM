"""Phenotype-domain scores, soft subtype profiles, and stability reports.

Language contract (see ADR-003): discovered groups are *profiles* that a patient
may "resemble". They are never described as validated clinical subtypes.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

INDETERMINATE = "indeterminate"


class DomainScore(BaseModel):
    """A transparent, coverage-aware composite score for one phenotype domain."""

    domain: str
    score: float | None = None
    coverage: float = Field(ge=0.0, le=1.0)
    observed_features: list[str] = Field(default_factory=list)
    missing_features: list[str] = Field(default_factory=list)
    #: Set when the domain is scored from symptoms only and must not be read as
    #: biochemical evidence (e.g. "androgenic-symptom evidence").
    evidence_qualifier: str | None = None
    #: What kind of evidence this domain is made of, from the registry:
    #: "symptoms" | "biochemical" | "imaging" | "mixed". Carried on the score
    #: itself so a consumer never has to re-derive it from the feature list.
    evidence_source: str = "mixed"
    warnings: list[str] = Field(default_factory=list)

    @property
    def is_reportable(self) -> bool:
        return self.score is not None and self.coverage > 0.0

    @property
    def is_assessable(self) -> bool:
        """Whether this domain was measured at all for this patient.

        Distinct from ``is_reportable``: a domain can be assessable in principle
        and still fall under its coverage floor. Consumers use this to render
        "not assessed" rather than silently treating the domain as average.
        """
        return self.score is not None


class PhenotypeProfile(BaseModel):
    """Soft, probabilistic membership over discovered profiles."""

    patient_id: str
    domain_scores: dict[str, DomainScore] = Field(default_factory=dict)
    phenotype_probabilities: dict[str, float] = Field(default_factory=dict)
    dominant_profile: str | None = None
    dominant_probability: float | None = None
    representation: str = "domain_scores"
    n_profiles: int | None = None
    model_version: str = "0.1.0"
    warnings: list[str] = Field(default_factory=list)

    def normalized(self) -> dict[str, float]:
        total = sum(self.phenotype_probabilities.values())
        if total <= 0:
            return {INDETERMINATE: 1.0}
        return {k: v / total for k, v in self.phenotype_probabilities.items()}


class StabilityReport(BaseModel):
    """How much a profile assignment survives resampling and ablation."""

    patient_id: str
    dominant_profile: str
    dominant_probability: float
    stability_score: float = Field(ge=0.0, le=1.0)
    subtype_flip_rate: float = Field(ge=0.0, le=1.0)
    assignment_entropy: float = 0.0
    bootstrap_jaccard: float | None = None
    highest_fragility_feature: str | None = None
    fragility_by_feature: dict[str, float] = Field(default_factory=dict)
    abstain: bool = False
    abstain_reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ClusteringBenchmark(BaseModel):
    """One (representation, algorithm, K) configuration and its metrics."""

    representation: str
    algorithm: str
    k: int
    seed: int
    silhouette: float | None = None
    calinski_harabasz: float | None = None
    davies_bouldin: float | None = None
    mean_bootstrap_jaccard: float | None = None
    mean_ari_across_seeds: float | None = None
    mean_nmi_across_seeds: float | None = None
    n_samples: int = 0
    warnings: list[str] = Field(default_factory=list)
