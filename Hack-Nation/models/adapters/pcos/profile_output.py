"""The PCOS adapter's output contract.

``learned_components_used`` and ``rule_based_components_used`` are the two most
important fields in this schema. Everything else describes *what* the system
concluded; those two describe *how*, and without them a reader cannot tell a
probability fit against 541 labelled patients from a score produced by weights
someone chose. Both appear in every output, always populated.

``pcos_evidence_probability`` is None whenever the adapter abstains, enforced by
a validator rather than by convention -- a caller must not be able to read a
number the adapter declined to stand behind.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator

__all__ = ["AxisEvidenceOutput", "PCOSProfileOutput"]


class PhenotypeDomainDetail(BaseModel):
    """One phenotype domain with the context needed to read its score."""

    domain: str
    #: Human-readable name from registry/phenotype_domains.yaml, carried here so
    #: a client never has to keep its own domain -> label map in sync.
    label: str | None = None
    #: Cohort z-score, or None when the domain was not assessable. Never 0.0 as
    #: a stand-in for absent: zero is a measurement.
    score: float | None = None
    #: Weight-weighted fraction of the domain's variables that were observed.
    coverage: float | None = Field(default=None, ge=0.0, le=1.0)
    assessable: bool = False
    #: "symptoms" | "biochemical" | "imaging" | "mixed".
    evidence_source: str = "mixed"
    #: Set when the domain rests on reported symptoms alone and must not be read
    #: as biochemical evidence.
    evidence_qualifier: str | None = None
    observed_variables: list[str] = Field(default_factory=list)
    missing_variables: list[str] = Field(default_factory=list)


class AxisEvidenceOutput(BaseModel):
    """Serializable form of one diagnostic axis assessment."""

    axis: str
    level: str
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    #: "met" | "not_met" | "not_assessable" from the guideline assessment.
    axis_status: str = "not_assessable"

    supporting_evidence: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    threshold_sources: dict[str, str] = Field(default_factory=dict)
    assay_dependent: bool = False
    caveats: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class PCOSProfileOutput(BaseModel):
    """PCOS-specific interpretation of coordinated multimodal evidence."""

    patient_id: str

    #: From the learned static clinical head only. None when it did not run or
    #: when the adapter abstained.
    pcos_evidence_probability: float | None = Field(default=None, ge=0.0, le=1.0)

    diagnostic_feature_evidence: dict[str, AxisEvidenceOutput] = Field(default_factory=dict)

    #: The learned static head's score exactly as the model emits it. Preserved
    #: alongside the calibrated score so the two are always comparable and so a
    #: recalibration can never silently rewrite the number earlier results used.
    raw_model_score: float | None = Field(default=None, ge=0.0, le=1.0)
    #: The raw score after the frozen Platt calibrator fitted on out-of-fold
    #: TRAINING predictions. None when no calibrator was available.
    calibrated_model_score: float | None = Field(default=None, ge=0.0, le=1.0)

    # -- Phenotype: continuous domain scores are the PRIMARY output ----------
    #: Continuous domain composites in z-score units. None where the domain
    #: abstained for insufficient coverage -- a missing androgen assay yields
    #: None, never 0.0, which would read as "average". This, not the profile
    #: label, is what the phenotype layer is entitled to assert.
    phenotype_domain_scores: dict[str, float | None] = Field(default_factory=dict)
    #: The same domains with the context a bare number cannot carry: how much of
    #: the domain was actually observed, which variables were and were not, and
    #: the registry label. Kept alongside `phenotype_domain_scores` rather than
    #: replacing it so existing consumers of the flat map keep working.
    #:
    #: A score without its coverage is not interpretable: a domain composited
    #: from one observation and one composited from all of them are the same
    #: z-score to a reader, and only the second means much.
    phenotype_domain_detail: dict[str, PhenotypeDomainDetail] = Field(default_factory=dict)
    #: Domain -> was it assessed at all for this patient. Rendered as
    #: "not assessed" rather than left to be inferred from a missing key.
    domain_assessability: dict[str, bool] = Field(default_factory=dict)
    #: Domain -> "symptoms" | "biochemical" | "imaging" | "mixed".
    domain_evidence_source: dict[str, str] = Field(default_factory=dict)
    #: One of "symptoms_only" | "biochemical_only" | "both" | "unavailable".
    #: Stated explicitly because a symptom-only androgenic score read as a
    #: measured androgen level is the single most consequential misreading of
    #: this output.
    androgenic_evidence_source: str = "unavailable"

    # -- Phenotype: soft profile similarities are SECONDARY, exploratory -----
    #: Affinity scores over research prototypes -- NOT calibrated probabilities
    #: and NOT validated clinical subtypes. They sum to 1 but were never
    #: calibrated against an outcome; sharpness is set by an arbitrary softmax
    #: temperature.
    phenotype_affinities: dict[str, float] = Field(default_factory=dict)
    #: Populated ONLY when the stability engine calls the assignment stable.
    dominant_profile: str | None = None
    #: Raw cosine similarity per profile, so a near-tie is visible. Contains only
    #: eligible profiles: a profile whose defining domain was never assessed is
    #: removed and the rest renormalized, never scored and then suppressed.
    profile_similarities: dict[str, float] = Field(default_factory=dict)
    #: Profiles offered to this patient, and why each of the others was not.
    eligible_profiles: list[str] = Field(default_factory=list)
    ineligible_profiles: dict[str, str] = Field(default_factory=dict)
    assignment_entropy: float | None = None
    #: True whenever no dominant profile may be published.
    indeterminate: bool = True
    indeterminate_reasons: list[str] = Field(default_factory=list)
    #: Stability engine verdict; None when stability was not assessed.
    assignment_is_stable: bool | None = None
    #: Which domains drove each profile's similarity.
    profile_supporting_domains: dict[str, list[str]] = Field(default_factory=dict)

    stability_score: float = Field(default=0.0, ge=0.0, le=1.0)
    subtype_flip_rate: float | None = None
    #: Full per-patient stability report: bootstrap agreement, domain ablation,
    #: modality removal, temperature and threshold sensitivity.
    profile_stability: dict[str, Any] = Field(default_factory=dict)
    #: Structured explanation sections, from explanation.py.
    explanation: dict[str, Any] = Field(default_factory=dict)

    abstain: bool = False
    abstention_reason: str | None = None

    available_modalities: list[str] = Field(default_factory=list)
    missing_modalities: list[str] = Field(default_factory=list)

    agreements: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)

    learned_components_used: list[str] = Field(default_factory=list)
    rule_based_components_used: list[str] = Field(default_factory=list)

    provenance_ids: list[str] = Field(default_factory=list)
    clinician_review_status: str = "model_generated"
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _abstention_withholds_the_probability(self) -> PCOSProfileOutput:
        if self.abstain and self.pcos_evidence_probability is not None:
            raise ValueError(
                "abstain=True requires pcos_evidence_probability=None; otherwise a caller "
                "can read the number the adapter refused to stand behind."
            )
        return self

    @model_validator(mode="after")
    def _method_is_always_declared(self) -> PCOSProfileOutput:
        """Every output must say how it was constructed."""
        if not self.learned_components_used and not self.rule_based_components_used:
            raise ValueError(
                "A profile must declare at least one of learned_components_used or "
                "rule_based_components_used, so a reader can tell what produced it."
            )
        return self

    @model_validator(mode="after")
    def _probability_requires_the_learned_head(self) -> PCOSProfileOutput:
        """Only the learned static head may back a PCOS probability."""
        if self.pcos_evidence_probability is not None and not any(
            "static" in component for component in self.learned_components_used
        ):
            raise ValueError(
                "pcos_evidence_probability was set without the learned static clinical "
                "head in learned_components_used. No rule-based component is entitled "
                "to issue a whole-patient PCOS probability."
            )
        return self

    @model_validator(mode="after")
    def _dominant_profile_requires_stability(self) -> PCOSProfileOutput:
        """A profile label is published only when it survived the stability checks.

        Enforced in the schema rather than in the caller: an assignment that a
        single dropped domain or a different softmax temperature would overturn
        reads exactly like a stable one once it is on a page, and there is no
        way for a downstream consumer to tell them apart after the fact.
        """
        if self.dominant_profile is not None:
            if self.assignment_is_stable is not True:
                raise ValueError(
                    f"dominant_profile='{self.dominant_profile}' requires "
                    "assignment_is_stable=True. An unstable or unassessed assignment must be "
                    "returned as dominant_profile=None with indeterminate=True."
                )
            if self.indeterminate:
                raise ValueError(
                    "indeterminate=True requires dominant_profile=None; a profile cannot be "
                    "both withheld and published."
                )
        elif not self.indeterminate:
            raise ValueError(
                "dominant_profile=None requires indeterminate=True, so a consumer that finds "
                "no profile knows it was withheld rather than simply absent."
            )
        return self

    @model_validator(mode="after")
    def _androgenic_leaning_requires_androgenic_evidence(self) -> PCOSProfileOutput:
        """The specific claim this whole split exists to make impossible."""
        if self.dominant_profile == "androgenic_leaning" and (
            self.androgenic_evidence_source == "unavailable"
        ):
            raise ValueError(
                "dominant_profile='androgenic_leaning' with "
                "androgenic_evidence_source='unavailable': the profile is named for an axis "
                "that was never assessed for this patient."
            )
        return self

    @model_validator(mode="after")
    def _unassessed_domains_are_not_zero_filled(self) -> PCOSProfileOutput:
        """A domain marked unassessable must carry None, never a number."""
        for domain, assessable in self.domain_assessability.items():
            if not assessable and self.phenotype_domain_scores.get(domain) is not None:
                raise ValueError(
                    f"domain '{domain}' is marked unassessable but carries a score. An "
                    "unavailable domain must be None; 0.0 would read as 'average'."
                )
        return self

    @model_validator(mode="after")
    def _calibrated_score_requires_a_raw_score(self) -> PCOSProfileOutput:
        if self.calibrated_model_score is not None and self.raw_model_score is None:
            raise ValueError(
                "calibrated_model_score was set without raw_model_score. Both are kept so a "
                "reader can see what the recalibration did."
            )
        return self

    def write_json(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.model_dump(mode="json"), indent=2) + "\n")
        return path
