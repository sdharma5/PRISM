"""Schemas for coordinated multimodal evidence and the patient report.

These types carry one invariant that the rest of the package exists to protect:
**a number must always arrive with the method that produced it.** A domain score
built by averaging two rule-weighted encoder outputs and a probability emitted
by a model trained on labelled patients are different kinds of claim, and a
consumer that cannot distinguish them will read the first as if it were the
second.

Hence ``combination_mode``, ``learned_components_used`` and
``rule_based_components_used`` are required parts of the report rather than
documentation, and :class:`DomainEvidence` records which modalities contributed
rather than only the resulting score.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from schemas.evidence import EvidenceConflict
from schemas.modality_token import ModalityToken

__all__ = [
    "CoordinatedEvidence",
    "DomainEvidence",
    "PatientEvidenceReport",
]

#: How cross-modal evidence was combined.
#:
#: ``separate``    -- each encoder reported independently, nothing combined.
#: ``rule_based``  -- combined with the design-rule weights. THE DEFAULT.
#: ``calibrated``  -- weights fit on matched validation data. Not available yet;
#:                    selecting it without such data must raise, not warn.
CombinationMode = Literal["separate", "rule_based", "calibrated"]

AgreementLevel = Literal["strong", "moderate", "conflicting", "single_source", "none"]

#: ``not_combined`` is distinct from ``insufficient_evidence``: the first means
#: the caller asked for per-encoder reporting and no combined score was computed,
#: the second means the evidence was too thin to stand behind. Both withhold the
#: number; only one is a statement about the evidence.
EvidenceLevel = Literal["low", "moderate", "high", "insufficient_evidence", "not_combined"]


class DomainEvidence(BaseModel):
    """Coordinated evidence for one clinical domain."""

    domain: str
    #: None when the domain abstained for lack of evidence mass.
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    level: EvidenceLevel = "insufficient_evidence"

    #: Per-modality contribution, before weighting. Keeps the report auditable:
    #: a reader can recompute the score from these and the published weights.
    modality_scores: dict[str, float] = Field(default_factory=dict)
    supporting_modalities: list[str] = Field(default_factory=list)
    agreement: AgreementLevel = "none"

    supporting_evidence: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _abstention_withholds_the_number(self) -> DomainEvidence:
        if self.level in ("insufficient_evidence", "not_combined") and self.score is not None:
            raise ValueError(
                f"domain '{self.domain}': level '{self.level}' requires score=None, so a "
                "consumer cannot read a number the model declined to stand behind."
            )
        return self


class CoordinatedEvidence(BaseModel):
    """What the PCOS adapter consumes: tokens plus coordinated domain evidence.

    The adapter never sees raw audio, PDFs or pixels -- only encoder output. That
    boundary is what keeps the adapter condition-specific but modality-agnostic.
    """

    patient_id: str

    static_token: ModalityToken | None = None
    ultrasound_token: ModalityToken | None = None
    temporal_token: ModalityToken | None = None

    domain_evidence: dict[str, DomainEvidence] = Field(default_factory=dict)
    conflicts: list[EvidenceConflict] = Field(default_factory=list)

    available_modalities: list[str] = Field(default_factory=list)
    missing_modalities: list[str] = Field(default_factory=list)
    coverage: float = Field(default=0.0, ge=0.0, le=1.0)

    combination_mode: CombinationMode = "rule_based"
    provenance_ids: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    def tokens(self) -> dict[str, ModalityToken]:
        """Available tokens keyed by modality name."""
        pairs = {
            "static_clinical": self.static_token,
            "ovarian_ultrasound": self.ultrasound_token,
            "longitudinal_hormonal_state": self.temporal_token,
        }
        return {name: token for name, token in pairs.items() if token is not None}


class PatientEvidenceReport(BaseModel):
    """The unified, provenance-aware report for one patient."""

    patient_id: str

    available_modalities: list[str] = Field(default_factory=list)
    missing_modalities: list[str] = Field(default_factory=list)
    coverage: float = Field(default=0.0, ge=0.0, le=1.0)

    tokens: dict[str, ModalityToken] = Field(default_factory=dict)
    domain_summary: dict[str, DomainEvidence] = Field(default_factory=dict)

    #: Populated by the PCOS adapter. Absent when only coordination has run.
    pcos_profile: dict = Field(default_factory=dict)

    combination_mode: CombinationMode = "rule_based"
    #: Components whose parameters were fit to labelled data.
    learned_components_used: list[str] = Field(default_factory=list)
    #: Components governed by published thresholds or design-rule weights.
    rule_based_components_used: list[str] = Field(default_factory=list)
    joint_model_used: bool = False

    agreements: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)

    clinician_review_status: str = "model_generated"
    provenance_ids: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _never_claim_joint_training(self) -> PatientEvidenceReport:
        """No combination mode available today constitutes a jointly trained model."""
        if self.joint_model_used:
            raise ValueError(
                "joint_model_used=True is not reachable: no jointly trained multimodal "
                "model exists in this repository (ADR-002). Set it only when a fusion "
                "model has actually been trained on matched multimodal patients."
            )
        return self

    def write_json(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.model_dump(mode="json"), indent=2) + "\n")
        return path
