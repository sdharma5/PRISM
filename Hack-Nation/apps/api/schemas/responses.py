"""The website-facing response schema. Treat this as a public API.

A presentation contract, not a dump of the internal ``PCOSProfileOutput``. That
object changes shape as models change, and carries fields that are misleading
without their thresholds (raw affinities, entropy, flip rates).

Several values need a qualifier to be safe to display -- symptoms-only
androgenic evidence reads as biochemical confirmation without one -- so the two
live in the same object and can't be separated by a careless render.

Rules when extending:

* Every field must be fillable. A field the mapper can't populate is worse than
  an absent one: clients bind to it and render null forever.
* No bare dicts -- ``dict[str, Any]`` generates ``unknown`` in TypeScript.
* Enumerate what's enumerable; ``Literal`` becomes a real union type.
* ``None`` means not measured, ``0.0`` means measured and average.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "AxisView",
    "BranchStatusView",
    "ConflictView",
    "CurrentStateView",
    "DomainScoreView",
    "ErrorResponse",
    "EvidenceStatementView",
    "HormoneEstimateView",
    "ModelStatusResponse",
    "PcosAssessmentView",
    "PhenotypeView",
    "ProvenanceRecordView",
    "SpeechStatusView",
    "WebsitePCOSProfileResponse",
]

#: Bump on anything a client could notice. Additive optional fields don't count;
#: renames, removals and semantic changes do.
RESPONSE_SCHEMA_VERSION = "1.0.0"

#: Coarse on purpose -- the score doesn't support finer discrimination.
EvidenceLevel = Literal["low", "moderate", "elevated", "high", "not_available"]

#: Matches the internal vocabulary one-to-one.
#:
#: ``awaiting_confirmation`` is absent because nothing emits it yet -- add it in
#: the same change that makes something emit it.
AxisStatus = Literal["met", "not_met", "uncertain", "not_assessable"]

#: Declared so a client can't render a z-score of 1.9 as "190%".
ScoreScale = Literal["cohort_z_score"]

PhenotypeStatus = Literal["stable_dominant_profile", "no_stable_dominant_profile"]

StabilityLabel = Literal["stable", "moderately_stable", "unstable", "not_assessed"]

#: Where a displayed claim came from.
EvidenceOrigin = Literal[
    "patient_reported",
    "clinician_confirmed",
    "document_extracted",
    "device_measured",
    "model_estimate",
    "rule_based_interpretation",
]


class _View(BaseModel):
    model_config = ConfigDict(extra="forbid")


# -- assessment ------------------------------------------------------------


class PcosAssessmentView(_View):
    """The learned whole-patient score, with the conditions for reading it."""

    available: bool
    raw_model_score: float | None = None
    calibrated_model_score: float | None = None
    evidence_level: EvidenceLevel = "not_available"
    #: Always static-clinical when present -- the only branch trained on whole
    #: patients.
    source: str | None = None
    #: Text that must accompany the score wherever it is shown.
    qualifier: str | None = None
    #: Populated only when `available` is False, and always populated then.
    unavailable_reason: str | None = None
    #: Fraction of the model's features observed rather than imputed. The same
    #: score means very different things at 16% and 90%.
    feature_coverage: float | None = Field(default=None, ge=0.0, le=1.0)
    calibrated: bool = False


# -- axes ------------------------------------------------------------------


class AxisView(_View):
    """One Rotterdam axis."""

    status: AxisStatus
    level: str | None = None
    supporting_evidence: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    #: `symptoms_only`, `biochemical_only`, `both`, `unavailable`, or None for
    #: axes where the distinction does not apply.
    evidence_source: str | None = None
    biochemical_evidence_available: bool | None = None
    reason: str | None = None
    caveats: list[str] = Field(default_factory=list)
    #: Threshold expression -> the guideline text it comes from. Lets a client
    #: show why a threshold is what it is instead of asserting it bare.
    threshold_sources: dict[str, str] = Field(default_factory=dict)


# -- phenotype -------------------------------------------------------------


class DomainScoreView(_View):
    """One phenotype domain. ``score`` is a cohort z-score -- not a probability,
    not a percentage. ``None`` means never measured; 0.0 means exactly average."""

    #: Registry label, so clients don't keep their own key -> label map.
    label: str | None = None
    score: float | None = None
    scale: ScoreScale = "cohort_z_score"
    available: bool = False
    #: Weight-weighted fraction of the domain's variables that were observed.
    coverage: float | None = Field(default=None, ge=0.0, le=1.0)
    #: "symptoms" | "biochemical" | "imaging" | "mixed".
    evidence_source: str | None = None
    #: Set when the domain rests on reported symptoms alone.
    qualifier: str | None = None
    observed_variables: list[str] = Field(default_factory=list)
    missing_variables: list[str] = Field(default_factory=list)
    #: Dict iteration order isn't a contract; sort by this.
    display_order: int = 0


class StabilityView(_View):
    """How fragile the profile assignment is, in plain terms first."""

    label: StabilityLabel = "not_assessed"
    plain_language: str | None = None
    stability_score: float | None = None
    bootstrap_agreement: float | None = None
    profile_flip_rate: float | None = None
    #: Domains whose removal flips the assignment (§11).
    unstable_domains: list[str] = Field(default_factory=list)
    withheld_reason: str | None = None


class PhenotypeView(_View):
    """Continuous domains first; soft similarities second."""

    domain_scores: dict[str, DomainScoreView] = Field(default_factory=dict)
    profile_similarities: dict[str, float] = Field(default_factory=dict)
    dominant_profile: str | None = None
    stable_dominant_profile: bool = False
    indeterminate: bool = True
    indeterminate_reasons: list[str] = Field(default_factory=list)
    status: PhenotypeStatus = "no_stable_dominant_profile"
    stability: StabilityView = Field(default_factory=StabilityView)
    #: In the payload rather than the client so a redesign can't drop it.
    interpretation_note: str = (
        "Phenotype profiles are exploratory similarities to patterns described "
        "in the literature, not validated clinical subtypes."
    )


# -- current state ---------------------------------------------------------


class HormoneEstimateView(_View):
    """One hormone estimate, with the method that produced it."""

    #: Canonical code, not a display abbreviation, so it joins to events.
    code: str
    display_name: str
    value: float | None = None
    #: Human-readable method, translated from `locf` / `ridge_window` / `logistic`.
    method: str | None = None
    method_code: str | None = None
    interval_low: float | None = None
    interval_high: float | None = None
    unit: str | None = None


class CurrentStateView(_View):
    """The longitudinal branch's read on where the patient is right now."""

    available: bool = False
    predicted_cycle_phase: str | None = None
    cycle_phase_probabilities: dict[str, float] = Field(default_factory=dict)
    hormone_estimates: dict[str, HormoneEstimateView] = Field(default_factory=dict)
    input_coverage: float | None = None
    methods_used: list[str] = Field(default_factory=list)
    #: Encoder self-reported confidence for this branch, 0-1.
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    observed_days: int | None = None
    unavailable_reason: str | None = None


# -- evidence and provenance ----------------------------------------------


class EvidenceStatementView(_View):
    """Supporting evidence, split up rather than a raw string.

    Internally these are threshold expressions like ``"cycle_length > 35.0"`` --
    machine expressions, not sentences, so the parts are separated here and the
    client renders what it needs.
    """

    statement: str
    variable_code: str | None = None
    axis: str | None = None
    guideline_source: str | None = None


class ConflictView(_View):
    """Two sources disagreeing about the same variable."""

    detail: str
    variable_code: str | None = None
    modalities: list[str] = Field(default_factory=list)
    severity: str | None = None


class ProvenanceRecordView(_View):
    """Where one displayed claim came from, for the detail drawer."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    label: str
    origin: EvidenceOrigin
    source_id: str | None = None
    observed_at: str | None = None
    confirmation_status: str | None = None
    model_version: str | None = None
    method: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class ProvenanceView(_View):
    """Traceability for the report as a whole."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    records: list[ProvenanceRecordView] = Field(default_factory=list)
    provenance_ids: list[str] = Field(default_factory=list)
    model_versions: dict[str, str] = Field(default_factory=dict)
    combination_mode: str | None = None
    clinician_review_status: str | None = None


# -- top level -------------------------------------------------------------


class WebsitePCOSProfileResponse(_View):
    """Everything one patient-facing report needs, and nothing the UI must guess."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    #: Contract version. A client should refuse a major version it does not know
    #: rather than render fields it may be misreading.
    schema_version: str = RESPONSE_SCHEMA_VERSION
    #: Stable id for this report, for support and for provenance references.
    report_id: str
    patient_id: str
    generated_at: str

    #: Fraction of branches that contributed. Distinct from
    #: `pcos_assessment.feature_coverage`, which is about one model's inputs.
    modality_coverage: float | None = Field(default=None, ge=0.0, le=1.0)

    pcos_assessment: PcosAssessmentView
    rotterdam_axes: dict[str, AxisView] = Field(default_factory=dict)
    phenotype: PhenotypeView = Field(default_factory=PhenotypeView)
    current_state: CurrentStateView = Field(default_factory=CurrentStateView)

    #: "symptoms_only" | "biochemical_only" | "both" | "unavailable", from the
    #: phenotype domains. Distinct from an axis's own `evidence_source`: the two
    #: can disagree, and attaching this to an axis reads as a contradiction.
    androgenic_evidence_source: str | None = None

    supporting_evidence: list[EvidenceStatementView] = Field(default_factory=list)
    conflicting_evidence: list[ConflictView] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)

    available_modalities: list[str] = Field(default_factory=list)
    missing_modalities: list[str] = Field(default_factory=list)

    learned_components_used: list[str] = Field(default_factory=list)
    rule_based_components_used: list[str] = Field(default_factory=list)

    provenance: ProvenanceView = Field(default_factory=ProvenanceView)
    warnings: list[str] = Field(default_factory=list)

    #: In the payload so any client, including ones we didn't write, gets the
    #: disclaimer with the numbers it qualifies.
    is_diagnosis: Literal[False] = False
    disclaimer: str = (
        "Research prototype. This is not a diagnosis and does not establish "
        "whether you have PCOS. Discuss these results with a clinician."
    )


# -- model status ----------------------------------------------------------


class BranchStatusView(_View):
    """One model branch's status.

    ``trained`` and ``validated_for_inference`` are separate because of
    ultrasound: the checkpoint loads, but its follicle Dice is oracle-assisted.
    """

    available: bool
    trained: bool
    persisted: bool
    validated_for_inference: bool
    version: str | None = None
    implementation: str | None = None
    reason: str | None = None


class CalibrationStatusView(_View):
    available: bool
    method: str | None = None
    note: str | None = None


class SpeechStatusView(_View):
    """Whether voice transcription is usable.

    Separate from the model branches: those are checkpoints this repo ships,
    this needs `faster-whisper` on the host. The frontend disables the recorder
    from it rather than letting a recording fail on upload.
    """

    available: bool
    model: str | None = None
    reason: str | None = None


class ModelStatusResponse(_View):
    """Body of ``GET /api/v1/models/status``. Fully typed -- the frontend keys
    off this to disable features, so shape shouldn't be guesswork."""

    schema_version: str = RESPONSE_SCHEMA_VERSION
    static_clinical: BranchStatusView
    temporal_state: BranchStatusView
    ovarian_ultrasound: BranchStatusView
    calibration: CalibrationStatusView
    speech: SpeechStatusView | None = None
    warnings: list[str] = Field(default_factory=list)


class ErrorResponse(_View):
    """One error shape for every failing route -- FastAPI's default ``detail``
    can be a string or an object, so clients end up handling both."""

    error: str
    message: str
    detail: dict[str, str | bool | None] = Field(default_factory=dict)
