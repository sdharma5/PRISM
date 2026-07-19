"""The final PCOS research output contract.

Scientific WHY
--------------
The structure of an output shapes how it is read. This model is arranged so the
four things a reader most needs are impossible to skip:

* what was actually **observed** (and, equally, what was **not**);
* how the model **organized** the participant, in hedged language;
* how **uncertain** that organization is, including abstention;
* an explicit **non-diagnostic statement**, which is a required field with no
  default that can be emptied.

``missing_evidence`` is mandatory rather than optional because absence of a
measurement is the single most under-reported fact in phenotyping outputs, and
because a not-assessable axis and a not-met axis look identical once serialized
unless something forces the distinction into the schema.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

NON_DIAGNOSTIC_STATEMENT = (
    "This is a research output. It does not diagnose polycystic ovary syndrome or any "
    "other condition, it is not medical advice, and it has not been validated for "
    "clinical use. The groups described here were discovered by an unsupervised model "
    "in one research cohort; they are exploratory patterns, not established clinical "
    "subtypes. Any clinical question must be directed to a qualified clinician."
)


class ObservedEvidence(BaseModel):
    """What was measured, and what each recognized feature axis showed."""

    observed_variables: list[str] = Field(default_factory=list)
    #: axis name -> "met" | "not_met" | "not_assessable"
    axis_status: dict[str, str] = Field(default_factory=dict)
    axis_evidence_available: dict[str, bool] = Field(default_factory=dict)
    #: The documented provenance of every threshold that was applied.
    threshold_sources: dict[str, str] = Field(default_factory=dict)
    assay_dependent_axes: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)


class MissingEvidence(BaseModel):
    """What was not measured, and which conclusions that blocks."""

    missing_variables: list[str] = Field(default_factory=list)
    not_assessable_axes: list[str] = Field(default_factory=list)
    domains_below_coverage: list[str] = Field(default_factory=list)
    n_defining_features_observed: int = 0
    n_defining_features_total: int = 0
    notes: list[str] = Field(default_factory=list)


class ModelOrganizedPhenotype(BaseModel):
    """How the unsupervised model grouped this participant. Hedged by construction."""

    representation: str
    algorithm: str
    n_profiles: int
    #: Soft membership including the ``indeterminate`` outcome.
    profile_probabilities: dict[str, float] = Field(default_factory=dict)
    dominant_profile: str | None = None
    dominant_probability: float | None = None
    #: e.g. "resembles the Metabolic-leaning research profile"
    resemblance_statement: str | None = None
    profile_descriptions: dict[str, str] = Field(default_factory=dict)
    k_selection_rationale: str | None = None


class UncertaintyReport(BaseModel):
    """Every number that says how much to trust the grouping above."""

    assignment_entropy: float = 0.0
    stability_score: float = 0.0
    subtype_flip_rate: float = 0.0
    bootstrap_jaccard: float | None = None
    cohort_mean_bootstrap_jaccard: float | None = None
    calibration_temperature: float | None = None
    expected_calibration_error: float | None = None
    highest_fragility_feature: str | None = None
    mean_perturbation_flip_rate: float | None = None
    mean_perturbation_js_divergence: float | None = None


class AbstentionReport(BaseModel):
    """Whether the model declined to name a profile, and why."""

    abstained: bool = False
    reasons: list[str] = Field(default_factory=list)
    indeterminate_probability: float = 0.0


class PcosResearchOutput(BaseModel):
    """The complete, self-describing research output for one participant."""

    patient_id: str
    adapter: str = "pcos"
    adapter_version: str = "0.1.0"
    source_dataset: str | None = None
    generated_at: str | None = None

    observed_evidence: ObservedEvidence
    model_organized_phenotype: ModelOrganizedPhenotype
    missing_evidence: MissingEvidence
    uncertainty: UncertaintyReport
    abstention: AbstentionReport

    limitations: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    #: Mandatory. Validated non-empty so it cannot be stripped downstream.
    non_diagnostic_statement: str = NON_DIAGNOSTIC_STATEMENT

    @field_validator("non_diagnostic_statement")
    @classmethod
    def _must_be_present(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError(
                "non_diagnostic_statement is mandatory and must not be blank: an output "
                "that can be read as clinical without it must not be serializable."
            )
        return value

    def write_json(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.model_dump(mode="json"), indent=2) + "\n")
        return path

    @classmethod
    def read_json(cls, path: Path) -> PcosResearchOutput:
        return cls.model_validate(json.loads(Path(path).read_text()))
