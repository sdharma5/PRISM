"""Longitudinal participant-day records and current-state model outputs.

The temporal model estimates *state* (where a person is now), never *trait*
(what condition or subtype they have). See ``docs/concepts/trait_vs_state.md``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

CyclePhase = Literal["menstrual", "follicular", "peri_ovulatory", "luteal", "unknown"]

#: Every time-varying feature is carried as this triple, never as a bare float.
TEMPORAL_FEATURE_SUFFIXES = ("value", "is_observed", "time_since_last_observed")


class ParticipantDay(BaseModel):
    """One row per participant per day."""

    participant_id: str
    study_day: int
    calendar_date: str | None = None
    cycle_day: int | None = None
    cycle_phase: CyclePhase = "unknown"

    values: dict[str, float | None] = Field(default_factory=dict)
    is_observed: dict[str, bool] = Field(default_factory=dict)
    time_since_last_observed: dict[str, float] = Field(default_factory=dict)

    daily_symptoms: dict[str, bool] = Field(default_factory=dict)
    source_dataset: str | None = None

    def observed_fraction(self) -> float:
        if not self.is_observed:
            return 0.0
        return sum(self.is_observed.values()) / len(self.is_observed)


class TemporalStateOutput(BaseModel):
    """Current hormonal-state estimate for one participant on one date."""

    patient_id: str
    as_of_date: str

    state_embedding: list[float] = Field(default_factory=list)

    hormone_predictions: dict[str, float] = Field(default_factory=dict)
    cycle_phase_probabilities: dict[str, float] = Field(default_factory=dict)
    symptom_probabilities: dict[str, float] = Field(default_factory=dict)

    uncertainty: dict[str, float] = Field(default_factory=dict)
    input_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    lookback_days: int = 0
    model_version: str = "0.1.0"
    warnings: list[str] = Field(default_factory=list)

    #: Fixed disclaimer carried with every export so downstream consumers cannot
    #: reinterpret a state estimate as a diagnosis or subtype.
    interpretation: str = (
        "Current hormonal-state estimate. Not a subtype, diagnosis, or clinical decision."
    )

    def predicted_phase(self) -> CyclePhase:
        if not self.cycle_phase_probabilities:
            return "unknown"
        return max(self.cycle_phase_probabilities.items(), key=lambda kv: kv[1])[0]  # type: ignore[return-value]
