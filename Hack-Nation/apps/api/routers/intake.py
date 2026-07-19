"""Intake form fields, served from the variable registry.

The frontend shouldn't hardcode which variables exist, their units, or their
valid ranges. A hardcoded unit is an undetectable hundredfold error waiting to
happen, and a hardcoded name is how ``cycle_length`` came to mean two different
things here.

Which fields appear is curated; every label, unit and range comes from the
registry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from fastapi import APIRouter

from registry.loader import load_variable_registry

router = APIRouter(prefix="/api/v1/intake", tags=["intake"])


@dataclass(frozen=True)
class _Field:
    """One question, and the help text a patient needs to answer it correctly."""

    code: str
    group: str
    #: Overrides the registry's canonical name when a patient-facing phrasing is
    #: clearer. The registry name stays authoritative for the variable itself.
    prompt: str | None = None
    help_text: str | None = None


#: Fields in the order they're asked.
#:
#: `cycle_length` and `menses_duration` sit adjacent with help text separating
#: them -- the trained model learned a column called "Cycle length(days)" that
#: was actually bleeding duration, and an ambiguous question repeats that at the
#: point of entry.
_FIELDS: tuple[_Field, ...] = (
    _Field("age", "about_you"),
    _Field("weight", "about_you"),
    _Field("height", "about_you"),
    _Field(
        "cycle_length",
        "cycle",
        prompt="Typical cycle length",
        help_text=(
            "Days from the first day of one period to the first day of the next. "
            "Around 28 is typical. This is not the number of days you bleed."
        ),
    ),
    _Field(
        "menses_duration",
        "cycle",
        prompt="Days of bleeding",
        help_text="How many days your bleeding usually lasts. Around 5 is typical.",
    ),
    _Field(
        "cycle_irregularity",
        "cycle",
        prompt="Are your cycles irregular?",
        help_text="Cycles that vary widely in length, or are often late or missed.",
    ),
    _Field(
        "menstrual_frequency_per_year",
        "cycle",
        prompt="Periods per year",
        help_text="Roughly how many periods you had in the last 12 months.",
    ),
    _Field("amenorrhea", "cycle", prompt="Have your periods stopped entirely?"),
    _Field(
        "hirsutism",
        "symptoms",
        prompt="Coarse dark hair growth",
        help_text=(
            "On the face, chest or back. Note if you remove hair cosmetically -- "
            "that can mask this sign."
        ),
    ),
    _Field("acne", "symptoms", prompt="Persistent acne"),
    _Field("androgenic_alopecia", "symptoms", prompt="Thinning scalp hair"),
    _Field(
        "skin_darkening",
        "symptoms",
        prompt="Darkened skin patches",
        help_text="Often at the neck, armpits or groin.",
    ),
    _Field("weight_gain", "symptoms", prompt="Unexplained weight gain"),
    _Field("fatigue", "symptoms"),
    _Field("family_history_pcos", "history", prompt="PCOS in a close relative"),
    _Field("family_history_diabetes", "history", prompt="Type 2 diabetes in a close relative"),
    _Field("waist_circumference", "measurements"),
    _Field("hip_circumference", "measurements"),
    _Field("systolic_blood_pressure", "measurements"),
    _Field("diastolic_blood_pressure", "measurements"),
    _Field("total_testosterone", "labs"),
    _Field("free_testosterone", "labs"),
    _Field("dheas", "labs"),
    _Field("shbg", "labs"),
    _Field("anti_mullerian_hormone", "labs"),
    _Field("luteinizing_hormone", "labs"),
    _Field("follicle_stimulating_hormone", "labs"),
    _Field("fasting_glucose", "labs"),
    _Field("fasting_insulin", "labs"),
    _Field("hdl_cholesterol", "labs"),
    _Field("triglycerides", "labs"),
)

#: Group -> (title, description). Labs are last and explicitly optional.
_GROUPS: dict[str, tuple[str, str]] = {
    "about_you": ("About you", "Basic details used across the assessment."),
    "cycle": ("Your cycle", "The strongest single signal PRISM can assess."),
    "symptoms": (
        "Symptoms you have noticed",
        "Answer only what you are confident about. Leave anything else blank.",
    ),
    "history": ("Family history", "Optional."),
    "measurements": ("Measurements", "Optional. Leave blank if you do not know them."),
    "labs": (
        "Laboratory results",
        "Optional. Enter these only from an actual lab report, using the units shown.",
    ),
}


def _field_type(kind: str) -> Literal["number", "boolean", "text"]:
    if kind == "binary":
        return "boolean"
    if kind in {"continuous", "count", "ordinal"}:
        return "number"
    return "text"


@router.get("/schema")
def intake_schema() -> dict[str, Any]:
    """Field definitions for the intake form.

    Only registry-known codes are served. Collecting a code the encoder will
    discard wastes the patient's effort and looks like it was recorded.
    """
    registry = load_variable_registry().variables

    groups: dict[str, dict[str, Any]] = {}
    for key, (title, description) in _GROUPS.items():
        groups[key] = {"key": key, "title": title, "description": description, "fields": []}

    dropped: list[str] = []
    for field in _FIELDS:
        meta = registry.get(field.code)
        if meta is None:
            dropped.append(field.code)
            continue

        valid_range = getattr(meta, "valid_range", None)
        groups[field.group]["fields"].append(
            {
                "code": field.code,
                "label": field.prompt or meta.canonical_name,
                "canonical_name": meta.canonical_name,
                "type": _field_type(str(meta.type)),
                "unit": getattr(meta, "unit", None),
                "min": getattr(valid_range, "min", None) if valid_range else None,
                "max": getattr(valid_range, "max", None) if valid_range else None,
                "help_text": field.help_text,
                "description": getattr(meta, "description", None),
            }
        )

    return {
        "groups": [g for g in groups.values() if g["fields"]],
        "dropped_unknown_codes": dropped,
        # The form is where a well-meaning UI is most tempted to send 0.
        "guidance": (
            "Leave anything you do not know blank. A blank answer is recorded as "
            "not measured, which is different from a value of zero and is handled "
            "correctly by the models."
        ),
    }
