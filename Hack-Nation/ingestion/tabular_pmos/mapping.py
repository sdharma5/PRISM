"""Source-column mapping for the public PMOS tabular dataset.

Real-world exports of this dataset carry inconsistent header spelling: stray
leading spaces (``" Age (yrs)"``), inconsistent capitalisation, and units baked
into the header text. Lookup is therefore normalized case- and
whitespace-insensitively rather than by exact string match — a header typo must
not silently become an unmapped (and therefore dropped) clinical variable.

Every column in the source file must appear in exactly one of
:data:`SOURCE_COLUMN_MAP` or :data:`EXCLUDED_COLUMNS`. An unaccounted column is
a validation failure, not a shrug.
"""

from __future__ import annotations

import re

from schemas.event import Modality

__all__ = [
    "EXCLUDED_COLUMNS",
    "SOURCE_COLUMN_MAP",
    "UNIT_BY_CODE",
    "canonical_code_for",
    "excluded_reason_for",
    "modality_for",
    "normalize_column",
]

_WS = re.compile(r"\s+")


def normalize_column(column: str) -> str:
    """Normalize a source header for lookup: lowercase, collapsed whitespace."""
    return _WS.sub(" ", str(column).replace(" ", " ")).strip().lower()


#: Source header -> canonical variable code. Keys are written the way they
#: appear in the wild; lookups are normalized, so case/spacing need not match.
_SOURCE_COLUMN_CANDIDATES: dict[str, str | None] = {
    # Demographics / anthropometry
    " Age (yrs)": "age",
    "Age (yrs)": "age",
    "Weight (Kg)": "weight",
    "Height(Cm)": "height",
    "Height (Cm) ": "height",
    "BMI": "bmi",
    "Waist(inch)": "waist_circumference",
    "Hip(inch)": "hip_circumference",
    "Waist:Hip Ratio": "waist_hip_ratio",
    # Menstrual history
    # Named "Cycle length" in the source, but the values centre near 5 days:
    # this is the duration of bleeding, not the menses-to-menses interval. It
    # maps to menses_duration so that cycle_length keeps its registry meaning.
    "Cycle length(days)": "menses_duration",
    "Cycle(R/I)": "cycle_regularity",
    "No. of pregnancies": "pregnancy_history_count",
    # Laboratory
    "FSH(mIU/mL)": "follicle_stimulating_hormone",
    "LH(mIU/mL)": "luteinizing_hormone",
    "FSH/LH": "lh_fsh_ratio",
    "AMH(ng/mL)": "anti_mullerian_hormone",
    "PRG(ng/mL)": "progesterone",
    "RBS(mg/dl)": "fasting_glucose",
    "Vit D3 (ng/mL)": None,  # placeholder replaced below
    "TT(ng/dL)": "total_testosterone",
    "DHEAS(ug/dL)": "dheas",
    "SHBG(nmol/L)": "shbg",
    # Vitals
    "BP _Systolic (mmHg)": "systolic_blood_pressure",
    "BP _Diastolic (mmHg)": "diastolic_blood_pressure",
    "Pulse rate(bpm) ": "resting_heart_rate",
    # Ultrasound report (transcribed numbers, not image measurements)
    "Follicle No. (L)": "follicle_count_left",
    "Follicle No. (R)": "follicle_count_right",
    "Avg. F size (L) (mm)": None,
    "Endometrium (mm)": None,
    # Questionnaire symptoms
    "Weight gain(Y/N)": "weight_gain",
    "hair growth(Y/N)": "hair_growth_face",
    "Skin darkening (Y/N)": "skin_darkening",
    "Hair loss(Y/N)": "androgenic_alopecia",
    "Pimples(Y/N)": "acne",
    "Fast food (Y/N)": None,
    "Reg.Exercise(Y/N)": None,
    # Label
    "PMOS (Y/N)": "pmos_binary",
}

# Drop the placeholder Nones: those headers are excluded, not mapped. Keeping
# them above documents that we considered them rather than missed them.
SOURCE_COLUMN_MAP: dict[str, str] = {
    k: v for k, v in _SOURCE_COLUMN_CANDIDATES.items() if v is not None
}

#: Source header -> why it is deliberately not ingested.
EXCLUDED_COLUMNS: dict[str, str] = {
    "Sl. No": "Row counter; carries no clinical information.",
    "Patient File No.": "Site-local identifier; used as patient_id, not as a variable.",
    "Unnamed: 44": "Empty trailing column produced by the source CSV export.",
    "Vit D3 (ng/mL)": "No canonical variable registered; vitamin D is out of scope for v1.",
    "Avg. F size (L) (mm)": "Mean follicle size is not in the canonical registry yet.",
    "Avg. F size (R) (mm)": "Mean follicle size is not in the canonical registry yet.",
    "Endometrium (mm)": "Endometrial thickness has no canonical variable in v1.",
    "Fast food (Y/N)": "Dietary self-report is unvalidated and not modelled in v1.",
    "Reg.Exercise(Y/N)": "Exercise self-report is unvalidated and not modelled in v1.",
    "Blood Group": "Not relevant to any allowed use of this dataset.",
    "Pregnant(Y/N)": "Current pregnancy status has no canonical variable and confounds hormones.",
    "No. of aborptions": "Pregnancy-loss count differs from pregnancy count; not canonical.",
    "Marraige Status (Yrs)": "Socially confounded proxy; excluded to avoid encoding bias.",
    "II beta-HCG(mIU/mL)": "Pregnancy assay; not a canonical variable and confounds hormones.",
    "I beta-HCG(mIU/mL)": "Pregnancy assay; not a canonical variable and confounds hormones.",
    "TSH (mIU/L)": "Thyroid axis is out of scope for the v1 canonical registry.",
    "PRL(ng/mL)": "Prolactin has no canonical variable in v1.",
    "Hb(g/dl)": "Haemoglobin has no canonical variable in v1.",
    "RR (breaths/min)": "Respiratory rate has no canonical variable in v1.",
}

#: Canonical code -> the unit the source column is expressed in. The source
#: bakes units into the header, so they are declared once here instead of being
#: re-parsed from header text at runtime.
UNIT_BY_CODE: dict[str, str] = {
    "weight": "kg",
    "height": "cm",
    "bmi": "kg/m^2",
    "waist_circumference": "in",
    "hip_circumference": "in",
    "cycle_length": "day",
    "menses_duration": "day",
    "follicle_stimulating_hormone": "mIU/mL",
    "luteinizing_hormone": "mIU/mL",
    "anti_mullerian_hormone": "ng/mL",
    "progesterone": "ng/mL",
    "fasting_glucose": "mg/dL",
    "total_testosterone": "ng/dL",
    "dheas": "ug/dL",
    "shbg": "nmol/L",
    "systolic_blood_pressure": "mmHg",
    "diastolic_blood_pressure": "mmHg",
    "resting_heart_rate": "bpm",
    "age": "year",
}

#: Canonical code -> modality. Modality is a property of *how the value was
#: obtained*, so it is declared per variable rather than per file.
_MODALITY_BY_CODE: dict[str, Modality] = {
    "follicle_stimulating_hormone": "laboratory",
    "luteinizing_hormone": "laboratory",
    "lh_fsh_ratio": "laboratory",
    "anti_mullerian_hormone": "laboratory",
    "progesterone": "laboratory",
    "fasting_glucose": "laboratory",
    "total_testosterone": "laboratory",
    "dheas": "laboratory",
    "shbg": "laboratory",
    "cycle_length": "menstrual_history",
    "menses_duration": "menstrual_history",
    "cycle_regularity": "menstrual_history",
    "cycle_irregularity": "menstrual_history",
    "pregnancy_history_count": "menstrual_history",
    "follicle_count_left": "ultrasound_report",
    "follicle_count_right": "ultrasound_report",
    "ovary_volume_ml": "ultrasound_report",
}

_NORMALIZED_MAP = {normalize_column(k): v for k, v in SOURCE_COLUMN_MAP.items()}
_NORMALIZED_EXCLUDED = {normalize_column(k): v for k, v in EXCLUDED_COLUMNS.items()}


def canonical_code_for(column: str) -> str | None:
    """Return the canonical variable code for a source column, if mapped."""
    return _NORMALIZED_MAP.get(normalize_column(column))


def excluded_reason_for(column: str) -> str | None:
    """Return the documented exclusion reason for a source column, if excluded."""
    return _NORMALIZED_EXCLUDED.get(normalize_column(column))


def modality_for(code: str) -> Modality:
    """Modality for a canonical code; anything unlisted is questionnaire-derived.

    Defaulting to ``questionnaire`` is the conservative choice: it never claims
    a value was laboratory-measured when it was in fact self-reported.
    """
    return _MODALITY_BY_CODE.get(code, "questionnaire")
