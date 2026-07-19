"""Assembly of the static (single-timepoint) feature matrix.

Input is a DataFrame whose columns are already canonical variable codes. This
module only *selects, derives and groups* — it never imputes or scales, because
every fitted transform must live inside a training fold (see ``training/engine``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from features.missingness import build_indicator_columns, indicator_column, missingness_summary
from registry.loader import load_variable_registry

LABEL_CODES: frozenset[str] = frozenset({"pmos_binary"})
ID_COLUMNS: frozenset[str] = frozenset({"patient_id"})

#: Feature groups, keyed by group name -> canonical codes. Groups exist so that
#: ablations ("what if we had no labs?") are one config line, not a code change.
FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "reproductive": (
        "cycle_length",
        "cycle_irregularity",
        "menstrual_frequency_per_year",
        "amenorrhea",
        "infertility_history",
        "pregnancy_history_count",
    ),
    "hormonal_lab": (
        "luteinizing_hormone",
        "follicle_stimulating_hormone",
        "progesterone",
        "estradiol",
        "total_testosterone",
        "free_testosterone",
        "dheas",
        "shbg",
        "anti_mullerian_hormone",
    ),
    "metabolic": (
        "fasting_glucose",
        "fasting_insulin",
        "systolic_blood_pressure",
        "diastolic_blood_pressure",
        "hdl_cholesterol",
        "ldl_cholesterol",
        "triglycerides",
    ),
    "symptom": (
        "hirsutism",
        "ferriman_gallwey_score",
        "acne",
        "androgenic_alopecia",
        "skin_darkening",
        "hair_growth_face",
        "weight_gain",
        "fatigue",
        "mood_change",
        "pelvic_pain",
        "family_history_pmos",
        "family_history_diabetes",
    ),
    "anthropometric": (
        "age",
        "bmi",
        "weight",
        "height",
        "waist_circumference",
        "hip_circumference",
    ),
    "derived_ultrasound": (
        "follicle_number_per_ovary",
        "follicle_count_left",
        "follicle_count_right",
        "ovary_volume_ml",
    ),
}

#: Codes this module can compute from other codes, with their inputs.
DERIVED_SPECS: dict[str, tuple[str, ...]] = {
    "lh_fsh_ratio": ("luteinizing_hormone", "follicle_stimulating_hormone"),
    "waist_hip_ratio": ("waist_circumference", "hip_circumference"),
    "homa_ir": ("fasting_glucose", "fasting_insulin"),
    "bmi": ("weight", "height"),
}

#: Which group each derived code joins once computed.
_DERIVED_GROUP: dict[str, str] = {
    "lh_fsh_ratio": "hormonal_lab",
    "waist_hip_ratio": "anthropometric",
    "homa_ir": "metabolic",
    "bmi": "anthropometric",
}


@dataclass
class StaticFeatureMatrix:
    """The assembled matrix plus everything needed to reproduce it."""

    X: pd.DataFrame
    patient_ids: pd.Series
    y: pd.Series | None = None
    feature_groups: dict[str, list[str]] = field(default_factory=dict)
    derived_columns: list[str] = field(default_factory=list)
    derivation_notes: dict[str, str] = field(default_factory=dict)
    missingness_summary: dict[str, dict[str, int]] = field(default_factory=dict)
    dropped_columns: list[str] = field(default_factory=list)

    @property
    def feature_names(self) -> list[str]:
        return list(self.X.columns)

    def group_of(self, column: str) -> str | None:
        for group, cols in self.feature_groups.items():
            if column in cols:
                return group
        return None


def _clip_to_valid_range(series: pd.Series, code: str) -> pd.Series:
    """Blank out physiologically impossible derived values instead of trusting them."""
    spec = load_variable_registry().variables.get(code)
    if spec is None or spec.valid_range is None:
        return series
    lo, hi = spec.valid_range.min, spec.valid_range.max
    out = series.copy()
    if lo is not None:
        out = out.mask(out < lo)
    if hi is not None:
        out = out.mask(out > hi)
    return out


def derive_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Compute ratio/index features, but only from *observed* inputs.

    A derived value built on an imputed input would launder a guess into what
    looks like a measurement, so every input must be genuinely present.
    """
    out = df.copy()
    notes: dict[str, str] = {}

    def _assign(code: str, values: pd.Series, note: str) -> None:
        values = _clip_to_valid_range(values, code)
        if code in out.columns:
            # Never overwrite a measured value; only fill where it is absent.
            filled = int((out[code].isna() & values.notna()).sum())
            out[code] = out[code].where(out[code].notna(), values)
            notes[code] = f"{note}; filled {filled} previously-missing row(s)"
        else:
            out[code] = values
            notes[code] = note

    have = set(df.columns)

    if have.issuperset(DERIVED_SPECS["lh_fsh_ratio"]):
        lh = pd.to_numeric(df["luteinizing_hormone"], errors="coerce")
        fsh = pd.to_numeric(df["follicle_stimulating_hormone"], errors="coerce")
        _assign(
            "lh_fsh_ratio",
            (lh / fsh.where(fsh > 0)),
            "luteinizing_hormone / follicle_stimulating_hormone (FSH > 0)",
        )

    if have.issuperset(DERIVED_SPECS["waist_hip_ratio"]):
        waist = pd.to_numeric(df["waist_circumference"], errors="coerce")
        hip = pd.to_numeric(df["hip_circumference"], errors="coerce")
        _assign(
            "waist_hip_ratio",
            (waist / hip.where(hip > 0)),
            "waist_circumference / hip_circumference (cm / cm)",
        )

    if have.issuperset(DERIVED_SPECS["homa_ir"]):
        glucose = pd.to_numeric(df["fasting_glucose"], errors="coerce")
        insulin = pd.to_numeric(df["fasting_insulin"], errors="coerce")
        _assign(
            "homa_ir",
            (glucose * insulin) / 405.0,
            "fasting_glucose[mg/dL] * fasting_insulin[uIU/mL] / 405",
        )

    if have.issuperset(DERIVED_SPECS["bmi"]):
        weight = pd.to_numeric(df["weight"], errors="coerce")
        height_m = pd.to_numeric(df["height"], errors="coerce") / 100.0
        _assign("bmi", weight / (height_m.where(height_m > 0) ** 2), "weight[kg] / height[m]^2")

    return out, notes


def build_static_features(
    df: pd.DataFrame,
    *,
    label_column: str | None = "pmos_binary",
    id_column: str = "patient_id",
    include_groups: list[str] | None = None,
    add_missingness_indicators: bool = True,
    per_status_indicators: bool = False,
    min_observed_fraction: float = 0.0,
) -> StaticFeatureMatrix:
    """Assemble the static feature matrix from a canonical-coded DataFrame.

    Values stay as ``NaN`` where unobserved; imputation belongs to the fold-local
    pipeline, never here.
    """
    if id_column not in df.columns:
        raise KeyError(f"Expected an id column '{id_column}' in the input DataFrame.")

    enriched, notes = derive_features(df)

    groups = dict(FEATURE_GROUPS)
    for code, group in _DERIVED_GROUP.items():
        if code not in groups[group]:
            groups[group] = (*groups[group], code)

    selected_groups = list(include_groups) if include_groups else list(groups)
    unknown = set(selected_groups) - set(groups)
    if unknown:
        raise KeyError(f"Unknown feature group(s): {sorted(unknown)}.")

    resolved: dict[str, list[str]] = {}
    dropped: list[str] = []
    for group in selected_groups:
        present = [c for c in groups[group] if c in enriched.columns]
        dropped.extend(c for c in groups[group] if c not in enriched.columns)
        if present:
            resolved[group] = present

    feature_cols = [c for group in resolved.values() for c in group]
    feature_cols = list(dict.fromkeys(feature_cols))
    feature_cols = [c for c in feature_cols if c not in LABEL_CODES and c not in ID_COLUMNS]

    X = enriched[feature_cols].apply(pd.to_numeric, errors="coerce")

    if min_observed_fraction > 0:
        keep = X.notna().mean() >= min_observed_fraction
        dropped.extend(sorted(X.columns[~keep]))
        X = X.loc[:, keep]
        resolved = {g: [c for c in cols if c in X.columns] for g, cols in resolved.items()}
        resolved = {g: cols for g, cols in resolved.items() if cols}
        feature_cols = list(X.columns)

    summary = missingness_summary(enriched, feature_cols)

    if add_missingness_indicators and feature_cols:
        indicators = build_indicator_columns(
            enriched, feature_cols, per_status=per_status_indicators
        )
        # Constant indicators carry no information and only inflate the matrix.
        informative = [c for c in indicators.columns if indicators[c].nunique(dropna=False) > 1]
        indicators = indicators[informative]
        X = pd.concat([X, indicators], axis=1)
        resolved["missingness_indicators"] = list(indicators.columns)

    y: pd.Series | None = None
    if label_column is not None and label_column in df.columns:
        y = pd.to_numeric(df[label_column], errors="coerce")

    derived_present = [c for c in DERIVED_SPECS if c in X.columns]

    return StaticFeatureMatrix(
        X=X,
        patient_ids=df[id_column].astype(str),
        y=y,
        feature_groups=resolved,
        derived_columns=derived_present,
        derivation_notes=notes,
        missingness_summary=summary,
        dropped_columns=sorted(set(dropped)),
    )


def indicator_columns_of(X: pd.DataFrame) -> list[str]:
    """The missingness-indicator columns inside an assembled matrix."""
    return [c for c in X.columns if c.endswith("__is_missing") or "__missing_" in c]


def value_columns_of(X: pd.DataFrame) -> list[str]:
    """The measured/derived value columns (everything that is not an indicator)."""
    indicators = set(indicator_columns_of(X))
    return [c for c in X.columns if c not in indicators]


def observed_value_fraction(X: pd.DataFrame) -> float:
    """Overall fraction of observed cells among value columns — a data-quality read."""
    cols = value_columns_of(X)
    if not cols:
        return 0.0
    return float(np.asarray(X[cols].notna()).mean())


__all__ = [
    "DERIVED_SPECS",
    "FEATURE_GROUPS",
    "StaticFeatureMatrix",
    "build_static_features",
    "derive_features",
    "indicator_column",
    "indicator_columns_of",
    "observed_value_fraction",
    "value_columns_of",
]
