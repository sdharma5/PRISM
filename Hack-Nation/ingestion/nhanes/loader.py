"""NHANES ingestion adapter.

NHANES supports population reference ranges, unit harmonization and external
stress testing — and nothing else. The adapter refuses any use the dataset
registry does not list, before it opens a single file, so a prohibited claim can
never be produced by accident downstream.

Survey weights travel with every event as metadata rather than as clinical
variables: they describe the sample, not the person.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ingestion.base import BaseIngestionAdapter, file_checksum
from ingestion.nhanes.merge import (
    SEQN,
    collect_survey_metadata,
    merge_components,
)
from ingestion.nhanes.validation import (
    absent_mapped_columns,
    assert_use_permitted,
    validate_merged_frame,
    validate_weight_availability,
)
from schemas.dataset import ProcessingManifest
from schemas.event import HormonalHealthEvent, Modality

#: NHANES variable -> canonical code. NHANES names are opaque codes, so the unit
#: each one is reported in is declared alongside it rather than guessed.
NHANES_COLUMN_MAP: dict[str, str] = {
    "RIDAGEYR": "age",
    "BMXBMI": "bmi",
    "BMXWT": "weight",
    "BMXHT": "height",
    "BMXWAIST": "waist_circumference",
    "BMXHIP": "hip_circumference",
    "LBXTST": "total_testosterone",
    "LBXSHBG": "shbg",
    "LBXEST": "estradiol",
    "LBXGLU": "fasting_glucose",
    "LBXIN": "fasting_insulin",
    "LBDHDD": "hdl_cholesterol",
    "LBDLDL": "ldl_cholesterol",
    "LBXTR": "triglycerides",
    "BPXOSY1": "systolic_blood_pressure",
    "BPXODI1": "diastolic_blood_pressure",
}

#: Canonical code -> the unit NHANES reports it in.
NHANES_UNIT_BY_CODE: dict[str, str] = {
    "age": "year",
    "bmi": "kg/m^2",
    "weight": "kg",
    "height": "cm",
    "waist_circumference": "cm",
    "hip_circumference": "cm",
    "total_testosterone": "ng/dL",
    "shbg": "nmol/L",
    "estradiol": "pg/mL",
    "fasting_glucose": "mg/dL",
    "fasting_insulin": "uIU/mL",
    "hdl_cholesterol": "mg/dL",
    "ldl_cholesterol": "mg/dL",
    "triglycerides": "mg/dL",
    "systolic_blood_pressure": "mmHg",
    "diastolic_blood_pressure": "mmHg",
}

_LABORATORY_CODES = frozenset(
    {
        "total_testosterone",
        "shbg",
        "estradiol",
        "fasting_glucose",
        "fasting_insulin",
        "hdl_cholesterol",
        "ldl_cholesterol",
        "triglycerides",
    }
)


class NhanesAdapter(BaseIngestionAdapter):
    """Merges NHANES component tables and emits canonical events."""

    dataset_id = "nhanes_2021_2023"
    adapter_version = "0.1.0"

    def __init__(
        self,
        *,
        dataset_version: str = "2021-2023",
        use: str = "population_reference",
        weight_columns: tuple[str, ...] = ("WTMEC2YR",),
    ) -> None:
        super().__init__(dataset_version=dataset_version)
        assert_use_permitted(self.dataset_id, use)
        self.use = use
        self.weight_columns = weight_columns

    # -- Lifecycle ----------------------------------------------------------

    def load_raw(self, source: Any) -> pd.DataFrame:
        """Load and merge component tables on SEQN.

        Args:
            source: Mapping of component name -> CSV path or DataFrame.
        """
        if isinstance(source, pd.DataFrame):
            merged = source.copy()
        else:
            components: dict[str, pd.DataFrame] = {}
            for name, item in dict(source).items():
                if isinstance(item, pd.DataFrame):
                    components[name] = item
                    continue
                path = Path(item)
                self.file_checksums[path.name] = file_checksum(path)
                components[name] = pd.read_csv(path)
            merged = merge_components(components)
        self.n_source_records = len(merged)
        return merged

    def validate_raw(self, raw: pd.DataFrame) -> list[str]:
        """Validate the merged frame and note any missing survey weights."""
        errors = validate_merged_frame(raw, NHANES_COLUMN_MAP)
        self.validation_errors = errors
        self.warnings.extend(absent_mapped_columns(raw, NHANES_COLUMN_MAP))
        self.warnings.extend(validate_weight_availability(raw, self.weight_columns))
        return errors

    def transform(self, raw: pd.DataFrame) -> list[HormonalHealthEvent]:
        """Emit one event per mapped column per respondent."""
        events: list[HormonalHealthEvent] = []
        present = {c: code for c, code in NHANES_COLUMN_MAP.items() if c in raw.columns}

        for index, row in raw.iterrows():
            patient_id = self._scoped_patient_id(row[SEQN])
            survey_metadata = collect_survey_metadata(row)
            evidence = "; ".join(f"{k}={v}" for k, v in sorted(survey_metadata.items()))
            for column, code in present.items():
                value = row[column]
                usable = pd.notna(value)
                events.append(
                    self.emit_event(
                        patient_id=patient_id,
                        code=code,
                        value=float(value) if usable else None,
                        unit=NHANES_UNIT_BY_CODE.get(code),
                        modality=self._modality_for(code),
                        provenance="dataset_provided",
                        confirmation_status="not_required",
                        missingness_status=None if usable else "not_collected",
                        record_index=index,
                        source_file_id=column,
                        # Survey weights ride along as evidence text so a
                        # downstream weighted estimate never has to re-open
                        # the source files to find them.
                        evidence_text=evidence or None,
                        raw_value=None if not usable else float(value),
                    )
                )
        self.n_events_emitted = len(events)
        return events

    def build_manifest(self) -> ProcessingManifest:
        return self.make_manifest()

    # -- Internals ----------------------------------------------------------

    def _scoped_patient_id(self, seqn: Any) -> str:
        """SEQN is only unique within an NHANES cycle, so scope it by dataset."""
        raw = int(seqn) if pd.notna(seqn) else "unknown"
        return f"{self.dataset_id}:{raw}"

    @staticmethod
    def _modality_for(code: str) -> Modality:
        return "laboratory" if code in _LABORATORY_CODES else "questionnaire"

    def variable_mapping(self) -> dict[str, str]:
        return dict(NHANES_COLUMN_MAP)

    def excluded_source_columns(self) -> dict[str, str]:
        return {
            "WTINT2YR": "Survey design weight; carried as event metadata, not a variable.",
            "WTMEC2YR": "Survey design weight; carried as event metadata, not a variable.",
            "WTSAF2YR": "Fasting subsample weight; carried as event metadata.",
            "SDMVPSU": "Design primary sampling unit; sample structure, not a person-level value.",
            "SDMVSTRA": "Design stratum; sample structure, not a person-level value.",
            "RIDSTATR": "Interview/exam status flag; used for eligibility, not modelled.",
        }

    def run(self, source: Any, *, strict: bool = True) -> list[HormonalHealthEvent]:
        """Load, validate and transform in one call."""
        raw = self.load_raw(source)
        errors = self.validate_raw(raw)
        if errors and strict:
            raise ValueError("NHANES validation failed:\n" + "\n".join(errors))
        self.warnings.extend(errors)
        return self.transform(raw)
