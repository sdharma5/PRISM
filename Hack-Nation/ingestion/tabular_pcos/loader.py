"""Ingestion adapter for the public PCOS clinical tabular dataset.

The dataset is cross-sectional and dataset-provided, so every emitted event
carries ``provenance='dataset_provided'`` and
``confirmation_status='not_required'``: there is no patient or clinician
available to confirm anything, and pretending otherwise would let unreviewed
values masquerade as reviewed ones.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ingestion.base import BaseIngestionAdapter, file_checksum
from ingestion.tabular_pcos.cleaning import clean_value
from ingestion.tabular_pcos.mapping import (
    EXCLUDED_COLUMNS,
    SOURCE_COLUMN_MAP,
    UNIT_BY_CODE,
    canonical_code_for,
    modality_for,
)
from ingestion.tabular_pcos.validation import validate_columns, validate_patient_ids
from registry.loader import load_dataset_registry
from schemas.dataset import ProcessingManifest
from schemas.event import EVIDENCE_REQUIRED_MODALITIES, HormonalHealthEvent

DEFAULT_ID_COLUMN = "Patient File No."


class PcosTabularAdapter(BaseIngestionAdapter):
    """Turns one PCOS tabular CSV into canonical :class:`HormonalHealthEvent` rows."""

    dataset_id = "pcos_tabular_public"
    adapter_version = "0.1.0"

    def __init__(
        self,
        *,
        dataset_version: str = "unversioned",
        id_column: str = DEFAULT_ID_COLUMN,
        use: str = "binary_baseline",
    ) -> None:
        super().__init__(dataset_version=dataset_version)
        self.id_column = id_column
        # Fail closed before reading a byte if the registry forbids this use.
        load_dataset_registry().require(self.dataset_id, use)
        self.use = use
        self._source_path: Path | None = None

    # -- Lifecycle ----------------------------------------------------------

    def load_raw(self, source: Any) -> pd.DataFrame:
        """Read the CSV verbatim: everything stays a string until cleaning."""
        path = Path(source)
        self._source_path = path
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        self.file_checksums[path.name] = file_checksum(path)
        self.n_source_records = len(frame)
        return frame

    def validate_raw(self, raw: pd.DataFrame) -> list[str]:
        """Validate the header and the identifier column."""
        errors = validate_columns(list(raw.columns), self.id_column)
        if self.id_column in raw.columns:
            errors.extend(validate_patient_ids(list(raw[self.id_column])))
        self.validation_errors = errors
        return errors

    def transform(self, raw: pd.DataFrame) -> list[HormonalHealthEvent]:
        """Emit one event per mapped, non-empty cell."""
        events: list[HormonalHealthEvent] = []
        column_codes = {c: canonical_code_for(c) for c in raw.columns}

        for index, row in raw.iterrows():
            patient_id = self._scoped_patient_id(row[self.id_column])
            for column, code in column_codes.items():
                if code is None or column == self.id_column:
                    continue
                events.append(self._event_for_cell(patient_id, code, column, row[column], index))

        self.n_events_emitted = len(events)
        return events

    def build_manifest(self) -> ProcessingManifest:
        return self.make_manifest()

    # -- Internals ----------------------------------------------------------

    def _scoped_patient_id(self, raw_id: Any) -> str:
        """Namespace the identifier by dataset so ids can never be cross-merged."""
        return f"{self.dataset_id}:{str(raw_id).strip()}"

    def _event_for_cell(
        self, patient_id: str, code: str, column: str, raw_cell: Any, index: Any
    ) -> HormonalHealthEvent:
        unit = UNIT_BY_CODE.get(code)
        modality = modality_for(code)
        cleaned = clean_value(code, raw_cell, source_unit=unit)
        # Ultrasound-derived columns in this dataset are report-transcribed
        # numbers, so the "evidence" is the source cell itself. Naming the
        # column keeps the number traceable to a location in the file.
        evidence = (
            f"column '{column}' row {index}: {raw_cell!r}"
            if modality in EVIDENCE_REQUIRED_MODALITIES
            else None
        )

        if not cleaned.ok:
            if cleaned.missingness_status != "not_collected":
                self.record_drop(
                    int(index), cleaned.missingness_status, f"{code}: {cleaned.reason}"
                )
            return self.emit_event(
                patient_id=patient_id,
                code=code,
                value=None,
                unit=unit,
                modality=modality,
                provenance="dataset_provided",
                confirmation_status="not_required",
                missingness_status=cleaned.missingness_status,  # type: ignore[arg-type]
                record_index=index,
                raw_value=raw_cell,
                evidence_text=evidence,
                source_file_id=self._source_path.name if self._source_path else None,
            )

        return self.emit_event(
            patient_id=patient_id,
            code=code,
            value=cleaned.value,
            unit=unit,
            modality=modality,
            provenance="dataset_provided",
            confirmation_status="not_required",
            record_index=index,
            # The untouched source cell, not the cleaned value: raw_value must
            # always answer "what did the file literally say?".
            raw_value=raw_cell,
            evidence_text=evidence,
            source_file_id=self._source_path.name if self._source_path else None,
        )

    def variable_mapping(self) -> dict[str, str]:
        return dict(SOURCE_COLUMN_MAP)

    def excluded_source_columns(self) -> dict[str, str]:
        return dict(EXCLUDED_COLUMNS)

    # -- Convenience --------------------------------------------------------

    def run(self, source: Path | str, *, strict: bool = True) -> list[HormonalHealthEvent]:
        """Load, validate and transform in one call.

        Args:
            source: Path to the CSV.
            strict: When True, structural validation errors raise instead of warn.
        """
        raw = self.load_raw(source)
        errors = self.validate_raw(raw)
        if errors and strict:
            raise ValueError("PCOS tabular validation failed:\n" + "\n".join(errors))
        self.warnings.extend(errors)
        return self.transform(raw)
