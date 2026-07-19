"""Shared ingestion contract.

Every adapter in PRISM subclasses :class:`BaseIngestionAdapter`. The base class
owns the parts of ingestion that must never diverge between adapters:

* unit conversion (always via ``registry.loader.convert_to_canonical``),
* valid-range checking (out-of-range values are *recorded*, never clipped),
* checksumming of every source file,
* the processing manifest written next to the outputs.

The reason these live here rather than in each adapter is auditability: a
reviewer should be able to read one file and know how *every* number in the
event store was normalized.
"""

from __future__ import annotations

import csv
import hashlib
import json
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from registry.loader import (
    ConversionResult,
    UnitConversionError,
    convert_to_canonical,
    in_valid_range,
    load_variable_registry,
)
from schemas.dataset import DroppedRecord, ProcessingManifest
from schemas.event import (
    ConfirmationStatus,
    HormonalHealthEvent,
    MissingnessStatus,
    Modality,
    Provenance,
)

CHUNK_BYTES = 1 << 20


def file_checksum(path: Path | str) -> str:
    """Return the sha256 hex digest of a file, streamed in chunks.

    Checksums are the only way a manifest can prove which bytes produced which
    events, so they are computed for every source file including small ones.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while chunk := fh.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def checksum_many(paths: Iterable[Path | str]) -> dict[str, str]:
    """Checksum a collection of files, keyed by file name."""
    return {Path(p).name: file_checksum(p) for p in paths}


class BaseIngestionAdapter(ABC):
    """Abstract base for all dataset adapters.

    Subclasses implement the four lifecycle methods. Everything else — unit
    conversion, range checking, manifest assembly — is inherited so that no
    adapter can quietly invent its own normalization rules.

    Attributes:
        dataset_id: Key into ``registry/datasets.yaml``.
        dataset_version: Version string recorded in the manifest.
        adapter_version: Version of the adapter code itself.
    """

    dataset_id: str = "unspecified"
    dataset_version: str = "unversioned"
    adapter_version: str = "0.1.0"

    def __init__(self, dataset_version: str | None = None) -> None:
        if dataset_version is not None:
            self.dataset_version = dataset_version
        self.dropped_records: list[DroppedRecord] = []
        self.validation_errors: list[str] = []
        self.warnings: list[str] = []
        self.unit_conversions_applied: dict[str, int] = {}
        self.file_checksums: dict[str, str] = {}
        self.n_source_records: int = 0
        self.n_events_emitted: int = 0

    # -- Abstract lifecycle -------------------------------------------------

    @abstractmethod
    def load_raw(self, source: Any) -> Any:
        """Read the source into memory without altering any value."""

    @abstractmethod
    def validate_raw(self, raw: Any) -> list[str]:
        """Return a list of structural problems with the raw input."""

    @abstractmethod
    def transform(self, raw: Any) -> list[HormonalHealthEvent]:
        """Convert validated raw input into canonical events."""

    @abstractmethod
    def build_manifest(self) -> ProcessingManifest:
        """Assemble the processing manifest for the last run."""

    # -- Shared helpers -----------------------------------------------------

    def variable_mapping(self) -> dict[str, str]:
        """Source-column -> canonical-code map. Overridden by adapters."""
        return {}

    def excluded_source_columns(self) -> dict[str, str]:
        """Source-column -> documented reason for exclusion."""
        return {}

    def record_drop(self, index: int | str, reason: str, detail: str = "") -> None:
        """Log a dropped or degraded record instead of silently discarding it."""
        self.dropped_records.append(DroppedRecord(record_index=index, reason=reason, detail=detail))

    def make_manifest(self) -> ProcessingManifest:
        """Build a :class:`ProcessingManifest` from accumulated run state."""
        return ProcessingManifest(
            dataset_id=self.dataset_id,
            dataset_version=self.dataset_version,
            adapter=type(self).__name__,
            adapter_version=self.adapter_version,
            created_at=datetime.now(UTC).isoformat(),
            file_checksums=dict(self.file_checksums),
            variable_mapping=self.variable_mapping(),
            excluded_source_columns=self.excluded_source_columns(),
            n_source_records=self.n_source_records,
            n_events_emitted=self.n_events_emitted,
            n_dropped=len(self.dropped_records),
            unit_conversions_applied=dict(self.unit_conversions_applied),
            validation_errors=list(self.validation_errors),
            warnings=list(self.warnings),
        )

    def emit_event(
        self,
        *,
        patient_id: str,
        code: str,
        value: Any,
        unit: str | None = None,
        modality: Modality,
        provenance: Provenance,
        confirmation_status: ConfirmationStatus = "not_required",
        missingness_status: MissingnessStatus | None = None,
        extraction_confidence: float = 1.0,
        observed_at: datetime | None = None,
        record_index: int | str = "-",
        **extra: Any,
    ) -> HormonalHealthEvent:
        """Build one canonical event, converting units and checking ranges.

        Numeric values are converted into the registry's canonical unit and
        checked against the registry's plausible range. An implausible value is
        never clipped or dropped: the event is emitted as ``not_available`` with
        the original number preserved in ``raw_value`` and the reason logged to
        ``dropped_records``. Clipping would silently manufacture a plausible
        patient who does not exist.

        Args:
            patient_id: Dataset-scoped patient identifier. Never merged across datasets.
            code: Canonical variable code from ``registry/variables.yaml``.
            value: Source value; ``None`` means not observed.
            unit: Source unit string, if any.
            modality: Event modality.
            provenance: How the value came to exist.
            confirmation_status: Review state; defaults to ``not_required``.
            missingness_status: Force a missingness status (skips conversion).
            extraction_confidence: 0-1 confidence in the extraction step.
            observed_at: Observation timestamp, if known.
            record_index: Source row index, used in drop logs.
            **extra: Extra ``HormonalHealthEvent`` fields (provenance pointers etc).

        Returns:
            A validated :class:`HormonalHealthEvent`.
        """
        variable_name = self._variable_name(code)
        base: dict[str, Any] = {
            "patient_id": patient_id,
            "variable_name": variable_name,
            "canonical_variable_code": code,
            "raw_value": value,
            "raw_unit": unit,
            "modality": modality,
            "provenance": provenance,
            "confirmation_status": confirmation_status,
            "extraction_confidence": extraction_confidence,
            "observed_at": observed_at,
            "source_dataset": self.dataset_id,
            "parser_version": f"{type(self).__name__}/{self.adapter_version}",
            **extra,
        }

        if missingness_status is not None and missingness_status != "observed":
            return HormonalHealthEvent(
                value=None, unit=None, missingness_status=missingness_status, **base
            )

        if value is None:
            return HormonalHealthEvent(
                value=None, unit=None, missingness_status="not_collected", **base
            )

        if isinstance(value, bool) or not isinstance(value, (int, float)):
            # Non-numeric values (bools, categories, free text) have no unit
            # conversion; the canonical unit is whatever the registry declares.
            return HormonalHealthEvent(
                value=value,
                unit=unit or self._canonical_unit(code),
                missingness_status="observed",
                **base,
            )

        try:
            conversion = convert_to_canonical(code, float(value), unit)
        except UnitConversionError as exc:
            self.record_drop(record_index, "unit_conversion_failed", f"{code}: {exc}")
            return HormonalHealthEvent(
                value=None, unit=None, missingness_status="not_available", **base
            )

        self._count_conversion(code, conversion)

        if not in_valid_range(code, conversion.value):
            # Out-of-range is recorded, not repaired. See module docstring.
            self.record_drop(
                record_index,
                "value_out_of_valid_range",
                f"{code}={conversion.value} {conversion.canonical_unit} "
                "outside registry valid_range",
            )
            return HormonalHealthEvent(
                value=None, unit=None, missingness_status="not_available", **base
            )

        return HormonalHealthEvent(
            value=conversion.value,
            unit=conversion.canonical_unit,
            missingness_status="observed",
            **base,
        )

    def _count_conversion(self, code: str, conversion: ConversionResult) -> None:
        if conversion.conversion_applied:
            key = f"{code}:{conversion.source_unit}->{conversion.canonical_unit}"
            self.unit_conversions_applied[key] = self.unit_conversions_applied.get(key, 0) + 1

    @staticmethod
    def _variable_name(code: str) -> str:
        spec = load_variable_registry().variables.get(code)
        return spec.canonical_name if spec else code

    @staticmethod
    def _canonical_unit(code: str) -> str | None:
        spec = load_variable_registry().variables.get(code)
        if spec is None:
            return None
        return spec.canonical_unit or spec.unit

    # -- Manifest artifacts -------------------------------------------------

    def write_manifest_artifacts(
        self,
        directory: Path | str,
        *,
        raw_manifest: dict[str, Any] | None = None,
        processing_config: dict[str, Any] | None = None,
        events: Sequence[HormonalHealthEvent] | None = None,
    ) -> dict[str, Path]:
        """Write the full audit bundle for one ingestion run.

        Files written:
            ``raw_manifest.json``, ``file_checksums.json``, ``variable_mapping.json``,
            ``validation_report.json``, ``dropped_records.csv``,
            ``processing_config.yaml``, ``processed_manifest.json``.

        Args:
            directory: Destination directory; created if absent.
            raw_manifest: Description of the raw inputs (paths, row counts).
            processing_config: The effective configuration for this run.
            events: Emitted events, used only for summary counts.

        Returns:
            Mapping of artifact name to written path.
        """
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        manifest = self.build_manifest()
        written: dict[str, Path] = {}

        written["raw_manifest.json"] = self._write_json(
            out / "raw_manifest.json",
            raw_manifest
            or {
                "dataset_id": self.dataset_id,
                "dataset_version": self.dataset_version,
                "files": sorted(self.file_checksums),
                "n_source_records": self.n_source_records,
            },
        )
        written["file_checksums.json"] = self._write_json(
            out / "file_checksums.json", self.file_checksums
        )
        written["variable_mapping.json"] = self._write_json(
            out / "variable_mapping.json",
            {
                "mapping": manifest.variable_mapping,
                "excluded_source_columns": manifest.excluded_source_columns,
            },
        )
        written["validation_report.json"] = self._write_json(
            out / "validation_report.json",
            {
                "validation_errors": manifest.validation_errors,
                "warnings": manifest.warnings,
                "n_dropped": manifest.n_dropped,
                "n_events_emitted": manifest.n_events_emitted if events is None else len(events),
                "unit_conversions_applied": manifest.unit_conversions_applied,
            },
        )
        written["dropped_records.csv"] = self._write_dropped_csv(out / "dropped_records.csv")
        cfg_path = out / "processing_config.yaml"
        cfg_path.write_text(
            yaml.safe_dump(
                processing_config
                or {
                    "adapter": manifest.adapter,
                    "adapter_version": manifest.adapter_version,
                    "dataset_id": self.dataset_id,
                    "dataset_version": self.dataset_version,
                },
                sort_keys=True,
            )
        )
        written["processing_config.yaml"] = cfg_path
        written["processed_manifest.json"] = self._write_json(
            out / "processed_manifest.json", manifest.model_dump(mode="json")
        )
        return written

    @staticmethod
    def _write_json(path: Path, payload: Any) -> Path:
        path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
        return path

    def _write_dropped_csv(self, path: Path) -> Path:
        with path.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["record_index", "reason", "detail"])
            for rec in self.dropped_records:
                writer.writerow([rec.record_index, rec.reason, rec.detail])
        return path
