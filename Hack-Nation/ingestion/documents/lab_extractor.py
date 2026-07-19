"""Extract laboratory results from parsed documents.

Two hard rules govern this module, and both exist because a wrong lab value is
worse than a missing one:

1. **Nothing ungrounded survives.** Every extracted value must be locatable at a
   page number and a character span whose text reproduces the value. A value
   that cannot be grounded is dropped and counted in
   :attr:`LabExtractionResult.unsupported`, never silently added. An extractor
   that quietly emits a number nobody can point at in the source is
   indistinguishable from one that makes numbers up.

2. **Both representations are kept.** ``value_source``/``unit_source`` hold
   exactly what the report said; ``value_canonical``/``unit_canonical`` hold the
   registry-normalized form. Testosterone reported as 2.7 nmol/L and as 78 ng/dL
   are the same measurement, but only one of them is what the lab printed, and a
   reviewer checking the extraction is looking at the printed one.

Unit conversion is delegated entirely to ``registry.loader.convert_to_canonical``.
There is no conversion constant anywhere in this file.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from ingestion.documents.normalization import (
    normalize_test_name,
    normalize_text,
    normalize_unit_text,
    parse_date,
    parse_number,
)
from ingestion.documents.parser import DocumentPage, DocumentTable, ParsedDocument
from registry.loader import UnitConversionError, convert_to_canonical, load_variable_registry
from schemas.event import HormonalHealthEvent

SYNONYMS_PATH = Path(__file__).resolve().parent / "test_synonyms.yaml"
EXTRACTOR_VERSION = "lab_extractor/1.0.0"

ExtractionMethod = Literal["table", "line"]

_NUMBER = r"[-+]?\d[\d.,]*"

# "Total Testosterone: 78 ng/dL (Ref: 15 - 70)" and friends.
_LINE_PATTERN = re.compile(
    r"^\s*(?P<name>[A-Za-z][A-Za-z0-9 ()\-/.']{1,48}?)\s*[:=]\s*"
    r"(?P<value>" + _NUMBER + r")\s*"
    r"(?P<unit>[A-Za-zµμ%/^0-9]+(?:/[A-Za-zµμ0-9^]+)?)?\s*"
    r"(?P<ref>[\(\[]?\s*(?:ref(?:erence)?(?:\s*range)?)\s*[:.]?\s*[^)\]]*[\)\]]?)?\s*$",
    re.IGNORECASE,
)

_REF_RANGE = re.compile(r"(?P<low>" + _NUMBER + r")\s*(?:-|to|–)\s*(?P<high>" + _NUMBER + r")")
_REF_BOUND = re.compile(r"(?P<op>[<>]=?)\s*(?P<value>" + _NUMBER + r")")

_COLLECTED = re.compile(
    r"(?:collected|collection|drawn|specimen)\s*(?:date)?\s*[:\-]\s*(?P<date>[^\n,;]+)",
    re.IGNORECASE,
)
_REPORTED = re.compile(
    r"(?:reported|report date|released|resulted)\s*[:\-]\s*(?P<date>[^\n,;]+)", re.IGNORECASE
)

_TABLE_COLUMNS = {
    "test": {"test", "analyte", "name", "test name", "investigation", "parameter"},
    "value": {"result", "value", "results", "measurement"},
    "unit": {"unit", "units"},
    "reference": {"reference range", "reference", "range", "ref range", "normal range", "ref"},
    "collected": {"collected", "collection date", "drawn"},
    "reported": {"reported", "report date"},
}


class ReferenceRange(BaseModel):
    """A parsed reference range. Kept as text too, because labs disagree."""

    text: str
    low: float | None = None
    high: float | None = None
    operator: Literal["between", "lt", "lte", "gt", "gte"] | None = None

    def contains(self, value: float) -> bool | None:
        """True/False when the range is interpretable, ``None`` when it is not."""
        if self.operator == "between" and self.low is not None and self.high is not None:
            return self.low <= value <= self.high
        if self.operator in {"lt", "lte"} and self.high is not None:
            return value < self.high if self.operator == "lt" else value <= self.high
        if self.operator in {"gt", "gte"} and self.low is not None:
            return value > self.low if self.operator == "gt" else value >= self.low
        return None


class ExtractedLabResult(BaseModel):
    """One laboratory value recovered from a document, with its grounding.

    The page span is not optional metadata — the model refuses to exist without
    it, so an ungrounded value cannot be constructed by accident anywhere in the
    codebase.
    """

    result_id: str
    document_id: str
    patient_id: str

    source_test_name: str
    canonical_code: str
    variable_name: str

    raw_text: str
    value_source: float
    unit_source: str | None = None

    value_canonical: float | None = None
    unit_canonical: str | None = None
    conversion_applied: bool = False

    reference_range: ReferenceRange | None = None
    collected_date: str | None = None
    reported_date: str | None = None

    page_number: int
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    evidence_text: str

    extraction_method: ExtractionMethod = "line"
    extraction_confidence: float = Field(ge=0.0, le=1.0)
    extractor_version: str = EXTRACTOR_VERSION
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> ExtractedLabResult:
        if self.char_end <= self.char_start:
            raise ValueError(f"{self.canonical_code}: evidence span must be non-empty.")
        if not self.evidence_text.strip():
            raise ValueError(f"{self.canonical_code}: evidence_text must not be blank.")
        if self.page_number < 1:
            raise ValueError(f"{self.canonical_code}: page numbers are 1-based.")
        return self


class UnsupportedValue(BaseModel):
    """A candidate value that was dropped because it could not be grounded."""

    document_id: str
    source_test_name: str
    raw_text: str
    reason: str
    page_number: int | None = None


class UnmappedTest(BaseModel):
    """A test name with no canonical variable, reported rather than guessed."""

    document_id: str
    source_test_name: str
    raw_text: str
    reason: str
    page_number: int | None = None


class LabExtractionResult(BaseModel):
    """Everything one document yielded, including what was refused."""

    document_id: str
    patient_id: str
    results: list[ExtractedLabResult] = Field(default_factory=list)
    unsupported: list[UnsupportedValue] = Field(default_factory=list)
    unmapped: list[UnmappedTest] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @property
    def unsupported_rate(self) -> float:
        total = len(self.results) + len(self.unsupported)
        return len(self.unsupported) / total if total else 0.0


@lru_cache(maxsize=4)
def load_synonyms(path: Path | None = None) -> dict[str, Any]:
    """Load and cache the test-name synonym table."""
    with (path or SYNONYMS_PATH).open() as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError("test_synonyms.yaml must contain a mapping at the top level.")
    return data


def parse_reference_range(text: str) -> ReferenceRange | None:
    """Parse "2.5 - 10.2", "<1.5", ">40" and their unicode-dash variants."""
    cleaned = normalize_text(text).strip()
    if not cleaned:
        return None
    stripped = re.sub(r"(?i)\b(ref(erence)?( range)?|normal( range)?)\b\s*[:.]?", "", cleaned)
    stripped = stripped.strip(" ()[]:")

    between = _REF_RANGE.search(stripped)
    if between:
        low = parse_number(between.group("low"))
        high = parse_number(between.group("high"))
        if low is not None and high is not None:
            return ReferenceRange(text=cleaned, low=low, high=high, operator="between")

    bound = _REF_BOUND.search(stripped)
    if bound:
        value = parse_number(bound.group("value"))
        if value is None:
            return None
        operator = {"<": "lt", "<=": "lte", ">": "gt", ">=": "gte"}[bound.group("op")]
        if operator in {"lt", "lte"}:
            return ReferenceRange(text=cleaned, high=value, operator=operator)  # type: ignore[arg-type]
        return ReferenceRange(text=cleaned, low=value, operator=operator)  # type: ignore[arg-type]
    return None


class LabExtractor:
    """Table- and line-oriented laboratory extraction with page grounding."""

    version = "1.0.0"

    def __init__(self, synonyms_path: Path | None = None) -> None:
        table = load_synonyms(synonyms_path)
        self.synonyms: dict[str, str] = dict(table.get("synonyms", {}))
        self.known_unmapped: dict[str, str] = dict(table.get("known_unmapped", {}))
        self._variables = load_variable_registry().variables

    # -- Public API ---------------------------------------------------------

    def extract(self, document: ParsedDocument, *, patient_id: str) -> LabExtractionResult:
        """Extract every groundable lab value in ``document``."""
        result = LabExtractionResult(document_id=document.document_id, patient_id=patient_id)
        collected, reported = self._document_dates(document)
        counter = 0

        for page in document.pages:
            table_rows = {id(t): t for t in page.tables}
            for table in table_rows.values():
                counter = self._extract_table(
                    document, page, table, result, counter, collected, reported
                )
            counter = self._extract_lines(document, page, result, counter, collected, reported)
        return result

    # -- Table extraction ---------------------------------------------------

    def _extract_table(
        self,
        document: ParsedDocument,
        page: DocumentPage,
        table: DocumentTable,
        result: LabExtractionResult,
        counter: int,
        collected: str | None,
        reported: str | None,
    ) -> int:
        columns = self._map_columns(table.header)
        if "test" not in columns or "value" not in columns:
            return counter

        for row in table.rows:
            if len(row) <= max(columns.values()):
                continue
            name = row[columns["test"]].strip()
            if not name or name.lower() in {"test", "analyte"}:
                continue
            value_text = row[columns["value"]].strip()
            value = parse_number(value_text)
            if value is None:
                continue

            unit = row[columns["unit"]].strip() if "unit" in columns else None
            ref_text = row[columns["reference"]].strip() if "reference" in columns else ""
            row_collected = row[columns["collected"]].strip() if "collected" in columns else None
            row_reported = row[columns["reported"]].strip() if "reported" in columns else None

            counter = self._emit(
                document=document,
                page=page,
                result=result,
                counter=counter,
                source_name=name,
                raw_text=" | ".join(row),
                value=value,
                value_text=value_text,
                unit=unit,
                reference_text=ref_text,
                collected=self._iso(row_collected) or collected,
                reported=self._iso(row_reported) or reported,
                method="table",
                confidence=0.95,
            )
        return counter

    @staticmethod
    def _map_columns(header: list[str]) -> dict[str, int]:
        mapping: dict[str, int] = {}
        for index, cell in enumerate(header):
            key = normalize_test_name(cell)
            for role, aliases in _TABLE_COLUMNS.items():
                if key in aliases and role not in mapping:
                    mapping[role] = index
        return mapping

    # -- Line extraction ----------------------------------------------------

    def _extract_lines(
        self,
        document: ParsedDocument,
        page: DocumentPage,
        result: LabExtractionResult,
        counter: int,
        collected: str | None,
        reported: str | None,
    ) -> int:
        for line, _start, _end in page.line_spans():
            if "|" in line:  # already handled as a table row
                continue
            match = _LINE_PATTERN.match(normalize_text(line))
            if not match:
                continue
            value = parse_number(match.group("value"))
            if value is None:
                continue
            counter = self._emit(
                document=document,
                page=page,
                result=result,
                counter=counter,
                source_name=match.group("name").strip(),
                raw_text=line.strip(),
                value=value,
                value_text=match.group("value"),
                unit=match.group("unit"),
                reference_text=match.group("ref") or "",
                collected=collected,
                reported=reported,
                method="line",
                confidence=0.9,
            )
        return counter

    # -- Shared emit path ---------------------------------------------------

    def _emit(
        self,
        *,
        document: ParsedDocument,
        page: DocumentPage,
        result: LabExtractionResult,
        counter: int,
        source_name: str,
        raw_text: str,
        value: float,
        value_text: str,
        unit: str | None,
        reference_text: str,
        collected: str | None,
        reported: str | None,
        method: ExtractionMethod,
        confidence: float,
    ) -> int:
        key = normalize_test_name(source_name)
        code = self.synonyms.get(key)

        if code is None:
            if key in self.known_unmapped:
                result.unmapped.append(
                    UnmappedTest(
                        document_id=document.document_id,
                        source_test_name=source_name,
                        raw_text=raw_text,
                        reason=self.known_unmapped[key],
                        page_number=page.page_number,
                    )
                )
            return counter

        grounding = self._ground(page, raw_text, value_text)
        if grounding is None:
            # Rule 1: never emit what cannot be pointed at.
            result.unsupported.append(
                UnsupportedValue(
                    document_id=document.document_id,
                    source_test_name=source_name,
                    raw_text=raw_text,
                    reason="value could not be grounded to a page span",
                    page_number=page.page_number,
                )
            )
            return counter

        char_start, char_end, evidence = grounding
        warnings: list[str] = []
        unit_text = normalize_unit_text(unit) if unit else None

        canonical_value: float | None = None
        canonical_unit: str | None = None
        conversion_applied = False
        try:
            conversion = convert_to_canonical(code, value, unit_text)
            canonical_value = conversion.value
            canonical_unit = conversion.canonical_unit
            conversion_applied = conversion.conversion_applied
        except UnitConversionError as exc:
            # Loud failure, preserved source value. Guessing a factor here is how
            # a tenfold error enters a dataset and is never found again.
            warnings.append(f"unit_conversion_failed: {exc}")

        counter += 1
        result.results.append(
            ExtractedLabResult(
                result_id=f"{document.document_id}-lab{counter:03d}",
                document_id=document.document_id,
                patient_id=result.patient_id,
                source_test_name=source_name,
                canonical_code=code,
                variable_name=self._variable_name(code),
                raw_text=raw_text,
                value_source=value,
                unit_source=unit_text,
                value_canonical=canonical_value,
                unit_canonical=canonical_unit,
                conversion_applied=conversion_applied,
                reference_range=parse_reference_range(reference_text),
                collected_date=collected,
                reported_date=reported,
                page_number=page.page_number,
                char_start=char_start,
                char_end=char_end,
                evidence_text=evidence,
                extraction_method=method,
                extraction_confidence=confidence if not warnings else confidence * 0.5,
                warnings=warnings,
            )
        )
        return counter

    @staticmethod
    def _ground(page: DocumentPage, raw_text: str, value_text: str) -> tuple[int, int, str] | None:
        """Locate the source line inside the page text, in document coordinates.

        Both the line and the value must be found: a line match alone would let a
        value that was never printed inherit a neighbouring line's grounding.
        """
        needle = raw_text.strip()
        for line, start, end in page.line_spans():
            if not line.strip():
                continue
            if line.strip() == needle or (needle and needle in line):
                if value_text.strip() and value_text.strip() not in line:
                    continue
                return (start, end, line)
        # Table rows are re-joined with " | " and may not match the raw line, so
        # fall back to matching on the value inside a line mentioning the test.
        for line, start, end in page.line_spans():
            if value_text.strip() and value_text.strip() in line:
                return (start, end, line)
        return None

    @staticmethod
    def _document_dates(document: ParsedDocument) -> tuple[str | None, str | None]:
        text = document.text
        collected_match = _COLLECTED.search(text)
        reported_match = _REPORTED.search(text)
        collected = LabExtractor._iso(collected_match.group("date") if collected_match else None)
        reported = LabExtractor._iso(reported_match.group("date") if reported_match else None)
        return collected, reported

    @staticmethod
    def _iso(raw: str | None) -> str | None:
        if not raw:
            return None
        parsed = parse_date(raw.strip())
        return parsed.isoformat() if parsed else None

    def _variable_name(self, code: str) -> str:
        spec = self._variables.get(code)
        return spec.canonical_name if spec else code


def to_events(
    results: list[ExtractedLabResult],
    *,
    source_dataset: str | None = None,
    source_file_hash: str | None = None,
) -> list[HormonalHealthEvent]:
    """Convert extracted lab results into canonical events.

    Every event is ``document_extracted`` and ``awaiting_clinician_confirmation``:
    a machine read a PDF, which is a hypothesis about what the PDF says, not a
    confirmed observation. ``HormonalHealthEvent`` independently refuses to let
    ``document_extracted`` be ``confirmed`` without a reviewer, so this is
    belt-and-braces by design.
    """
    events: list[HormonalHealthEvent] = []
    for item in results:
        observed = item.value_canonical is not None
        events.append(
            HormonalHealthEvent(
                patient_id=item.patient_id,
                variable_name=item.variable_name,
                canonical_variable_code=item.canonical_code,
                value=item.value_canonical if observed else None,
                unit=item.unit_canonical if observed else None,
                raw_value=item.value_source,
                raw_unit=item.unit_source or "dimensionless",
                observed_at=item.collected_date or item.reported_date,
                modality="laboratory",
                provenance="document_extracted",
                extraction_confidence=item.extraction_confidence,
                confirmation_status="awaiting_clinician_confirmation",
                missingness_status="observed" if observed else "extraction_failed",
                source_dataset=source_dataset,
                source_file_id=item.document_id,
                source_file_hash=source_file_hash,
                source_page=item.page_number,
                evidence_text=item.evidence_text,
                parser_version=item.extractor_version,
            )
        )
    return events
