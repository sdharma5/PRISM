"""Ultrasound-report and clinical-summary extraction.

Why ultrasound reports are handled separately from lab panels: a lab panel is a
name/value/unit grid, whereas an ultrasound report is prose with numbers
embedded in clinical phrasing ("right ovary measures 12.4 mL and contains 19
follicles"). The same line-oriented lab rules would either miss those numbers or
attach them to the wrong side.

Why impressions are extracted as text, not as a diagnosis: "appearances are
consistent with polycystic ovarian morphology" is a radiologist's impression
about an image, and PRISM records it as ``ovarian_morphology_evidence`` evidence
awaiting clinician confirmation. Turning it into a PMOS diagnosis would be the
model asserting something no human in the loop asserted.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

from ingestion.documents.normalization import normalize_text, parse_number
from ingestion.documents.parser import DocumentPage, ParsedDocument
from registry.loader import load_variable_registry
from schemas.event import HormonalHealthEvent

REPORT_EXTRACTOR_VERSION = "report_extractor/1.0.0"

Side = Literal["left", "right", "unspecified"]

_NUMBER = r"\d+(?:[.,]\d+)?"

_FOLLICLE_PATTERNS = (
    re.compile(
        r"(?P<side>left|right)\s+ovary[^.\n]{0,80}?(?P<count>\d{1,3})\s*"
        r"(?:antral\s+)?follicles?",
        re.IGNORECASE,
    ),
    re.compile(
        r"follicle\s+count[,:]?\s*(?P<side>left|right)\s*[:=]?\s*(?P<count>\d{1,3})",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?P<side>left|right)\s*(?:ovary)?\s*follicle(?:\s+count|s)?\s*[:=]\s*(?P<count>\d{1,3})",
        re.IGNORECASE,
    ),
)

_VOLUME_PATTERNS = (
    re.compile(
        r"(?P<side>left|right)\s+ovary[^.\n]{0,80}?volume\s*[:=]?\s*(?P<value>" + _NUMBER + r")"
        r"\s*(?P<unit>mL|ml|cm3|cm\^3|cc)?",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?P<side>left|right)\s+ovar(?:y|ian)\s+volume\s*[:=]?\s*(?P<value>" + _NUMBER + r")"
        r"\s*(?P<unit>mL|ml|cm3|cm\^3|cc)?",
        re.IGNORECASE,
    ),
    re.compile(
        r"ovarian\s+volume\s*[,:]?\s*(?P<side>left|right)\s*[:=]?\s*(?P<value>" + _NUMBER + r")"
        r"\s*(?P<unit>mL|ml|cm3|cm\^3|cc)?",
        re.IGNORECASE,
    ),
)

_IMPRESSION_LINE = re.compile(
    r"^\s*(?:impression|conclusion|summary)\s*[:\-]\s*(?P<body>.+)$", re.IGNORECASE
)

_PCOM_PRESENT = re.compile(
    r"(polycystic\s+(?:ovarian|ovary)\s+morpholog|pcom|polycystic\s+appearance"
    r"|consistent\s+with\s+polycystic)",
    re.IGNORECASE,
)
_PCOM_ABSENT = re.compile(
    r"(no\s+(?:evidence\s+of\s+)?polycystic|normal\s+ovarian\s+(?:appearance|morpholog)"
    r"|ovaries\s+(?:appear\s+)?normal)",
    re.IGNORECASE,
)
_PCOM_INDETERMINATE = re.compile(
    r"(indeterminate|suboptimal|limited\s+(?:study|visuali[sz]ation)|cannot\s+be\s+excluded"
    r"|equivocal)",
    re.IGNORECASE,
)

_CYST_FLAG = re.compile(
    r"(\d+(?:[.,]\d+)?\s*(?:mm|cm)\s+cyst|large\s+cyst|complex\s+cyst|dominant\s+follicle"
    r"|cystic\s+structure)",
    re.IGNORECASE,
)


class ExtractedReportFinding(BaseModel):
    """One finding from an ultrasound report or clinical summary."""

    finding_id: str
    document_id: str
    patient_id: str

    canonical_code: str
    variable_name: str
    value: float | int | str | bool
    unit: str | None = None
    raw_value: str

    side: Side = "unspecified"
    page_number: int
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    evidence_text: str

    extraction_confidence: float = Field(ge=0.0, le=1.0)
    extractor_version: str = REPORT_EXTRACTOR_VERSION
    warnings: list[str] = Field(default_factory=list)


class ReportExtractionResult(BaseModel):
    """Findings plus anything that was recognized but not groundable."""

    document_id: str
    patient_id: str
    findings: list[ExtractedReportFinding] = Field(default_factory=list)
    unsupported: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ReportExtractor:
    """Extract follicle counts, ovarian volumes and impressions from prose."""

    version = "1.0.0"

    def __init__(self) -> None:
        self._variables = load_variable_registry().variables

    def extract(self, document: ParsedDocument, *, patient_id: str) -> ReportExtractionResult:
        """Extract every groundable finding from an ultrasound/summary document."""
        result = ReportExtractionResult(document_id=document.document_id, patient_id=patient_id)
        counter = 0
        for page in document.pages:
            counter = self._extract_page(document, page, result, counter)
        return result

    def _extract_page(
        self,
        document: ParsedDocument,
        page: DocumentPage,
        result: ReportExtractionResult,
        counter: int,
    ) -> int:
        seen: set[tuple[str, str]] = set()

        for line, start, end in page.line_spans():
            clean = normalize_text(line)
            if not clean.strip():
                continue

            for pattern in _FOLLICLE_PATTERNS:
                for match in pattern.finditer(clean):
                    side = match.group("side").lower()
                    code = f"follicle_count_{side}"
                    if (code, line) in seen:
                        continue
                    seen.add((code, line))
                    counter += 1
                    result.findings.append(
                        self._finding(
                            document,
                            page,
                            counter,
                            result.patient_id,
                            code=code,
                            value=int(match.group("count")),
                            unit="count",
                            raw_value=match.group(0),
                            side=side,  # type: ignore[arg-type]
                            span=(start, end),
                            evidence=line,
                            confidence=0.9,
                        )
                    )

            for pattern in _VOLUME_PATTERNS:
                for match in pattern.finditer(clean):
                    side = match.group("side").lower()
                    value = parse_number(match.group("value"))
                    if value is None:
                        continue
                    if ("ovary_volume_ml", f"{side}{line}") in seen:
                        continue
                    seen.add(("ovary_volume_ml", f"{side}{line}"))
                    counter += 1
                    result.findings.append(
                        self._finding(
                            document,
                            page,
                            counter,
                            result.patient_id,
                            code="ovary_volume_ml",
                            value=value,
                            unit="mL",
                            raw_value=match.group(0),
                            side=side,  # type: ignore[arg-type]
                            span=(start, end),
                            evidence=line,
                            confidence=0.9,
                        )
                    )

            impression = _IMPRESSION_LINE.match(clean)
            if impression:
                counter = self._extract_impression(
                    document, page, result, counter, impression.group("body"), (start, end), line
                )

            if _CYST_FLAG.search(clean):
                counter += 1
                result.findings.append(
                    self._finding(
                        document,
                        page,
                        counter,
                        result.patient_id,
                        code="large_or_uncertain_cystic_structure",
                        value=True,
                        unit=None,
                        raw_value=_CYST_FLAG.search(clean).group(0),  # type: ignore[union-attr]
                        side="unspecified",
                        span=(start, end),
                        evidence=line,
                        confidence=0.7,
                    )
                )
        return counter

    def _extract_impression(
        self,
        document: ParsedDocument,
        page: DocumentPage,
        result: ReportExtractionResult,
        counter: int,
        body: str,
        span: tuple[int, int],
        line: str,
    ) -> int:
        """Map an impression sentence onto ``ovarian_morphology_evidence``.

        Order matters: an explicit negation or an indeterminate qualifier beats a
        positive keyword, because "polycystic morphology cannot be excluded" is
        not a positive finding, and reading it as one would be exactly the kind
        of over-claim this pipeline exists to prevent.
        """
        if _PCOM_INDETERMINATE.search(body):
            value = "indeterminate"
            confidence = 0.75
        elif _PCOM_ABSENT.search(body):
            value = "absent"
            confidence = 0.85
        elif _PCOM_PRESENT.search(body):
            value = "present"
            confidence = 0.85
        else:
            return counter

        counter += 1
        result.findings.append(
            self._finding(
                document,
                page,
                counter,
                result.patient_id,
                code="ovarian_morphology_evidence",
                value=value,
                unit=None,
                raw_value=body.strip(),
                side="unspecified",
                span=span,
                evidence=line,
                confidence=confidence,
            )
        )
        return counter

    def _finding(
        self,
        document: ParsedDocument,
        page: DocumentPage,
        counter: int,
        patient_id: str,
        *,
        code: str,
        value: float | int | str | bool,
        unit: str | None,
        raw_value: str,
        side: Side,
        span: tuple[int, int],
        evidence: str,
        confidence: float,
    ) -> ExtractedReportFinding:
        spec = self._variables.get(code)
        return ExtractedReportFinding(
            finding_id=f"{document.document_id}-rep{counter:03d}",
            document_id=document.document_id,
            patient_id=patient_id,
            canonical_code=code,
            variable_name=spec.canonical_name if spec else code,
            value=value,
            unit=unit,
            raw_value=raw_value,
            side=side,
            page_number=page.page_number,
            char_start=span[0],
            char_end=span[1],
            evidence_text=evidence,
            extraction_confidence=confidence,
        )


def findings_to_events(
    findings: list[ExtractedReportFinding],
    *,
    source_dataset: str | None = None,
    source_file_hash: str | None = None,
    modality: str = "ultrasound_report",
) -> list[HormonalHealthEvent]:
    """Convert report findings into ``document_extracted`` events.

    ``ultrasound_report`` is one of ``EVIDENCE_REQUIRED_MODALITIES``, so the
    schema itself rejects any finding that lost its page grounding on the way
    here — the guarantee is enforced twice, in this module and in the schema.
    """
    events: list[HormonalHealthEvent] = []
    for finding in findings:
        events.append(
            HormonalHealthEvent(
                patient_id=finding.patient_id,
                variable_name=finding.variable_name,
                canonical_variable_code=finding.canonical_code,
                value=finding.value,
                unit=finding.unit,
                raw_value=finding.raw_value,
                raw_unit=finding.unit,
                modality=modality,  # type: ignore[arg-type]
                provenance="document_extracted",
                extraction_confidence=finding.extraction_confidence,
                confirmation_status="awaiting_clinician_confirmation",
                missingness_status="observed",
                source_dataset=source_dataset,
                source_file_id=finding.document_id,
                source_file_hash=source_file_hash,
                source_page=finding.page_number,
                evidence_text=finding.evidence_text,
                parser_version=finding.extractor_version,
            )
        )
    return events
