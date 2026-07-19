"""Validation for document extraction. Flags, never corrects.

The distinction matters more here than anywhere else in PRISM. If a report says
testosterone is 780 ng/dL and the registry's plausible ceiling is 1000, the value
is unusual but real. If it says 7800, the likeliest explanations are a decimal
point lost in the PDF text layer or a genuinely extraordinary result — and those
demand opposite responses. Clipping to the ceiling would erase the evidence
needed to tell them apart, so the value is recorded as extracted and flagged.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ingestion.documents.lab_extractor import ExtractedLabResult
from ingestion.documents.parser import ParsedDocument
from registry.loader import in_valid_range, load_variable_registry

Severity = Literal["error", "warning", "info"]


class DocumentValidationIssue(BaseModel):
    """One problem found with one extracted value."""

    result_id: str
    code: str
    severity: Severity
    message: str


class DocumentValidationReport(BaseModel):
    """Aggregate validation outcome for one document."""

    document_id: str
    n_checked: int = 0
    issues: list[DocumentValidationIssue] = Field(default_factory=list)
    ungrounded_result_ids: list[str] = Field(default_factory=list)

    @property
    def n_errors(self) -> int:
        return sum(1 for i in self.issues if i.severity == "error")

    @property
    def ok(self) -> bool:
        return self.n_errors == 0

    @property
    def grounding_accuracy(self) -> float:
        """Share of extracted values whose page span reproduces their evidence."""
        if self.n_checked == 0:
            return 0.0
        return round(1.0 - len(self.ungrounded_result_ids) / self.n_checked, 4)


def is_grounded(document: ParsedDocument, result: ExtractedLabResult) -> bool:
    """True when the recorded span really contains the recorded evidence text.

    Checks three things, because a span can be wrong in three ways: the page must
    exist, the span must reproduce the evidence text, and the evidence must
    actually mention the value that was extracted from it.
    """
    page = next((p for p in document.pages if p.page_number == result.page_number), None)
    if page is None:
        return False
    if result.char_end > len(document.text):
        return False
    if document.text[result.char_start : result.char_end] != result.evidence_text:
        return False
    return _mentions_value(result.evidence_text, result.value_source)


def _mentions_value(evidence: str, value: float) -> bool:
    """True when the evidence line contains the extracted number in some form."""
    candidates = {
        f"{value:g}",
        f"{value}",
        f"{value:.1f}",
        f"{value:.2f}",
        str(int(value)) if float(value).is_integer() else "",
    }
    normalized = evidence.replace(",", ".")
    return any(candidate and candidate in normalized for candidate in candidates)


def validate_lab_results(
    document: ParsedDocument,
    results: list[ExtractedLabResult],
) -> DocumentValidationReport:
    """Check grounding, registry membership, ranges and reference consistency."""
    registry = load_variable_registry().variables
    report = DocumentValidationReport(document_id=document.document_id, n_checked=len(results))

    for result in results:
        code = result.canonical_code

        if not is_grounded(document, result):
            report.ungrounded_result_ids.append(result.result_id)
            report.issues.append(
                DocumentValidationIssue(
                    result_id=result.result_id,
                    code=code,
                    severity="error",
                    message=(
                        "page span does not reproduce the evidence text or the extracted "
                        "value; the result must be dropped, not stored."
                    ),
                )
            )

        if code not in registry:
            report.issues.append(
                DocumentValidationIssue(
                    result_id=result.result_id,
                    code=code,
                    severity="error",
                    message=f"'{code}' is not in registry/variables.yaml.",
                )
            )
            continue

        if result.value_canonical is None:
            report.issues.append(
                DocumentValidationIssue(
                    result_id=result.result_id,
                    code=code,
                    severity="warning",
                    message=(
                        f"no canonical value: unit '{result.unit_source}' could not be "
                        "converted. Source value preserved."
                    ),
                )
            )
            continue

        if not in_valid_range(code, result.value_canonical):
            report.issues.append(
                DocumentValidationIssue(
                    result_id=result.result_id,
                    code=code,
                    severity="warning",
                    message=(
                        f"{result.value_canonical} {result.unit_canonical} is outside the "
                        "registry valid range. Flagged for review, not corrected."
                    ),
                )
            )

        if result.reference_range is not None:
            inside = result.reference_range.contains(result.value_source)
            if inside is False:
                report.issues.append(
                    DocumentValidationIssue(
                        result_id=result.result_id,
                        code=code,
                        severity="info",
                        message=(
                            f"value {result.value_source} is outside the report's own "
                            f"reference range ({result.reference_range.text})."
                        ),
                    )
                )

    return report


def drop_ungrounded(
    document: ParsedDocument, results: list[ExtractedLabResult]
) -> tuple[list[ExtractedLabResult], list[ExtractedLabResult]]:
    """Split results into (grounded, ungrounded). Callers must count the latter."""
    grounded: list[ExtractedLabResult] = []
    ungrounded: list[ExtractedLabResult] = []
    for result in results:
        (grounded if is_grounded(document, result) else ungrounded).append(result)
    return grounded, ungrounded
