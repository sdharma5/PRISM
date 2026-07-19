"""Validation for the speech pipeline.

These checks are the last line of defence before a speech extraction is offered
to a human reviewer. They flag rather than repair: an extractor that produces a
span pointing at the wrong text has a bug, and silently re-anchoring the span
would hide it.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ingestion.speech.extraction import ExtractedSymptomEvent
from ingestion.speech.transcription import Transcript
from registry.loader import in_valid_range, load_variable_registry

Severity = Literal["error", "warning"]


class ValidationIssue(BaseModel):
    """One problem found with one extraction."""

    extraction_id: str
    code: str
    severity: Severity
    message: str


class SpeechValidationReport(BaseModel):
    """Aggregate validation outcome for one recording."""

    recording_id: str
    n_checked: int = 0
    issues: list[ValidationIssue] = Field(default_factory=list)
    unsupported_extraction_ids: list[str] = Field(default_factory=list)

    @property
    def n_errors(self) -> int:
        return sum(1 for i in self.issues if i.severity == "error")

    @property
    def unsupported_rate(self) -> float:
        """Fraction of extractions whose evidence span does not check out."""
        if self.n_checked == 0:
            return 0.0
        return len(self.unsupported_extraction_ids) / self.n_checked

    @property
    def ok(self) -> bool:
        return self.n_errors == 0


def span_is_supported(transcript: Transcript, event: ExtractedSymptomEvent) -> bool:
    """True when the event's span really points at its own evidence text.

    This is the check that catches a hallucinating extractor: it is not enough
    for an event to *carry* a span, the span must reproduce the quoted text.
    """
    span = event.evidence
    if span.char_end > len(transcript.text):
        return False
    return transcript.text[span.char_start : span.char_end] == span.text


def validate_extractions(
    transcript: Transcript,
    events: list[ExtractedSymptomEvent],
) -> SpeechValidationReport:
    """Check every extraction for grounding, registry validity and coherence."""
    registry = load_variable_registry().variables
    report = SpeechValidationReport(recording_id=transcript.recording_id, n_checked=len(events))

    for event in events:
        code = event.canonical_code

        if not span_is_supported(transcript, event):
            report.unsupported_extraction_ids.append(event.extraction_id)
            report.issues.append(
                ValidationIssue(
                    extraction_id=event.extraction_id,
                    code=code,
                    severity="error",
                    message=(
                        "evidence span does not match the transcript text at its offsets; "
                        "the extraction is unsupported and must not be offered for confirmation."
                    ),
                )
            )

        if event.evidence.start_seconds is None:
            report.issues.append(
                ValidationIssue(
                    extraction_id=event.extraction_id,
                    code=code,
                    severity="warning",
                    message="no audio timing could be linked to this span.",
                )
            )

        if code not in registry:
            report.issues.append(
                ValidationIssue(
                    extraction_id=event.extraction_id,
                    code=code,
                    severity="error",
                    message=f"'{code}' is not in registry/variables.yaml.",
                )
            )
            continue

        if code.startswith("family_history_") and event.attribution != "family_member":
            report.issues.append(
                ValidationIssue(
                    extraction_id=event.extraction_id,
                    code=code,
                    severity="error",
                    message="family-history code without family attribution.",
                )
            )

        is_number = isinstance(event.value, (int, float)) and not isinstance(event.value, bool)
        if is_number and not in_valid_range(code, float(event.value)):
            report.issues.append(
                ValidationIssue(
                    extraction_id=event.extraction_id,
                    code=code,
                    severity="warning",
                    message=(
                        f"value {event.value} is outside the registry valid range; "
                        "flagged, not corrected."
                    ),
                )
            )

        if event.negated and event.uncertain:
            report.issues.append(
                ValidationIssue(
                    extraction_id=event.extraction_id,
                    code=code,
                    severity="warning",
                    message="both negated and uncertain; needs clinician adjudication.",
                )
            )

    return report


def drop_unsupported(
    transcript: Transcript, events: list[ExtractedSymptomEvent]
) -> tuple[list[ExtractedSymptomEvent], list[ExtractedSymptomEvent]]:
    """Split extractions into (supported, unsupported).

    Callers must persist the unsupported list — it is a metric, not garbage.
    """
    supported: list[ExtractedSymptomEvent] = []
    unsupported: list[ExtractedSymptomEvent] = []
    for event in events:
        (supported if span_is_supported(transcript, event) else unsupported).append(event)
    return supported, unsupported
