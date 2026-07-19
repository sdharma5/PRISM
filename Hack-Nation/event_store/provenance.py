"""Provenance trust ordering and source tracing.

The ordering below is the single place where PRISM decides which kind of
evidence outranks which. It is trust in *who or what asserted the value*, not
in how recent or how precise it is: a clinician-confirmed number beats a
model-inferred one even when the model is newer, because a wrong automated
value entering a clinical snapshot is the failure mode we most need to avoid.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from schemas.event import HormonalHealthEvent

__all__ = [
    "PROVENANCE_TRUST_ORDER",
    "provenance_rank",
    "trace_event",
]

#: Most trusted first. Index in this tuple is the trust rank (lower is better).
PROVENANCE_TRUST_ORDER: tuple[str, ...] = (
    "clinician_confirmed",
    "patient_confirmed",
    "device_measured",
    "dataset_provided",
    "document_extracted",
    "model_measured",
    "model_inferred",
)

_RANK: dict[str, int] = {name: i for i, name in enumerate(PROVENANCE_TRUST_ORDER)}


def provenance_rank(provenance: str | None) -> int:
    """Trust rank for a provenance value; unknown provenance ranks last."""
    return _RANK.get(provenance or "", len(PROVENANCE_TRUST_ORDER))


def trace_event(event_id: str, events: Iterable[HormonalHealthEvent]) -> dict[str, Any]:
    """Trace one event back to the bytes and process that produced it.

    Args:
        event_id: The event's UUID as a string.
        events: The population of events to search.

    Returns:
        A dict describing the source file, hash, location and parser.

    Raises:
        KeyError: If no event with that id exists.
    """
    for event in events:
        if str(event.event_id) == str(event_id):
            return {
                "event_id": str(event.event_id),
                "patient_id": event.patient_id,
                "canonical_variable_code": event.canonical_variable_code,
                "source_dataset": event.source_dataset,
                "source_file_id": event.source_file_id,
                "source_file_hash": event.source_file_hash,
                "source_page": event.source_page,
                "source_time_start_seconds": event.source_time_start_seconds,
                "source_time_end_seconds": event.source_time_end_seconds,
                "evidence_text": event.evidence_text,
                "provenance": event.provenance,
                "provenance_rank": provenance_rank(event.provenance),
                "confirmation_status": event.confirmation_status,
                "reviewed_by": event.reviewed_by,
                "parser_version": event.parser_version,
                "model_version": event.model_version,
                "raw_value": event.raw_value,
                "raw_unit": event.raw_unit,
                "value": event.value,
                "unit": event.unit,
            }
    raise KeyError(f"No event with id '{event_id}' in the supplied population.")
