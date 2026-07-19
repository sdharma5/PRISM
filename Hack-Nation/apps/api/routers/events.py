"""Event ledger routes.

Backed by :class:`event_store.store.EventStore`, which is append-only: a
confirmation or rejection is stored as a new revision rather than an edit, so
the review history of any value is reconstructable from the ledger alone. That
property is why this uses the real store rather than a dict -- an event log a
patient's clinician might later read must not be silently rewritable.

Persistence is process-lifetime by default. Set ``PRISM_EVENT_LOG`` to a path to
mirror appends to JSONL.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field

from event_store.queries import by_patient
from event_store.store import AppendOnlyViolationError, EventStore
from schemas.event import HormonalHealthEvent

router = APIRouter(prefix="/api/v1/events", tags=["events"])


class EventBatch(BaseModel):
    """One or more events to append."""

    model_config = ConfigDict(extra="forbid")

    events: list[HormonalHealthEvent] = Field(min_length=1)


class EventBatchResult(BaseModel):
    stored: int
    patient_ids: list[str]


def build_event_store() -> EventStore:
    """Construct the process-wide ledger, honouring ``PRISM_EVENT_LOG``."""
    configured = os.environ.get("PRISM_EVENT_LOG")
    return EventStore(Path(configured)) if configured else EventStore()


def _store(request: Request) -> EventStore:
    store = getattr(request.app.state, "event_store", None)
    if store is None:  # pragma: no cover - startup always sets this
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The event store is not initialised.",
        )
    return store


@router.post("", response_model=EventBatchResult, status_code=status.HTTP_201_CREATED)
def append_events(payload: EventBatch, request: Request) -> EventBatchResult:
    """Append events to the ledger."""
    store = _store(request)
    try:
        stored = store.extend(payload.events)
    except AppendOnlyViolationError as exc:
        # Re-submitting an event id is a client bug, not a server error: the
        # ledger is append-only by design and will not overwrite.
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

    return EventBatchResult(
        stored=stored,
        patient_ids=sorted({event.patient_id for event in payload.events}),
    )


class ReviewDecision(BaseModel):
    """A patient's or clinician's verdict on one proposed event."""

    model_config = ConfigDict(extra="forbid")

    confirmation_status: Literal["confirmed", "rejected"]
    reviewed_by: str = "patient"


@router.post("/{event_id}/review", response_model=HormonalHealthEvent)
def review_event(event_id: str, payload: ReviewDecision, request: Request) -> HormonalHealthEvent:
    """Record a review decision as a new revision.

    The original event is never mutated. `append_confirmation_revision` writes a
    copy carrying the new status, so the history of a value stays reconstructable
    from the ledger alone -- which is the whole reason the store is append-only.
    A clinician looking at a confirmed value can still see it was once proposed,
    by whom it was confirmed, and when.
    """
    store = _store(request)
    try:
        return store.append_confirmation_revision(
            event_id,
            confirmation_status=payload.confirmation_status,
            reviewed_by=payload.reviewed_by,
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"No event with id '{event_id}'."
        ) from exc


@router.get("/{patient_id}", response_model=list[HormonalHealthEvent])
def get_events(
    patient_id: str, request: Request, include_superseded: bool = False
) -> list[HormonalHealthEvent]:
    """One patient's current events.

    Superseded events -- an earlier extraction of a re-uploaded document, or a
    value replaced by a later revision -- are excluded by default, so callers
    see one live copy of each reading rather than one per upload. Pass
    ``include_superseded=true`` to read the full ledger for audit.

    An unknown patient returns an empty list rather than a 404: "this patient
    has no events yet" is the normal state during onboarding, not an error.
    """
    store = _store(request)
    if include_superseded:
        return by_patient(store.events, patient_id)
    return store.current(patient_id)
