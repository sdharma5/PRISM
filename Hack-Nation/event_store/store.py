"""Append-only event store and snapshot assembly.

The store is a ledger. Events are appended and never mutated or deleted: a
confirmation, a rejection or a correction is a *new* event that supersedes the
old one, and the old one stays readable forever. That is what makes it possible
to answer "what did the model see, and why?" months after a prediction.

Snapshot assembly is where the safety rules bite. A snapshot may only contain
events that are ``observed`` and whose confirmation status the caller
explicitly allowed. Unconfirmed and model-inferred content is excluded and the
reason recorded, never quietly dropped.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from event_store.conflict_resolution import detect_conflicts, select_winner
from event_store.provenance import provenance_rank
from event_store.serialization import append_jsonl, events_from_jsonl, events_to_jsonl
from schemas.event import HormonalHealthEvent
from schemas.evidence import PatientSnapshot, SnapshotValue

__all__ = ["DEFAULT_ALLOWED_CONFIRMATION_STATUSES", "EventStore"]

#: The only confirmation statuses that may reach a model by default. Everything
#: else — awaiting review, rejected — is excluded with a recorded reason.
DEFAULT_ALLOWED_CONFIRMATION_STATUSES: tuple[str, ...] = ("confirmed", "not_required")


class AppendOnlyViolationError(RuntimeError):
    """Raised when caller code tries to mutate or remove a stored event."""


class EventStore:
    """In-memory append-only event ledger with optional file persistence.

    Args:
        path: Optional JSONL file. When given, appends are mirrored to it.
    """

    def __init__(self, path: Path | str | None = None) -> None:
        self._events: list[HormonalHealthEvent] = []
        self._by_patient: dict[str, list[HormonalHealthEvent]] = defaultdict(list)
        self._by_id: dict[str, HormonalHealthEvent] = {}
        #: superseded event id -> the revision that replaced it.
        self._supersessions: dict[str, str] = {}
        self.path = Path(path) if path is not None else None

    # -- Writing ------------------------------------------------------------

    def append(self, event: HormonalHealthEvent) -> HormonalHealthEvent:
        """Append one event. Re-appending an existing event id is a violation."""
        key = str(event.event_id)
        if key in self._by_id:
            raise AppendOnlyViolationError(
                f"Event {key} is already stored. Append a new revision instead of rewriting it."
            )
        self._events.append(event)
        self._by_patient[event.patient_id].append(event)
        self._by_id[key] = event
        if self.path is not None:
            append_jsonl([event], self.path)
        return event

    def extend(self, events: Iterable[HormonalHealthEvent]) -> int:
        """Append many events; returns how many were stored."""
        count = 0
        for event in events:
            self.append(event)
            count += 1
        return count

    def append_confirmation_revision(
        self,
        event_id: str,
        *,
        confirmation_status: str,
        reviewed_by: str,
        reviewed_at: datetime | None = None,
    ) -> HormonalHealthEvent:
        """Record a review decision as a new event rather than an edit.

        The original event keeps its own id, value and status. The revision is a
        copy carrying the new confirmation state, so the review history of a
        value is reconstructable from the ledger alone.
        """
        original = self._by_id.get(str(event_id))
        if original is None:
            raise KeyError(f"No event with id '{event_id}'.")
        payload = original.model_dump()
        payload.pop("event_id")  # the revision is a new event, with a new id
        payload["confirmation_status"] = confirmation_status
        payload["reviewed_by"] = reviewed_by
        payload["reviewed_at"] = reviewed_at or datetime.now(UTC)
        # Re-validate: a review decision must still satisfy every event contract.
        revision = HormonalHealthEvent.model_validate(payload)
        self._supersessions[str(original.event_id)] = str(revision.event_id)
        return self.append(revision)

    # -- Reading ------------------------------------------------------------

    @property
    def events(self) -> tuple[HormonalHealthEvent, ...]:
        """Immutable view of the ledger."""
        return tuple(self._events)

    def __len__(self) -> int:
        return len(self._events)

    def get(self, patient_id: str) -> list[HormonalHealthEvent]:
        """All events for one patient, in append order."""
        return list(self._by_patient.get(patient_id, ()))

    def get_by_id(self, event_id: str) -> HormonalHealthEvent | None:
        """One event by id, or None."""
        return self._by_id.get(str(event_id))

    def patient_ids(self) -> list[str]:
        """Every patient id in the store."""
        return sorted(self._by_patient)

    def mark_superseded(self, event_id: str, *, replaced_by: str) -> None:
        """Record that a stored event has been replaced by a later one.

        Used when the same source document is ingested again: the earlier
        extraction is not wrong, it is simply no longer current. Both events
        stay in the ledger -- this only records which one supersedes which, so
        readers can show the current view without losing the history.
        """
        for key in (str(event_id), str(replaced_by)):
            if key not in self._by_id:
                raise KeyError(f"No event with id '{key}'.")
        self._supersessions[str(event_id)] = str(replaced_by)

    def superseded_ids(self) -> set[str]:
        """Ids of events replaced by a later revision or re-ingestion."""
        return set(self._supersessions)

    def current(self, patient_id: str) -> list[HormonalHealthEvent]:
        """One patient's events with superseded ones filtered out."""
        superseded = self._supersessions
        return [e for e in self.get(patient_id) if str(e.event_id) not in superseded]

    # -- Persistence --------------------------------------------------------

    def save(self, path: Path | str | None = None) -> Path:
        """Write the whole ledger to JSONL."""
        target = Path(path) if path is not None else self.path
        if target is None:
            raise ValueError("No path configured for this store.")
        return events_to_jsonl(self._events, target)

    def save_parquet(self, path: Path | str) -> Path:
        """Write the ledger to Parquet (optional ``pyarrow`` extra)."""
        from event_store.serialization import events_to_parquet

        return events_to_parquet(self._events, path)

    @classmethod
    def load(cls, path: Path | str, *, attach: bool = False) -> EventStore:
        """Read a ledger from JSONL.

        Args:
            path: Source JSONL file.
            attach: When True, later appends are mirrored back to this file.
        """
        store = cls(path if attach else None)
        for event in events_from_jsonl(path):
            store.append(event)
        return store

    # -- Snapshot -----------------------------------------------------------

    def build_snapshot(
        self,
        patient_id: str,
        as_of: datetime | None = None,
        allowed_confirmation_statuses: Sequence[str] = DEFAULT_ALLOWED_CONFIRMATION_STATUSES,
        include_modalities: Sequence[str] | None = None,
    ) -> PatientSnapshot:
        """Assemble a model-ready :class:`PatientSnapshot`.

        An event is included only if it is observed, dated at or before
        ``as_of`` (or undated), in an allowed confirmation status, and from an
        included modality. Every exclusion is recorded with its reason.

        Args:
            patient_id: Patient to snapshot. Never spans datasets.
            as_of: Point in time; defaults to now. Later events are excluded so
                a snapshot can be replayed without leaking the future.
            allowed_confirmation_statuses: Statuses permitted to reach a model.
            include_modalities: Optional modality allowlist.

        Returns:
            A :class:`PatientSnapshot` with selected values, missingness mask,
            conflicts, and the ids and reasons for every excluded event.
        """
        as_of = as_of or datetime.now(UTC)
        as_of_naive = _naive(as_of)
        allowed = set(allowed_confirmation_statuses)
        modality_filter = set(include_modalities) if include_modalities is not None else None

        candidates: dict[str, list[HormonalHealthEvent]] = defaultdict(list)
        excluded_ids: list[str] = []
        reasons: dict[str, str] = {}
        warnings: list[str] = []
        superseded = self.superseded_ids()

        for event in self.get(patient_id):
            key = str(event.event_id)
            reason = self._exclusion_reason(
                event, as_of_naive, allowed, modality_filter, superseded
            )
            if reason is not None:
                excluded_ids.append(key)
                reasons[key] = reason
                continue
            candidates[event.canonical_variable_code].append(event)

        values: dict[str, SnapshotValue] = {}
        for code, group in candidates.items():
            winner = select_winner(group)
            recency = (
                (as_of_naive - _naive(winner.observed_at)).total_seconds() / 86400.0
                if winner.observed_at is not None
                else None
            )
            values[code] = SnapshotValue(
                canonical_variable_code=code,
                value=winner.value,
                unit=winner.unit,
                missingness_status=winner.missingness_status,
                observed_at=winner.observed_at,
                recency_days=recency,
                source_event_id=str(winner.event_id),
                provenance=winner.provenance,
                confirmation_status=winner.confirmation_status,
                quality=_quality(winner),
                n_candidates=len(group),
            )
            for loser in group:
                if loser is winner:
                    continue
                # The loser stays in the store; the snapshot only notes it was
                # not selected, so nothing is lost by taking a snapshot.
                key = str(loser.event_id)
                excluded_ids.append(key)
                reasons[key] = f"not_selected_for_{code}: lower provenance trust or older"

        conflicts = detect_conflicts(e for group in candidates.values() for e in group)
        if conflicts:
            warnings.append(
                f"{len(conflicts)} unresolved evidence conflict(s) for {patient_id}; "
                "selected values are provisional pending human review."
            )

        return PatientSnapshot(
            patient_id=patient_id,
            as_of=as_of,
            created_at=datetime.now(UTC),
            values=values,
            missingness_mask={
                code: v.missingness_status != "observed" for code, v in values.items()
            },
            conflicts=conflicts,
            excluded_event_ids=excluded_ids,
            exclusion_reasons=reasons,
            included_modalities=sorted(modality_filter) if modality_filter else [],
            allowed_confirmation_statuses=sorted(allowed),
            warnings=warnings,
        )

    @staticmethod
    def _exclusion_reason(
        event: HormonalHealthEvent,
        as_of: datetime,
        allowed: set[str],
        modality_filter: set[str] | None,
        superseded: set[str],
    ) -> str | None:
        if str(event.event_id) in superseded:
            return "superseded_by_later_confirmation_revision"
        if event.confirmation_status not in allowed:
            return f"confirmation_status='{event.confirmation_status}' not in allowed set"
        if event.missingness_status != "observed":
            return f"missingness_status='{event.missingness_status}'"
        if modality_filter is not None and event.modality not in modality_filter:
            return f"modality='{event.modality}' not in include_modalities"
        if event.observed_at is not None and _naive(event.observed_at) > as_of:
            return "observed_at is after as_of"
        return None


def _quality(event: HormonalHealthEvent) -> float:
    """Cheap quality proxy: extraction confidence discounted by provenance trust.

    Deliberately monotonic and simple — a snapshot consumer should treat this as
    a sorting aid, not as a calibrated probability.
    """
    rank = provenance_rank(event.provenance)
    penalty = 1.0 - min(rank, 6) * 0.05
    score = event.extraction_confidence * penalty
    if event.uncertain:
        score *= 0.8
    return round(max(0.0, min(1.0, score)), 4)


def _naive(stamp: datetime) -> datetime:
    return stamp.replace(tzinfo=None) if stamp.tzinfo is not None else stamp


def snapshot_to_dict(snapshot: PatientSnapshot) -> dict[str, Any]:
    """JSON-safe dict for a snapshot."""
    return snapshot.model_dump(mode="json")
