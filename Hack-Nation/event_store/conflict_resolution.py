"""Conflict detection between events describing the same variable.

PRISM's rule: detection never destroys. When two events disagree, *both* stay
in the store and an :class:`~schemas.evidence.EvidenceConflict` is raised for a
human. Snapshot assembly still has to pick one value to hand a model, and it
does so by a fixed, documented order — higher-trust provenance first, then more
recent ``observed_at`` — recording ``n_candidates`` so a reader can always see
that a choice was made.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from datetime import datetime, timedelta

from event_store.provenance import provenance_rank
from schemas.event import HormonalHealthEvent
from schemas.evidence import EvidenceConflict

__all__ = [
    "DUPLICATE_WINDOW",
    "RELATIVE_DIFFERENCE_THRESHOLD",
    "TEMPORAL_DISAGREEMENT_WINDOW",
    "detect_conflicts",
    "select_winner",
]

#: Two numeric values for the same variable disagree when their relative
#: difference exceeds this. 0.10 is chosen to sit above typical inter-assay
#: analytical variation (roughly 5-8% CV for immunoassay hormones) so ordinary
#: measurement noise does not flood reviewers with false conflicts.
RELATIVE_DIFFERENCE_THRESHOLD = 0.10

#: Numerically identical values closer together than this are duplicates
#: (the same measurement ingested twice), not genuine repeat measurements.
DUPLICATE_WINDOW = timedelta(minutes=1)

#: Values that agree numerically but were stamped further apart than this are
#: flagged as a temporal disagreement: one of the timestamps is likely wrong.
TEMPORAL_DISAGREEMENT_WINDOW = timedelta(days=365)

_REVIEW_RESOLUTION = "Human review; snapshot uses higher-trust provenance meanwhile."


def _relative_difference(a: float, b: float) -> float:
    scale = max(abs(a), abs(b))
    if scale == 0.0:
        return 0.0
    return abs(a - b) / scale


def _numeric(event: HormonalHealthEvent) -> float | None:
    if isinstance(event.value, bool) or not isinstance(event.value, (int, float)):
        return None
    return float(event.value)


def select_winner(candidates: Sequence[HormonalHealthEvent]) -> HormonalHealthEvent:
    """Pick the event a snapshot should use, without discarding the others.

    Ordering: (a) higher-trust provenance, then (b) more recent ``observed_at``.
    Events with no timestamp sort oldest, because an unknown time cannot be
    claimed to be recent.
    """

    def key(event: HormonalHealthEvent) -> tuple[int, float]:
        stamp = _naive(event.observed_at) if event.observed_at else datetime.min
        return (provenance_rank(event.provenance), -(stamp - datetime.min).total_seconds())

    return sorted(candidates, key=key)[0]


def _naive(stamp: datetime) -> datetime:
    return stamp.replace(tzinfo=None) if stamp.tzinfo is not None else stamp


def _pair_conflict(a: HormonalHealthEvent, b: HormonalHealthEvent) -> EvidenceConflict | None:
    """Classify the disagreement between two events, if any."""
    ids = [str(a.event_id), str(b.event_id)]
    common = {
        "variable_name": a.variable_name,
        "canonical_variable_code": a.canonical_variable_code,
        "event_ids": ids,
    }

    if a.negated != b.negated:
        return EvidenceConflict(
            **common,
            conflict_type="presence_vs_negation",
            detail=f"One event asserts presence, the other negation ({a.value!r} vs {b.value!r}).",
            recommended_resolution="Ask a clinician which assertion is correct; do not average.",
        )

    a_num, b_num = _numeric(a), _numeric(b)

    if a.unit and b.unit and a.unit != b.unit:
        return EvidenceConflict(
            **common,
            conflict_type="unit_disagreement",
            detail=f"Canonical units differ: '{a.unit}' vs '{b.unit}'.",
            recommended_resolution="Re-run unit conversion; a canonical unit mismatch is a bug.",
        )

    both_stamped = a.observed_at is not None and b.observed_at is not None
    gap = (
        abs(_naive(a.observed_at) - _naive(b.observed_at))  # type: ignore[arg-type]
        if both_stamped
        else None
    )

    if a_num is not None and b_num is not None:
        difference = _relative_difference(a_num, b_num)
        if difference == 0.0 and (gap is None or gap <= DUPLICATE_WINDOW):
            return EvidenceConflict(
                **common,
                conflict_type="duplicate_measurement",
                detail=f"Identical value {a_num} ingested twice.",
                recommended_resolution="Keep both; de-duplicate at snapshot time only.",
                requires_human_review=False,
            )
        if difference > RELATIVE_DIFFERENCE_THRESHOLD:
            return EvidenceConflict(
                **common,
                conflict_type="value_disagreement",
                detail=(
                    f"{a_num} vs {b_num} ({a.unit or '-'}) differ by "
                    f"{difference:.1%}, above the {RELATIVE_DIFFERENCE_THRESHOLD:.0%} threshold."
                ),
                recommended_resolution=_REVIEW_RESOLUTION,
            )
        if gap is not None and gap > TEMPORAL_DISAGREEMENT_WINDOW:
            return EvidenceConflict(
                **common,
                conflict_type="temporal_disagreement",
                detail=f"Agreeing values stamped {gap.days} days apart; a timestamp may be wrong.",
                recommended_resolution="Verify observed_at against the source document.",
            )
        return None

    if a.value != b.value:
        return EvidenceConflict(
            **common,
            conflict_type="value_disagreement",
            detail=f"Non-numeric values disagree: {a.value!r} vs {b.value!r}.",
            recommended_resolution=_REVIEW_RESOLUTION,
        )

    if gap is not None and gap > TEMPORAL_DISAGREEMENT_WINDOW:
        return EvidenceConflict(
            **common,
            conflict_type="temporal_disagreement",
            detail=f"Agreeing values stamped {gap.days} days apart; a timestamp may be wrong.",
            recommended_resolution="Verify observed_at against the source document.",
        )
    if gap is None or gap <= DUPLICATE_WINDOW:
        return EvidenceConflict(
            **common,
            conflict_type="duplicate_measurement",
            detail=f"Identical value {a.value!r} ingested twice.",
            recommended_resolution="Keep both; de-duplicate at snapshot time only.",
            requires_human_review=False,
        )
    return None


def detect_conflicts(events: Iterable[HormonalHealthEvent]) -> list[EvidenceConflict]:
    """Detect all pairwise conflicts, grouped by patient and variable.

    Args:
        events: Candidate events. Only observed events can conflict — a
            missingness record makes no assertion to disagree with.

    Returns:
        One :class:`EvidenceConflict` per disagreeing pair. No event is removed.
    """
    grouped: dict[tuple[str, str], list[HormonalHealthEvent]] = defaultdict(list)
    for event in events:
        if event.missingness_status != "observed":
            continue
        grouped[(event.patient_id, event.canonical_variable_code)].append(event)

    conflicts: list[EvidenceConflict] = []
    for group in grouped.values():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                conflict = _pair_conflict(group[i], group[j])
                if conflict is not None:
                    conflicts.append(conflict)
    return conflicts
