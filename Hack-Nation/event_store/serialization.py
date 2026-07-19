"""Serialization for the event store.

JSONL is the reference format because it is append-only by nature, diffable,
and readable without any binary dependency — properties that matter more for an
auditable clinical record than read speed. Parquet is offered as an optional
accelerator via a lazy ``pyarrow`` import so it never becomes a hard dependency.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

from schemas.event import HormonalHealthEvent

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

__all__ = [
    "events_to_dataframe",
    "events_to_jsonl",
    "events_to_parquet",
    "events_from_dataframe",
    "events_from_jsonl",
    "events_from_parquet",
    "iter_jsonl",
]


def events_to_jsonl(events: Iterable[HormonalHealthEvent], path: Path | str) -> Path:
    """Write events as one JSON object per line."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w") as fh:
        for event in events:
            fh.write(json.dumps(event.model_dump(mode="json")) + "\n")
    return target


def append_jsonl(events: Iterable[HormonalHealthEvent], path: Path | str) -> Path:
    """Append events to an existing JSONL file, creating it if needed."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a") as fh:
        for event in events:
            fh.write(json.dumps(event.model_dump(mode="json")) + "\n")
    return target


def iter_jsonl(path: Path | str) -> Iterator[HormonalHealthEvent]:
    """Stream events from a JSONL file without loading it all into memory."""
    with Path(path).open() as fh:
        for line in fh:
            text = line.strip()
            if text:
                yield HormonalHealthEvent.model_validate(json.loads(text))


def events_from_jsonl(path: Path | str) -> list[HormonalHealthEvent]:
    """Read all events from a JSONL file."""
    return list(iter_jsonl(path))


def events_to_dataframe(events: Iterable[HormonalHealthEvent]) -> pd.DataFrame:
    """Flatten events into a pandas DataFrame, one row per event."""
    import pandas as pd

    rows: list[dict[str, Any]] = [e.model_dump(mode="json") for e in events]
    if not rows:
        return pd.DataFrame(columns=list(HormonalHealthEvent.model_fields))
    return pd.DataFrame(rows)


def events_from_dataframe(frame: pd.DataFrame) -> list[HormonalHealthEvent]:
    """Rebuild events from a DataFrame produced by :func:`events_to_dataframe`."""
    import pandas as pd

    records = frame.replace({pd.NA: None}).to_dict(orient="records")
    return [
        HormonalHealthEvent.model_validate(
            {k: (None if _is_nan(v) else v) for k, v in record.items()}
        )
        for record in records
    ]


def _is_nan(value: Any) -> bool:
    return isinstance(value, float) and value != value


def events_to_parquet(events: Iterable[HormonalHealthEvent], path: Path | str) -> Path:
    """Write events to Parquet. Requires the optional ``pyarrow`` extra."""
    frame = events_to_dataframe(events)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pyarrow  # noqa: F401
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "Parquet support requires pyarrow. Use JSONL, or install pyarrow."
        ) from exc
    # value/raw_value are intentionally heterogeneous, so they are stringified
    # rather than allowed to force a lossy column type.
    frame = frame.astype({"value": "string", "raw_value": "string"})
    frame.to_parquet(target, index=False)
    return target


def events_from_parquet(path: Path | str) -> list[HormonalHealthEvent]:
    """Read events back from Parquet. Requires the optional ``pyarrow`` extra."""
    import pandas as pd

    try:
        import pyarrow  # noqa: F401
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "Parquet support requires pyarrow. Use JSONL, or install pyarrow."
        ) from exc
    return events_from_dataframe(pd.read_parquet(path))
