"""Value-level cleaning for the PMOS tabular source.

Cleaning here is strictly *interpretation*, never *repair*. A cell is either
understood (returned as a typed value) or not understood (returned as ``None``
with a reason). Nothing is imputed, clipped, or rounded into plausibility —
downstream models must be able to distinguish "10 mIU/mL" from "we could not
read this cell", and a silently repaired value destroys that distinction.
"""

from __future__ import annotations

from dataclasses import dataclass

from registry.loader import in_valid_range, load_variable_registry

__all__ = [
    "CleanResult",
    "clean_value",
    "coerce_bool",
    "coerce_numeric",
    "normalize_regularity",
]

#: Strings that mean "no value here". Compared case-insensitively.
NULL_TOKENS: frozenset[str] = frozenset(
    {"", "na", "n/a", "nan", "null", "none", "-", "--", ".", "?", "unknown"}
)

_TRUE_TOKENS: frozenset[str] = frozenset({"y", "yes", "1", "1.0", "true", "t", "present"})
_FALSE_TOKENS: frozenset[str] = frozenset({"n", "no", "0", "0.0", "false", "f", "absent"})


@dataclass(frozen=True)
class CleanResult:
    """Outcome of cleaning one cell.

    Attributes:
        value: The typed value, or ``None`` when unusable.
        missingness_status: ``observed`` only when ``value`` is not ``None``.
        reason: Human-readable explanation when the value was not usable.
    """

    value: float | int | bool | str | None
    missingness_status: str
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.missingness_status == "observed"


def _is_null(raw: object) -> bool:
    if raw is None:
        return True
    text = str(raw).strip().lower()
    return text in NULL_TOKENS


def coerce_bool(raw: object) -> bool | None:
    """Interpret a Y/N-style cell as a bool, or ``None`` if unrecognised."""
    if _is_null(raw):
        return None
    if isinstance(raw, bool):
        return raw
    text = str(raw).strip().lower()
    if text in _TRUE_TOKENS:
        return True
    if text in _FALSE_TOKENS:
        return False
    return None


def coerce_numeric(raw: object) -> float | None:
    """Interpret a cell as a float, tolerating stray whitespace and commas."""
    if _is_null(raw):
        return None
    if isinstance(raw, bool):
        return float(raw)
    if isinstance(raw, (int, float)):
        return float(raw)
    text = str(raw).strip().replace(",", "")
    try:
        return float(text)
    except ValueError:
        return None


def normalize_regularity(raw: object) -> str | None:
    """Map the source's R/I cycle code onto the canonical category set."""
    if _is_null(raw):
        return None
    text = str(raw).strip().lower()
    if text in {"r", "regular", "2"}:
        return "regular"
    if text in {"i", "irregular", "4", "5"}:
        return "irregular"
    if text in {"a", "absent", "amenorrhea"}:
        return "absent"
    return None


def clean_value(code: str, raw: object, source_unit: str | None = None) -> CleanResult:
    """Clean one cell for a canonical variable, honouring its registry type.

    Args:
        code: Canonical variable code.
        raw: The raw cell contents.
        source_unit: Unit the source expresses this column in. Range checking is
            skipped here when it differs from the canonical unit, because
            comparing e.g. inches against a centimetre range would flag healthy
            values as implausible; ``emit_event`` re-checks post-conversion.

    Returns:
        A :class:`CleanResult`. Out-of-range numerics come back as
        ``not_available`` with the reason recorded, so the caller can log them
        to ``dropped_records`` rather than pretending they were never seen.
    """
    spec = load_variable_registry().variables.get(code)
    if _is_null(raw):
        return CleanResult(None, "not_collected", "empty or null token in source cell")

    var_type = spec.type if spec else "continuous"

    if var_type == "binary":
        flag = coerce_bool(raw)
        if flag is None:
            return CleanResult(None, "not_available", f"unparseable Y/N value {raw!r}")
        return CleanResult(flag, "observed")

    if var_type == "categorical":
        if code == "cycle_regularity":
            category = normalize_regularity(raw)
        else:
            category = str(raw).strip().lower()
            if spec and spec.categories and category not in spec.categories:
                category = None
        if category is None:
            return CleanResult(None, "not_available", f"unrecognised category {raw!r}")
        return CleanResult(category, "observed")

    if var_type == "text":
        text = str(raw).strip()
        return CleanResult(text, "observed") if text else CleanResult(None, "not_collected", "")

    number = coerce_numeric(raw)
    if number is None:
        return CleanResult(None, "not_available", f"non-numeric value {raw!r}")

    canonical_unit = (spec.canonical_unit or spec.unit) if spec else None
    same_unit = source_unit is None or canonical_unit is None or source_unit == canonical_unit
    if spec and spec.valid_range is not None and same_unit and not in_valid_range(code, number):
        return CleanResult(
            None,
            "not_available",
            f"value {number} outside registry valid_range for {code}",
        )

    if var_type == "integer":
        return CleanResult(int(round(number)), "observed")
    return CleanResult(number, "observed")
