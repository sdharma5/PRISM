"""Structural validation for the PCOS tabular source.

Validation is deliberately loud and pre-emptive: a column we do not recognise
is an error, because the alternative is silently discarding a clinical variable
and reporting a smaller-but-clean dataset that no longer represents the source.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from ingestion.tabular_pcos.mapping import canonical_code_for, excluded_reason_for
from registry.loader import load_variable_registry

__all__ = [
    "unaccounted_columns",
    "validate_columns",
    "validate_patient_ids",
]


def unaccounted_columns(columns: Iterable[str]) -> list[str]:
    """Return source columns that are neither mapped nor documented as excluded."""
    return [c for c in columns if canonical_code_for(c) is None and excluded_reason_for(c) is None]


def validate_columns(columns: Sequence[str], id_column: str) -> list[str]:
    """Check the header row.

    Args:
        columns: Header names as read from the source file.
        id_column: The column used as the patient identifier.

    Returns:
        A list of human-readable validation errors; empty means valid.
    """
    errors: list[str] = []

    if id_column not in columns:
        errors.append(f"Missing patient id column '{id_column}'.")

    for column in unaccounted_columns(columns):
        if column == id_column:
            continue
        errors.append(
            f"Column '{column}' is neither in SOURCE_COLUMN_MAP nor EXCLUDED_COLUMNS. "
            "Map it or document why it is excluded."
        )

    registry = load_variable_registry().variables
    for column in columns:
        code = canonical_code_for(column)
        if code is not None and code not in registry:
            errors.append(f"Column '{column}' maps to unknown canonical code '{code}'.")

    return errors


def validate_patient_ids(ids: Sequence[str]) -> list[str]:
    """Reject empty or duplicated patient identifiers.

    Duplicate ids in a cross-sectional dataset mean two different people would
    be merged into one record, which is never recoverable downstream.
    """
    errors: list[str] = []
    seen: set[str] = set()
    duplicates: set[str] = set()
    for index, pid in enumerate(ids):
        text = str(pid).strip()
        if not text or text.lower() in {"nan", "none"}:
            errors.append(f"Row {index}: empty patient identifier.")
            continue
        if text in seen:
            duplicates.add(text)
        seen.add(text)
    if duplicates:
        errors.append(f"Duplicate patient identifiers: {sorted(duplicates)}")
    return errors
