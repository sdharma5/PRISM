"""Text normalization for values that came out of a document.

Why this is a separate module: OCR and PDF text layers mangle exactly the
characters that carry numeric meaning — a unicode minus that is not ASCII "-", a
micro sign that is not "u", a decimal comma that would otherwise parse as a
thousands separator. Getting one of these wrong turns 1.234 into 1234, which is
a three-orders-of-magnitude error that no valid-range check on a wide-range
analyte will necessarily catch.

Every function here is pure and reversible in the sense that matters: the caller
keeps the original string, and normalization output is stored alongside it,
never in place of it.
"""

from __future__ import annotations

import re
from datetime import date, datetime

#: Unicode characters that mean "minus" but are not ASCII hyphen-minus.
_MINUS_VARIANTS = {
    "−": "-",  # minus sign
    "–": "-",  # en dash
    "—": "-",  # em dash
    "‒": "-",  # figure dash
    "‐": "-",  # hyphen
    "‑": "-",  # non-breaking hyphen
}

#: Micro-sign variants; both normalize to ASCII "u" to match registry aliases.
_MICRO_VARIANTS = {"µ": "u", "μ": "u"}

_SPACE_VARIANTS = {" ": " ", " ": " ", " ": " "}

_DATE_FORMATS = (
    "%Y-%m-%d",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%d-%m-%Y",
    "%d %b %Y",
    "%d %B %Y",
    "%b %d, %Y",
    "%B %d, %Y",
    "%d.%m.%Y",
    "%Y/%m/%d",
)


def normalize_text(value: str) -> str:
    """Fold unicode minus, micro and space variants into ASCII."""
    result = value
    for source, target in {**_MINUS_VARIANTS, **_MICRO_VARIANTS, **_SPACE_VARIANTS}.items():
        result = result.replace(source, target)
    return result


def normalize_unit_text(unit: str) -> str:
    """Normalize a unit string before it reaches ``registry.loader.normalize_unit``.

    Only character-level folding happens here (µg/dL -> ug/dL). Unit *meaning*
    stays the registry's job — this module must never learn a conversion factor.
    """
    return re.sub(r"\s+", "", normalize_text(unit)).strip()


def parse_number(raw: str) -> float | None:
    """Parse a numeric string with European or US separators.

    Disambiguation rules, in order:
      1. Both "." and "," present: the *rightmost* is the decimal separator.
      2. Only "," present, with exactly two or one digits after it: decimal comma.
      3. Only "," present otherwise (e.g. "1,234"): thousands separator.

    Returns ``None`` rather than guessing when the string is not a number, so
    the caller can record an extraction failure instead of inventing a value.
    """
    text = normalize_text(str(raw)).strip()
    text = re.sub(r"[^\d.,\-+eE]", "", text)
    if not text or not re.search(r"\d", text):
        return None

    has_dot = "." in text
    has_comma = "," in text

    if has_dot and has_comma:
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif has_comma:
        decimals = len(text.split(",")[-1])
        if decimals in (1, 2) and text.count(",") == 1:
            text = text.replace(",", ".")
        else:
            text = text.replace(",", "")

    try:
        return float(text)
    except ValueError:
        return None


def parse_date(raw: str) -> date | None:
    """Parse a date in any of the formats the fixtures and real reports use.

    Ambiguous day/month orders (03/04/2024) are resolved by trying ISO first,
    then day-first, then month-first — and the caller always keeps the original
    string, so an ambiguity is recoverable rather than baked in.
    """
    text = normalize_text(str(raw)).strip()
    if not text:
        return None
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def normalize_test_name(name: str) -> str:
    """Canonical lookup key for a source test name: lowercase, alphanumeric."""
    folded = normalize_text(name).lower()
    folded = re.sub(r"[^a-z0-9]+", " ", folded)
    return re.sub(r"\s+", " ", folded).strip()
