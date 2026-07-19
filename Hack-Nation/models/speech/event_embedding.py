"""Deterministic embedding of speech-derived events.

Why not a neural encoder: at this stage there is no paired speech corpus to
train one on, and an untrained encoder would produce numbers that look learned
but mean nothing. A multi-hot vector over the canonical code vocabulary is
honest about what it is — a presence indicator — and is byte-for-byte
reproducible, which the stability evaluation depends on.

The vocabulary is derived from the lexicon, so adding a concept there
automatically widens the vector. A hashed fallback dimension exists for codes
that are not in the lexicon (e.g. a reviewer-corrected code), so an unknown code
degrades the representation slightly instead of raising.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache

from ingestion.speech.extraction import load_lexicon

EMBEDDING_VERSION = "speech_multihot/1.0.0"

#: Extra slots appended after the known vocabulary, addressed by stable hashing.
HASHED_BUCKETS = 16


@lru_cache(maxsize=1)
def build_vocabulary() -> tuple[str, ...]:
    """Sorted tuple of every canonical code the lexicon can emit."""
    lexicon = load_lexicon()
    codes: set[str] = set()
    for concept in lexicon["concepts"]:
        if concept.get("code"):
            codes.add(str(concept["code"]))
        if concept.get("family_code"):
            codes.add(str(concept["family_code"]))
    if lexicon.get("medications"):
        codes.add("medication_current")
    codes.update({"cycle_length", "menstrual_frequency_per_year"})
    return tuple(sorted(codes))


def embedding_dimension() -> int:
    """Total vector length: one slot per known code, times three assertion
    channels (present / negated / historical), plus the hashed buckets."""
    return len(build_vocabulary()) * 3 + HASHED_BUCKETS


def _hashed_index(code: str) -> int:
    digest = hashlib.sha256(code.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % HASHED_BUCKETS


def encode_codes(
    present: list[str],
    negated: list[str] | None = None,
    historical: list[str] | None = None,
) -> list[float]:
    """Build the multi-hot vector.

    Three separate channels rather than one signed value, because "patient
    denies acne" and "patient had acne years ago" are different pieces of
    evidence and must not cancel or alias each other.
    """
    vocabulary = build_vocabulary()
    index_of = {code: i for i, code in enumerate(vocabulary)}
    width = len(vocabulary)
    vector = [0.0] * embedding_dimension()

    channels = [
        (present or [], 0),
        (negated or [], width),
        (historical or [], 2 * width),
    ]
    for codes, offset in channels:
        for code in codes:
            position = index_of.get(code)
            if position is None:
                vector[3 * width + _hashed_index(code)] = 1.0
            else:
                vector[offset + position] = 1.0
    return vector
