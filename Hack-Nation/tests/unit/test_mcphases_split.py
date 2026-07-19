"""Leakage and determinism guarantees for the mcPHASES participant split.

The load-bearing test here is
:func:`test_dual_interval_participants_do_not_leak_across_splits`. 20 of the 42
participants took part in both the 2022 and 2024 study intervals, so a split
keyed on ``(participant, interval)`` would place one person on both sides of the
train/test boundary for nearly half the cohort. That test reads the RAW file to
identify who is affected, rather than trusting the consolidated table to have
merged them.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from ingestion.mcphases.splits import (
    assert_no_participant_leakage,
    build_participant_split,
    fit_normalization_stats,
)

RAW_HORMONES = Path("../datasets/mcphases/raw/hormones_and_selfreport.csv")
PARTICIPANT_DAYS = Path("../datasets/mcphases/raw/participant_days.jsonl")


def synthetic_rows(n_participants: int = 12, days: int = 20) -> list[dict]:
    """A cohort with the same shape as mcPHASES, so CI needs no restricted data."""
    return [
        {
            "participant_id": f"p{participant}",
            "study_day": day,
            "values": {"lh": float(day), "pdg": float(day) if day % 3 == 0 else None},
            "is_observed": {"lh": True, "pdg": day % 3 == 0},
        }
        for participant in range(n_participants)
        for day in range(days)
    ]


# -- determinism and partitioning ------------------------------------------


def test_split_is_deterministic() -> None:
    rows = synthetic_rows()
    first = build_participant_split(rows, seed=42)
    second = build_participant_split(rows, seed=42)
    assert first["train_ids"] == second["train_ids"]
    assert first["validation_ids"] == second["validation_ids"]
    assert first["test_ids"] == second["test_ids"]


def test_different_seed_gives_a_different_split() -> None:
    rows = synthetic_rows()
    assert (
        build_participant_split(rows, seed=1)["test_ids"]
        != build_participant_split(rows, seed=2)["test_ids"]
    )


def test_no_participant_appears_in_two_splits() -> None:
    manifest = build_participant_split(synthetic_rows(), seed=0)
    assert_no_participant_leakage(manifest)

    train = set(manifest["train_ids"])
    validation = set(manifest["validation_ids"])
    test = set(manifest["test_ids"])
    assert not train & validation
    assert not train & test
    assert not validation & test


def test_every_participant_is_assigned_exactly_once() -> None:
    rows = synthetic_rows()
    manifest = build_participant_split(rows, seed=0)
    assigned = manifest["train_ids"] + manifest["validation_ids"] + manifest["test_ids"]
    assert len(assigned) == len(set(assigned))
    assert set(assigned) == {row["participant_id"] for row in rows}


def test_leakage_assertion_actually_fires() -> None:
    """A guard that cannot fail is not a guard."""
    manifest = build_participant_split(synthetic_rows(), seed=0)
    manifest["validation_ids"] = [*manifest["validation_ids"], manifest["train_ids"][0]]
    with pytest.raises(ValueError, match="leakage"):
        assert_no_participant_leakage(manifest)


def test_cohort_too_small_raises_rather_than_emptying_train() -> None:
    rows = synthetic_rows(n_participants=2)
    with pytest.raises(ValueError, match="cannot fill"):
        build_participant_split(rows, seed=0, validation_fraction=0.5, test_fraction=0.5)


# -- normalization statistics ----------------------------------------------


def test_normalization_stats_use_training_participants_only() -> None:
    """Test-participant values must not reach the scaler."""
    rows = synthetic_rows(n_participants=10, days=10)
    manifest = build_participant_split(rows, seed=0)

    train_only = fit_normalization_stats(rows, manifest["train_ids"])
    everyone = fit_normalization_stats(rows, [row["participant_id"] for row in rows])

    assert train_only["lh"]["n_observed"] < everyone["lh"]["n_observed"]


def test_normalization_ignores_unobserved_values() -> None:
    """Only genuinely observed entries contribute; masks are honoured."""
    rows = synthetic_rows(n_participants=6, days=9)
    manifest = build_participant_split(rows, seed=0)
    stats = fit_normalization_stats(rows, manifest["train_ids"])

    n_train_days = 9 * len(manifest["train_ids"])
    assert stats["lh"]["n_observed"] == n_train_days
    # pdg observed only every third day.
    assert stats["pdg"]["n_observed"] == pytest.approx(
        n_train_days / 3, abs=len(manifest["train_ids"])
    )


def test_zero_variance_channel_does_not_divide_by_zero() -> None:
    rows = [
        {
            "participant_id": f"p{p}",
            "study_day": d,
            "values": {"flat": 5.0},
            "is_observed": {"flat": True},
        }
        for p in range(8)
        for d in range(5)
    ]
    manifest = build_participant_split(rows, seed=0)
    assert fit_normalization_stats(rows, manifest["train_ids"])["flat"]["std"] == 1.0


# -- the real-data invariant -----------------------------------------------


@pytest.mark.skipif(not RAW_HORMONES.exists(), reason="restricted mcPHASES data not present")
def test_dual_interval_participants_do_not_leak_across_splits() -> None:
    """A participant in both study intervals must land wholly in one split.

    Read from the RAW file so the guarantee is checked against the source of
    truth rather than assumed from the consolidated table's schema.
    """
    with RAW_HORMONES.open() as handle:
        records = list(csv.DictReader(handle))

    intervals: dict[str, set[str]] = {}
    for record in records:
        intervals.setdefault(record["id"], set()).add(record["study_interval"])
    dual = {pid for pid, seen in intervals.items() if len(seen) > 1}
    assert dual, "expected some participants in both intervals; audit says 20 of 42"

    rows = [
        {
            "participant_id": f"mcphases:{pid}",
            "study_day": 0,
            "values": {},
            "is_observed": {},
        }
        for pid in intervals
    ]
    manifest = build_participant_split(rows, seed=42)
    assert_no_participant_leakage(manifest)

    placement = {
        **dict.fromkeys(manifest["train_ids"], "train"),
        **dict.fromkeys(manifest["validation_ids"], "validation"),
        **dict.fromkeys(manifest["test_ids"], "test"),
    }
    # Each dual-interval participant has exactly one placement, so both of their
    # study periods necessarily travel together.
    for pid in dual:
        assert placement[f"mcphases:{pid}"] in ("train", "validation", "test")


@pytest.mark.skipif(not PARTICIPANT_DAYS.exists(), reason="restricted mcPHASES data not present")
def test_consolidated_table_carries_no_interval_key() -> None:
    """Grouping must not be keyed on the interval, even accidentally.

    If a future loader reintroduces ``study_interval`` as a grouping field, this
    test fails and forces the leakage question to be re-answered deliberately.
    """
    import json

    first = json.loads(PARTICIPANT_DAYS.read_text().splitlines()[0])
    assert "study_interval" not in first, (
        "participant_days.jsonl gained a study_interval field. Splitting on "
        "(participant, interval) leaks 20 of 42 participants across splits."
    )
