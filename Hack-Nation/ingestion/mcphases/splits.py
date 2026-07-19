"""Participant-grouped mcPHASES splits.

The one rule this module exists to enforce: **group on participant identity
alone, never on (participant, study_interval).**

20 of the 42 mcPHASES participants took part in *both* the 2022 and 2024 study
intervals (see ``docs/MCPHASES_DATA_AUDIT.md``). Splitting on the pair would put
the same person's 2022 days in training and their 2024 days in test. Their
physiology, device and self-report habits are shared, so the model would be
scored partly on someone it had already learned — inflating held-out performance
for nearly half the cohort.

The consolidated ``participant_days.jsonl`` already merges both intervals under
one ``participant_id`` and carries no ``study_interval`` field, so grouping on
``participant_id`` keeps the intervals together *structurally* rather than by
convention. :func:`assert_no_participant_leakage` checks it anyway — a future
loader that reintroduces the interval must not silently break the guarantee.

Normalization statistics are the other leakage surface and are handled by
:func:`fit_normalization_stats`, which accepts only training rows.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "assert_no_participant_leakage",
    "build_participant_split",
    "fit_normalization_stats",
    "load_participant_days",
    "write_split_manifest",
]


def load_participant_days(path: str | Path) -> list[dict[str, Any]]:
    """Read the consolidated participant-day table."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(
            f"No participant-day table at {source}. Build it with scripts/consolidate_mcphases.py."
        )
    return [json.loads(line) for line in source.read_text().splitlines() if line.strip()]


def _checksum(participant_ids: list[str], n_rows: int) -> str:
    digest = hashlib.sha256()
    for participant in sorted(participant_ids):
        digest.update(participant.encode())
    digest.update(str(n_rows).encode())
    return digest.hexdigest()


def build_participant_split(
    rows: list[dict[str, Any]],
    *,
    seed: int = 42,
    validation_fraction: float = 0.2,
    test_fraction: float = 0.2,
) -> dict[str, Any]:
    """Deterministic participant-level split.

    Args:
        rows: Participant-day records.
        seed: RNG seed, recorded in the manifest.
        validation_fraction: Share of PARTICIPANTS (not days) for validation.
        test_fraction: Share of participants held out for final evaluation.

    Returns:
        A manifest dict.

    Raises:
        ValueError: If the cohort is too small to fill every split.
    """
    participants = sorted({str(row["participant_id"]) for row in rows})
    n_total = len(participants)

    n_test = max(int(round(n_total * test_fraction)), 1)
    n_validation = max(int(round(n_total * validation_fraction)), 1)
    if n_test + n_validation >= n_total:
        raise ValueError(
            f"{n_total} participants cannot fill validation={n_validation} and "
            f"test={n_test} while leaving any for training."
        )

    rng = np.random.default_rng(seed)
    shuffled = [str(item) for item in rng.permutation(participants)]
    test_ids = sorted(shuffled[:n_test])
    validation_ids = sorted(shuffled[n_test : n_test + n_validation])
    train_ids = sorted(shuffled[n_test + n_validation :])

    days = {participant: 0 for participant in participants}
    for row in rows:
        days[str(row["participant_id"])] += 1

    return {
        "seed": seed,
        "grouping_key": "participant_id",
        "train_ids": train_ids,
        "validation_ids": validation_ids,
        "test_ids": test_ids,
        "dataset_checksum": _checksum(participants, len(rows)),
        "created_at": None,
        "n_participants": n_total,
        "n_days": len(rows),
        "days_by_split": {
            "train": sum(days[p] for p in train_ids),
            "validation": sum(days[p] for p in validation_ids),
            "test": sum(days[p] for p in test_ids),
        },
        "caveats": [
            "Grouping is on participant_id ALONE. 20 of 42 participants appear in both "
            "the 2022 and 2024 study intervals; splitting on (participant, interval) "
            "would leak the same person across splits.",
            "42 participants is a small cohort. A ~8-participant test split yields wide "
            "intervals; report per-participant results, not only pooled day-level means.",
            "Normalization statistics must be fitted on training participants only — see "
            "fit_normalization_stats().",
            "The test split is evaluated once. Selecting a model against it would make the "
            "reported number an in-sample estimate.",
        ],
    }


def assert_no_participant_leakage(manifest: dict[str, Any]) -> None:
    """Raise if any participant appears in more than one split.

    Raises:
        ValueError: On any overlap between splits.
    """
    splits = {
        "train": set(manifest["train_ids"]),
        "validation": set(manifest["validation_ids"]),
        "test": set(manifest["test_ids"]),
    }
    names = list(splits)
    for index, first in enumerate(names):
        for second in names[index + 1 :]:
            overlap = splits[first] & splits[second]
            if overlap:
                raise ValueError(
                    f"Participant leakage: {sorted(overlap)} appear in both '{first}' and "
                    f"'{second}'. Every study interval for one person must stay in one split."
                )


def fit_normalization_stats(
    rows: list[dict[str, Any]], train_ids: list[str]
) -> dict[str, dict[str, float]]:
    """Per-channel mean and std computed on TRAINING participants only.

    Statistics are taken over genuinely observed values, using each row's
    ``is_observed`` mask. Including imputed entries would centre the scaler on
    the filler rather than the measurement — which matters most for PdG, absent
    on two-thirds of days.

    Args:
        rows: All participant-day records.
        train_ids: Training participant identifiers.

    Returns:
        ``{channel: {"mean": float, "std": float, "n_observed": int}}``.
    """
    allowed = set(train_ids)
    collected: dict[str, list[float]] = {}

    for row in rows:
        if str(row["participant_id"]) not in allowed:
            continue
        observed = row.get("is_observed") or {}
        for channel, value in (row.get("values") or {}).items():
            if value is None or not observed.get(channel, value is not None):
                continue
            collected.setdefault(channel, []).append(float(value))

    stats: dict[str, dict[str, float]] = {}
    for channel, values in collected.items():
        array = np.asarray(values, dtype=float)
        std = float(array.std())
        stats[channel] = {
            "mean": float(array.mean()),
            # A zero-variance channel would divide to inf; 1.0 leaves it centred.
            "std": std if std > 1e-9 else 1.0,
            "n_observed": int(array.size),
        }
    return stats


def write_split_manifest(manifest: dict[str, Any], path: str | Path, *, created_at: str) -> Path:
    """Persist a split manifest."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({**manifest, "created_at": created_at}, indent=2) + "\n")
    return destination
