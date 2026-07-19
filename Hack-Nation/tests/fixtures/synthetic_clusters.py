"""Synthetic cohorts with *known* latent groups, for testing the subtype engine.

These are not simulations of PMOS biology. They are geometry with the right
column names: a few well-separated Gaussian blobs in a handful of canonical
variables, plus configurable missingness, so that a test can assert "the
clustering recovered the groups I planted" and "the abstention rules fired on the
participant I planted between two groups".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_FEATURES: tuple[str, ...] = (
    "bmi",
    "waist_circumference",
    "fasting_insulin",
    "homa_ir",
    "total_testosterone",
    "shbg",
    "luteinizing_hormone",
    "anti_mullerian_hormone",
)

#: Group centres in z-units, chosen to echo the named research profiles without
#: claiming to reproduce them: a metabolic-leaning blob, an androgen-leaning blob,
#: and an LH/AMH-leaning blob.
GROUP_CENTERS: dict[str, dict[str, float]] = {
    "metabolic_like": {
        "bmi": 2.0,
        "waist_circumference": 2.0,
        "fasting_insulin": 2.0,
        "homa_ir": 2.0,
        "total_testosterone": 0.0,
        "shbg": -1.0,
        "luteinizing_hormone": 0.0,
        "anti_mullerian_hormone": 0.0,
    },
    "androgen_like": {
        "bmi": 0.0,
        "waist_circumference": 0.0,
        "fasting_insulin": 0.0,
        "homa_ir": 0.0,
        "total_testosterone": 2.5,
        "shbg": -2.0,
        "luteinizing_hormone": 0.5,
        "anti_mullerian_hormone": 0.0,
    },
    "lh_amh_like": {
        "bmi": -1.5,
        "waist_circumference": -1.5,
        "fasting_insulin": -1.0,
        "homa_ir": -1.0,
        "total_testosterone": 0.0,
        "shbg": 1.0,
        "luteinizing_hormone": 2.5,
        "anti_mullerian_hormone": 2.5,
    },
}


def make_synthetic_cluster_frame(
    n_per_group: int = 40,
    groups: tuple[str, ...] = ("metabolic_like", "androgen_like", "lh_amh_like"),
    noise: float = 0.35,
    missing_rate: float = 0.0,
    features: tuple[str, ...] = DEFAULT_FEATURES,
    seed: int = 0,
) -> tuple[pd.DataFrame, list[str]]:
    """Return ``(frame, true_labels)`` for a cohort with known latent groups.

    The frame is indexed by synthetic participant id and contains only the given
    feature columns, already in z-like units. ``missing_rate`` punches values out
    at random (MCAR), which is the easy case — real missingness is not MCAR, and
    tests using this fixture must not be read as evidence about real missingness.
    """
    rng = np.random.default_rng(seed)
    rows: list[np.ndarray] = []
    labels: list[str] = []
    ids: list[str] = []

    for group in groups:
        center = np.array([GROUP_CENTERS[group].get(f, 0.0) for f in features])
        block = center[None, :] + rng.normal(0.0, noise, size=(n_per_group, len(features)))
        rows.append(block)
        labels.extend([group] * n_per_group)
        ids.extend(f"{group[:3]}_{i:03d}" for i in range(n_per_group))

    X = np.vstack(rows)
    if missing_rate > 0:
        mask = rng.random(X.shape) < missing_rate
        X[mask] = np.nan

    frame = pd.DataFrame(X, columns=list(features), index=ids)
    frame.index.name = "patient_id"
    return frame, labels


def make_borderline_participant(features: tuple[str, ...] = DEFAULT_FEATURES) -> dict[str, float]:
    """A participant sitting exactly between two planted groups.

    Used to check that abstention fires for someone who genuinely cannot be
    placed, rather than only for degenerate inputs.
    """
    a = np.array([GROUP_CENTERS["metabolic_like"].get(f, 0.0) for f in features])
    b = np.array([GROUP_CENTERS["androgen_like"].get(f, 0.0) for f in features])
    midpoint = (a + b) / 2.0
    return dict(zip(features, midpoint.tolist(), strict=True))


def make_far_outlier(
    features: tuple[str, ...] = DEFAULT_FEATURES,
    magnitude: float = 12.0,
) -> dict[str, float]:
    """A participant far from every planted group centre."""
    return dict.fromkeys(features, magnitude)
