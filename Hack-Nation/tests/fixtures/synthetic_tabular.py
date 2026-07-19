"""Synthetic canonical-coded cohort used by every test and smoke script.

Real cohorts are not committed to this repository, so tests need a generator that
speaks the *same* canonical vocabulary as ``registry/variables.yaml``. The signal
is deliberately modest: a fixture that is trivially separable would hide
calibration and leakage bugs instead of exposing them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Continuous canonical codes: (code, healthy_mean, sd, effect_of_latent_risk).
_CONTINUOUS: tuple[tuple[str, float, float, float], ...] = (
    ("age", 30.0, 6.0, -0.5),
    ("weight", 65.0, 12.0, 6.0),
    ("height", 163.0, 7.0, 0.0),
    ("waist_circumference", 82.0, 10.0, 6.0),
    ("hip_circumference", 98.0, 9.0, 3.0),
    ("cycle_length", 29.0, 4.0, 7.0),
    ("menstrual_frequency_per_year", 12.0, 1.5, -2.5),
    ("luteinizing_hormone", 6.0, 2.5, 3.0),
    ("follicle_stimulating_hormone", 6.5, 2.0, -0.8),
    ("progesterone", 6.0, 3.0, -1.5),
    ("estradiol", 90.0, 30.0, 5.0),
    ("fasting_glucose", 88.0, 10.0, 7.0),
    ("fasting_insulin", 8.0, 4.0, 4.5),
    ("systolic_blood_pressure", 116.0, 11.0, 5.0),
    ("diastolic_blood_pressure", 74.0, 8.0, 3.0),
    ("hdl_cholesterol", 55.0, 12.0, -6.0),
    ("triglycerides", 100.0, 35.0, 20.0),
    ("total_testosterone", 35.0, 12.0, 14.0),
    ("free_testosterone", 2.0, 0.8, 0.9),
    ("dheas", 200.0, 70.0, 45.0),
    ("shbg", 60.0, 20.0, -14.0),
    ("anti_mullerian_hormone", 3.5, 1.5, 2.2),
    ("ferriman_gallwey_score", 4.0, 3.0, 4.0),
    ("follicle_count_left", 8.0, 4.0, 6.0),
    ("follicle_count_right", 8.0, 4.0, 6.0),
    ("ovary_volume_ml", 7.0, 2.5, 3.0),
)

#: Binary canonical codes: (code, base_rate, latent-risk log-odds shift).
_BINARY: tuple[tuple[str, float, float], ...] = (
    ("cycle_irregularity", 0.20, 1.6),
    ("amenorrhea", 0.08, 1.2),
    ("infertility_history", 0.10, 1.0),
    ("hirsutism", 0.15, 1.5),
    ("acne", 0.25, 0.9),
    ("androgenic_alopecia", 0.10, 0.9),
    ("skin_darkening", 0.10, 0.8),
    ("hair_growth_face", 0.15, 1.3),
    ("weight_gain", 0.25, 1.1),
    ("fatigue", 0.30, 0.6),
    ("mood_change", 0.30, 0.5),
    ("pelvic_pain", 0.20, 0.4),
    ("family_history_pcos", 0.12, 0.7),
    ("family_history_diabetes", 0.20, 0.4),
)

#: Codes deliberately left fully unmeasured, to exercise not_collected handling.
_NEVER_COLLECTED: tuple[str, ...] = ("follicle_number_per_ovary",)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def make_synthetic_cohort(
    n: int = 120,
    seed: int = 0,
    missing_rate: float = 0.2,
) -> pd.DataFrame:
    """Return a synthetic cohort keyed by ``patient_id`` with a ``pcos_binary`` label.

    Columns are canonical variable codes. Missing values are real ``NaN`` — the
    generator never encodes "unknown" as zero, because downstream code must be
    able to tell the two apart.
    """
    if n <= 0:
        raise ValueError("n must be positive.")
    if not 0.0 <= missing_rate < 1.0:
        raise ValueError("missing_rate must lie in [0, 1).")

    rng = np.random.default_rng(seed)

    # One latent "risk" axis drives both the label and the observable variables,
    # which is what makes the signal real but modest rather than a giveaway.
    latent = rng.normal(size=n)

    data: dict[str, np.ndarray] = {}
    for code, mean, sd, effect in _CONTINUOUS:
        values = mean + effect * latent + rng.normal(scale=sd, size=n)
        data[code] = np.clip(values, 0.01, None)

    for code, base_rate, shift in _BINARY:
        logits = np.log(base_rate / (1 - base_rate)) + shift * latent
        data[code] = (rng.uniform(size=n) < _sigmoid(logits)).astype(float)

    for code in _NEVER_COLLECTED:
        data[code] = np.full(n, np.nan)

    df = pd.DataFrame(data)
    df.insert(0, "patient_id", [f"SYN{i:04d}" for i in range(n)])

    # Label: latent risk plus substantial noise, so AUROC lands well below 1.0.
    label_logits = 1.7 * latent + rng.normal(scale=1.0, size=n) - 0.25
    df["pcos_binary"] = (rng.uniform(size=n) < _sigmoid(label_logits)).astype(int)

    if missing_rate > 0:
        feature_cols = [c for c in df.columns if c not in {"patient_id", "pcos_binary"}]
        for col in feature_cols:
            if col in _NEVER_COLLECTED:
                continue
            mask = rng.uniform(size=n) < missing_rate
            df.loc[mask, col] = np.nan

    return df


def make_synthetic_cohort_with_groups(
    n: int = 120,
    seed: int = 0,
    missing_rate: float = 0.2,
    n_sites: int = 3,
) -> pd.DataFrame:
    """Same cohort plus a ``site`` column, for grouped splits and subgroup metrics."""
    df = make_synthetic_cohort(n=n, seed=seed, missing_rate=missing_rate)
    rng = np.random.default_rng(seed + 1)
    df["site"] = [f"site_{i}" for i in rng.integers(0, n_sites, size=len(df))]
    return df
