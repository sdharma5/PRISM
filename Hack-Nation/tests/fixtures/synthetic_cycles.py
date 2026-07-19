"""Synthetic longitudinal cycle data with known phase labels.

mcPHASES and comparable longitudinal hormone datasets are access-restricted and
are never committed here, so the temporal model needs a generator that reproduces
the *structure* those datasets have — otherwise the code would only ever be
tested on data that cannot expose its real failure modes.

The physiology encoded here is deliberately textbook, because the point is to
have known labels rather than to simulate a person:

* **E3G** (urinary estrone-3-glucuronide, an estradiol metabolite) rises through
  the follicular phase and peaks ~1 day before ovulation, then dips and shows a
  smaller luteal rise.
* **LH** shows a sharp, short surge peaking ~1 day before ovulation. Its
  narrowness is what makes peak-timing error a meaningful metric.
* **PdG** (pregnanediol glucuronide, a progesterone metabolite) is flat until
  ovulation and rises through the luteal phase — it is the confirmatory signal
  that ovulation occurred.
* Wearable channels (resting heart rate, wrist temperature, HRV) shift in the
  luteal phase, which is the physiological basis of wearable cycle tracking.

Missingness is **MNAR** on purpose. Real hormone data is missing in a way that
depends on the value and the day: people test more around expected ovulation and
skip on weekends and during menses. A generator with MCAR missingness would make
imputation look far better than it is, and would let a silent zero-fill bug pass
unnoticed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from schemas.temporal import CyclePhase, ParticipantDay

#: Hormone channels the reconstruction head predicts.
HORMONE_CHANNELS: tuple[str, ...] = ("lh", "e3g", "pdg")

#: Wearable and CGM channels.
WEARABLE_CHANNELS: tuple[str, ...] = ("resting_heart_rate", "wrist_temperature", "hrv_rmssd")
CGM_CHANNELS: tuple[str, ...] = ("mean_glucose",)

#: Daily self-reported symptoms, used by the next-day symptom head.
SYMPTOM_CHANNELS: tuple[str, ...] = ("cramps", "bloating", "mood_low", "breast_tenderness")

ALL_CHANNELS: tuple[str, ...] = (*HORMONE_CHANNELS, *WEARABLE_CHANNELS, *CGM_CHANNELS)


@dataclass
class SyntheticCohort:
    """A synthetic longitudinal cohort plus its ground truth."""

    days: list[ParticipantDay]
    participant_ids: list[str]
    ovulation_days: dict[str, list[int]] = field(default_factory=dict)
    channels: tuple[str, ...] = ALL_CHANNELS

    def for_participant(self, participant_id: str) -> list[ParticipantDay]:
        """All days for one participant, in study-day order."""
        return sorted(
            (d for d in self.days if d.participant_id == participant_id),
            key=lambda d: d.study_day,
        )

    def phase_labels(self) -> list[CyclePhase]:
        return [d.cycle_phase for d in self.days]


def _phase_for(cycle_day: int, cycle_length: int, ovulation_day: int) -> CyclePhase:
    """Map a cycle day onto a phase label.

    Peri-ovulatory is a +/-2 day window around ovulation, which is the window in
    which the LH surge and the fertile interval actually sit.
    """
    if cycle_day <= 5:
        return "menstrual"
    if abs(cycle_day - ovulation_day) <= 2:
        return "peri_ovulatory"
    if cycle_day < ovulation_day:
        return "follicular"
    if cycle_day <= cycle_length:
        return "luteal"
    return "unknown"


def _hormone_profile(
    cycle_day: int, cycle_length: int, ovulation_day: int, rng: np.random.Generator
) -> dict[str, float]:
    """Textbook LH / E3G / PdG values for one cycle day, with biological noise."""
    surge_centre = ovulation_day - 1.0
    lh = 4.0 + 38.0 * np.exp(-(((cycle_day - surge_centre) / 1.1) ** 2))

    follicular_rise = 20.0 + 45.0 * np.exp(-(((cycle_day - (ovulation_day - 1.5)) / 4.0) ** 2))
    luteal_bump = 22.0 * np.exp(-(((cycle_day - (ovulation_day + 7.0)) / 4.5) ** 2))
    e3g = follicular_rise + luteal_bump

    luteal_progress = np.clip((cycle_day - ovulation_day) / 7.0, 0.0, None)
    pdg = 1.0 + 9.0 * np.clip(luteal_progress, 0.0, 1.6) * float(cycle_day > ovulation_day)
    if cycle_day > ovulation_day + 11:
        pdg *= float(np.clip(1.0 - (cycle_day - ovulation_day - 11) / 4.0, 0.15, 1.0))

    return {
        "lh": float(max(lh * (1.0 + 0.12 * rng.standard_normal()), 0.1)),
        "e3g": float(max(e3g * (1.0 + 0.15 * rng.standard_normal()), 1.0)),
        "pdg": float(max(pdg * (1.0 + 0.18 * rng.standard_normal()), 0.05)),
    }


def _wearable_profile(
    cycle_day: int, ovulation_day: int, baseline: dict[str, float], rng: np.random.Generator
) -> dict[str, float]:
    """Luteal-phase shifts in resting HR, wrist temperature, HRV and glucose."""
    luteal = float(cycle_day > ovulation_day)
    return {
        "resting_heart_rate": baseline["rhr"] + 2.6 * luteal + 2.0 * rng.standard_normal(),
        "wrist_temperature": baseline["temp"] + 0.32 * luteal + 0.12 * rng.standard_normal(),
        "hrv_rmssd": baseline["hrv"] - 6.0 * luteal + 5.0 * rng.standard_normal(),
        "mean_glucose": baseline["glucose"] + 3.5 * luteal + 4.0 * rng.standard_normal(),
    }


def _symptom_probabilities(
    phase: CyclePhase, cycle_day: int, ovulation_day: int
) -> dict[str, float]:
    """Phase-dependent symptom probabilities."""
    late_luteal = phase == "luteal" and cycle_day >= ovulation_day + 8
    return {
        "cramps": 0.75 if phase == "menstrual" else (0.28 if late_luteal else 0.06),
        "bloating": 0.45 if phase == "menstrual" else (0.40 if late_luteal else 0.10),
        "mood_low": 0.30 if phase == "menstrual" else (0.35 if late_luteal else 0.10),
        "breast_tenderness": 0.40 if late_luteal else (0.10 if phase == "luteal" else 0.05),
    }


def _observation_probability(
    channel: str, phase: CyclePhase, cycle_day: int, ovulation_day: int, weekday: int
) -> float:
    """MNAR observation probability: testing behaviour depends on the state.

    Hormone strips are used most around expected ovulation and least during
    menses; wearables drop out on weekends. Both patterns correlate with the
    hormone value itself, which is precisely what makes the missingness
    non-ignorable and why the model must consume an explicit observation
    indicator rather than an imputed value.
    """
    if channel in HORMONE_CHANNELS:
        near_ovulation = abs(cycle_day - ovulation_day) <= 4
        base = 0.85 if near_ovulation else 0.35
        if phase == "menstrual":
            base *= 0.35
        if channel == "pdg":
            # PdG is typically tested only in the luteal window.
            base *= 1.2 if cycle_day > ovulation_day else 0.4
        return float(np.clip(base, 0.02, 0.95))
    if channel in WEARABLE_CHANNELS:
        return 0.72 if weekday >= 5 else 0.93
    return 0.60  # CGM: worn intermittently.


def generate_participant(
    participant_id: str,
    *,
    n_days: int = 90,
    rng: np.random.Generator,
    source_dataset: str = "synthetic_cycles",
) -> tuple[list[ParticipantDay], list[int]]:
    """Generate one participant's day series and their ovulation study-days."""
    baseline = {
        "rhr": float(rng.normal(62, 5)),
        "temp": float(rng.normal(33.4, 0.4)),
        "hrv": float(rng.normal(48, 9)),
        "glucose": float(rng.normal(95, 7)),
    }
    cycle_length = int(np.clip(rng.normal(28.5, 2.2), 24, 35))
    luteal_length = int(np.clip(rng.normal(13.0, 1.2), 10, 16))
    ovulation_day = cycle_length - luteal_length

    days: list[ParticipantDay] = []
    ovulation_study_days: list[int] = []
    last_seen: dict[str, int | None] = dict.fromkeys(ALL_CHANNELS)
    cycle_day = int(rng.integers(1, cycle_length + 1))
    cycle_start = -cycle_day + 1

    for study_day in range(n_days):
        if cycle_day > cycle_length:
            cycle_day = 1
            cycle_start = study_day
            cycle_length = int(np.clip(rng.normal(28.5, 2.2), 24, 35))
            luteal_length = int(np.clip(rng.normal(13.0, 1.2), 10, 16))
            ovulation_day = cycle_length - luteal_length
        if cycle_day == ovulation_day and cycle_start >= 0:
            ovulation_study_days.append(study_day)

        phase = _phase_for(cycle_day, cycle_length, ovulation_day)
        truth = {
            **_hormone_profile(cycle_day, cycle_length, ovulation_day, rng),
            **_wearable_profile(cycle_day, ovulation_day, baseline, rng),
        }

        values: dict[str, float | None] = {}
        observed: dict[str, bool] = {}
        gaps: dict[str, float] = {}
        for channel in ALL_CHANNELS:
            probability = _observation_probability(
                channel, phase, cycle_day, ovulation_day, study_day % 7
            )
            is_observed = bool(rng.random() < probability)
            observed[channel] = is_observed
            # Missing is None, never 0.0. A zero LH is a real, meaningful value.
            values[channel] = float(truth[channel]) if is_observed else None
            if is_observed:
                last_seen[channel] = study_day
                gaps[channel] = 0.0
            else:
                previous = last_seen[channel]
                gaps[channel] = (
                    float(study_day - previous) if previous is not None else float(study_day + 1)
                )

        probabilities = _symptom_probabilities(phase, cycle_day, ovulation_day)
        symptoms = {name: bool(rng.random() < p) for name, p in probabilities.items()}

        days.append(
            ParticipantDay(
                participant_id=participant_id,
                study_day=study_day,
                cycle_day=cycle_day,
                cycle_phase=phase,
                values=values,
                is_observed=observed,
                time_since_last_observed=gaps,
                daily_symptoms=symptoms,
                source_dataset=source_dataset,
            )
        )
        cycle_day += 1

    return days, ovulation_study_days


def generate_cohort(
    *,
    n_participants: int = 12,
    n_days: int = 90,
    seed: int = 0,
    source_dataset: str = "synthetic_cycles",
) -> SyntheticCohort:
    """Generate a synthetic longitudinal cohort.

    Args:
        n_participants: Number of participants.
        n_days: Days per participant.
        seed: RNG seed for reproducibility.
        source_dataset: Recorded on every day for provenance.

    Returns:
        A :class:`SyntheticCohort`.
    """
    rng = np.random.default_rng(seed)
    all_days: list[ParticipantDay] = []
    ovulations: dict[str, list[int]] = {}
    ids: list[str] = []
    for index in range(n_participants):
        participant_id = f"SYN{index:03d}"
        ids.append(participant_id)
        days, ovulation_days = generate_participant(
            participant_id, n_days=n_days, rng=rng, source_dataset=source_dataset
        )
        all_days.extend(days)
        ovulations[participant_id] = ovulation_days
    return SyntheticCohort(days=all_days, participant_ids=ids, ovulation_days=ovulations)


def true_hormone_series(cohort: SyntheticCohort, channel: str = "lh") -> dict[str, np.ndarray]:
    """Observed values per participant with NaN where unobserved.

    NaN rather than 0 so that any downstream code that forgets to mask will
    produce a loud NaN instead of a plausible-looking wrong number.
    """
    out: dict[str, np.ndarray] = {}
    for participant_id in cohort.participant_ids:
        series = cohort.for_participant(participant_id)
        out[participant_id] = np.array(
            [day.values.get(channel) if day.is_observed.get(channel) else np.nan for day in series],
            dtype=float,
        )
    return out
