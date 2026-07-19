"""Metrics for the dynamic hormonal-state model.

Three output families need three different kinds of scrutiny:

* **Hormones** — MAE and RMSE say how far off the level is, Spearman says whether
  the *shape* of the trajectory is right, peak-timing error says whether the
  surge is placed on the right day. The last one matters most: an LH prediction
  that is well-calibrated in magnitude but two days late is useless for anything
  people use ovulation timing for. Interval coverage checks that the model's
  stated uncertainty is honest rather than decorative.
* **Cycle state** — macro F1 and balanced accuracy rather than accuracy, because
  peri-ovulatory is a ~5-day window and plain accuracy rewards never predicting
  it. Per-participant accuracy is reported as a distribution, because a good
  cohort mean can hide participants the model completely fails on. Calibration
  error is reported because a phase probability that says 0.9 must be right about
  90% of the time to be usable.
* **Symptoms** — AUPRC rather than AUROC, since symptoms are imbalanced and AUROC
  flatters a model on rare events. Brier score for calibration.

:func:`missing_modality_ablation` exists because the headline number of any
multimodal model is measured under a data richness that most real users never
have. The degradation table answers the question that actually matters: what
happens to this person's estimate when they stop wearing the watch?
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats

from schemas.temporal import CyclePhase, ParticipantDay, TemporalStateOutput

CYCLE_CLASSES: tuple[CyclePhase, ...] = ("menstrual", "follicular", "peri_ovulatory", "luteal")
EPS = 1e-12


# --------------------------------------------------------------------------
# hormones
# --------------------------------------------------------------------------


def hormone_metrics(predicted: Sequence[float], truth: Sequence[float]) -> dict[str, float]:
    """MAE, RMSE and Spearman rank correlation over finite pairs."""
    p = np.asarray(predicted, dtype=float)
    t = np.asarray(truth, dtype=float)
    ok = np.isfinite(p) & np.isfinite(t)
    if ok.sum() < 2:
        return {"mae": float("nan"), "rmse": float("nan"), "spearman": float("nan"), "n": 0.0}
    p, t = p[ok], t[ok]
    spearman = float("nan")
    if np.std(p) > EPS and np.std(t) > EPS:
        spearman = float(stats.spearmanr(p, t).statistic)
    return {
        "mae": float(np.mean(np.abs(p - t))),
        "rmse": float(np.sqrt(np.mean((p - t) ** 2))),
        "spearman": spearman,
        "n": float(ok.sum()),
    }


def peak_timing_error(
    predicted: Sequence[float], truth: Sequence[float], *, window: int | None = None
) -> float:
    """Absolute difference in days between predicted and true series maxima.

    The single most decision-relevant hormone metric: the LH surge is a ~2 day
    event, so an error of 3 days means the surge was effectively missed even if
    the magnitude was right.

    Args:
        predicted: Predicted series in day order.
        truth: True series in day order.
        window: Optional restriction to the last ``window`` days.

    Returns:
        Absolute day offset, or NaN when either series has no finite values.
    """
    p = np.asarray(predicted, dtype=float)
    t = np.asarray(truth, dtype=float)
    if window is not None:
        p, t = p[-window:], t[-window:]
    if not np.isfinite(p).any() or not np.isfinite(t).any():
        return float("nan")
    return float(abs(int(np.nanargmax(p)) - int(np.nanargmax(t))))


def peak_timing_errors(
    predicted_by_participant: dict[str, Sequence[float]],
    truth_by_participant: dict[str, Sequence[float]],
) -> dict[str, float]:
    """Peak-timing error aggregated across participants."""
    errors = [
        peak_timing_error(predicted_by_participant[pid], truth_by_participant[pid])
        for pid in predicted_by_participant
        if pid in truth_by_participant
    ]
    finite = [e for e in errors if np.isfinite(e)]
    if not finite:
        return {"peak_timing_mae_days": float("nan"), "peak_within_1_day": float("nan")}
    array = np.asarray(finite, dtype=float)
    return {
        "peak_timing_mae_days": float(array.mean()),
        "peak_timing_median_days": float(np.median(array)),
        "peak_within_1_day": float((array <= 1.0).mean()),
        "peak_within_2_days": float((array <= 2.0).mean()),
    }


def interval_coverage(
    predicted: Sequence[float],
    truth: Sequence[float],
    sigma: Sequence[float] | float,
    *,
    z: float = 1.96,
) -> dict[str, float]:
    """Fraction of true values inside the predicted interval.

    A model claiming 95% intervals that contain the truth 60% of the time is
    overconfident, and that is invisible in MAE. Coverage is the check.
    """
    p = np.asarray(predicted, dtype=float)
    t = np.asarray(truth, dtype=float)
    # Broadcast a scalar sigma to one value per point. Going through asarray
    # first means a plain float, a numpy scalar and a 0-d array all behave the
    # same; the previous ``np.isscalar`` test let a 0-d array through as a
    # sequence and then failed on ``s[ok]``.
    s = np.asarray(sigma, dtype=float)
    if s.ndim == 0:
        s = np.full_like(p, float(s.item()))
    ok = np.isfinite(p) & np.isfinite(t) & np.isfinite(s) & (s > 0)
    if not ok.any():
        return {"coverage": float("nan"), "nominal": float(2 * stats.norm.cdf(z) - 1), "n": 0.0}
    inside = np.abs(t[ok] - p[ok]) <= z * s[ok]
    return {
        "coverage": float(inside.mean()),
        "nominal": float(2 * stats.norm.cdf(z) - 1),
        "mean_interval_width": float(2 * z * s[ok].mean()),
        "n": float(ok.sum()),
    }


# --------------------------------------------------------------------------
# cycle state
# --------------------------------------------------------------------------


def macro_f1(true_labels: Sequence[str], predicted_labels: Sequence[str]) -> float:
    """Unweighted mean F1 across the four phases."""
    scores: list[float] = []
    for phase in CYCLE_CLASSES:
        pairs = list(zip(true_labels, predicted_labels, strict=True))
        tp = sum(1 for t, p in pairs if t == phase and p == phase)
        fp = sum(1 for t, p in pairs if t != phase and p == phase)
        fn = sum(1 for t, p in pairs if t == phase and p != phase)
        if tp + fp + fn == 0:
            continue
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(np.mean(scores)) if scores else float("nan")


def balanced_accuracy(true_labels: Sequence[str], predicted_labels: Sequence[str]) -> float:
    """Mean per-class recall: immune to the phase-duration imbalance."""
    recalls: list[float] = []
    for phase in CYCLE_CLASSES:
        support = [p for t, p in zip(true_labels, predicted_labels, strict=True) if t == phase]
        if not support:
            continue
        recalls.append(float(np.mean([p == phase for p in support])))
    return float(np.mean(recalls)) if recalls else float("nan")


def per_participant_accuracy(
    participant_ids: Sequence[str],
    true_labels: Sequence[str],
    predicted_labels: Sequence[str],
) -> dict[str, float]:
    """Accuracy per participant, plus the distribution across participants.

    The minimum matters: a cohort mean of 0.8 built from one participant at 0.2
    and the rest at 0.9 describes a model that fails specific people.
    """
    per: dict[str, list[bool]] = {}
    for pid, t, p in zip(participant_ids, true_labels, predicted_labels, strict=True):
        per.setdefault(pid, []).append(t == p)
    accuracies = {pid: float(np.mean(v)) for pid, v in per.items()}
    values = np.asarray(list(accuracies.values()), dtype=float)
    summary = {
        "mean_participant_accuracy": float(values.mean()) if values.size else float("nan"),
        "min_participant_accuracy": float(values.min()) if values.size else float("nan"),
        "median_participant_accuracy": float(np.median(values)) if values.size else float("nan"),
        "n_participants": float(values.size),
    }
    return {**{f"participant::{k}": v for k, v in accuracies.items()}, **summary}


def expected_calibration_error(
    probabilities: np.ndarray, true_indices: Sequence[int], *, n_bins: int = 10
) -> float:
    """Confidence-binned gap between predicted confidence and observed accuracy."""
    probs = np.atleast_2d(np.asarray(probabilities, dtype=float))
    y = np.asarray(true_indices, dtype=int)
    if probs.shape[0] == 0:
        return float("nan")
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == y
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    error = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        in_bin = (confidence > lo) & (confidence <= hi)
        if not in_bin.any():
            continue
        error += float(in_bin.mean()) * abs(
            float(correct[in_bin].mean()) - float(confidence[in_bin].mean())
        )
    return float(error)


def cycle_state_metrics(
    outputs: Sequence[TemporalStateOutput], true_labels: Sequence[CyclePhase]
) -> dict[str, float]:
    """Full cycle-state metric bundle from model outputs and true phases."""
    predicted = [o.predicted_phase() for o in outputs]
    probabilities = np.array(
        [[o.cycle_phase_probabilities.get(c, 0.0) for c in CYCLE_CLASSES] for o in outputs]
    )
    known = [i for i, label in enumerate(true_labels) if label in CYCLE_CLASSES]
    if not known:
        return {"macro_f1": float("nan"), "balanced_accuracy": float("nan")}
    t = [true_labels[i] for i in known]
    p = [predicted[i] for i in known]
    metrics = {
        "accuracy": float(np.mean([a == b for a, b in zip(t, p, strict=True)])),
        "macro_f1": macro_f1(t, p),
        "balanced_accuracy": balanced_accuracy(t, p),
        "calibration_error": expected_calibration_error(
            probabilities[known], [CYCLE_CLASSES.index(label) for label in t]
        ),
    }
    metrics.update(
        {
            k: v
            for k, v in per_participant_accuracy(
                [outputs[i].patient_id for i in known], t, p
            ).items()
            if not k.startswith("participant::")
        }
    )
    return metrics


# --------------------------------------------------------------------------
# symptoms
# --------------------------------------------------------------------------


def average_precision(probabilities: Sequence[float], labels: Sequence[bool]) -> float:
    """Area under the precision-recall curve, computed by step interpolation.

    AUPRC rather than AUROC because symptom prevalence is low; AUROC would report
    a flattering number for a model that is useless at the operating point where
    a prediction would actually be acted on.
    """
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(labels, dtype=bool)
    if y.sum() == 0 or y.size == 0:
        return float("nan")
    order = np.argsort(-p)
    y = y[order]
    tp = np.cumsum(y)
    precision = tp / np.arange(1, y.size + 1)
    recall = tp / y.sum()
    return float(np.sum(np.diff(np.concatenate([[0.0], recall])) * precision))


def brier_score(probabilities: Sequence[float], labels: Sequence[bool]) -> float:
    """Mean squared error of the probability against the binary outcome."""
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(labels, dtype=float)
    if p.size == 0:
        return float("nan")
    return float(np.mean((p - y) ** 2))


def binary_f1(
    probabilities: Sequence[float], labels: Sequence[bool], *, threshold: float = 0.5
) -> float:
    """F1 at a fixed threshold."""
    pred = np.asarray(probabilities, dtype=float) >= threshold
    y = np.asarray(labels, dtype=bool)
    tp = float((pred & y).sum())
    fp = float((pred & ~y).sum())
    fn = float((~pred & y).sum())
    if tp + fp + fn == 0:
        return float("nan")
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0


def symptom_metrics(
    outputs: Sequence[TemporalStateOutput], next_day_symptoms: Sequence[dict[str, bool]]
) -> dict[str, float]:
    """AUPRC, F1 and Brier per symptom, plus macro averages."""
    names = sorted({k for d in next_day_symptoms for k in d})
    metrics: dict[str, float] = {}
    auprcs: list[float] = []
    briers: list[float] = []
    for name in names:
        rows = [i for i, d in enumerate(next_day_symptoms) if name in d]
        if not rows:
            continue
        probabilities = [outputs[i].symptom_probabilities.get(name, 0.0) for i in rows]
        labels = [next_day_symptoms[i][name] for i in rows]
        auprc = average_precision(probabilities, labels)
        brier = brier_score(probabilities, labels)
        metrics[f"{name}_auprc"] = auprc
        metrics[f"{name}_f1"] = binary_f1(probabilities, labels)
        metrics[f"{name}_brier"] = brier
        if np.isfinite(auprc):
            auprcs.append(auprc)
        if np.isfinite(brier):
            briers.append(brier)
    metrics["macro_auprc"] = float(np.mean(auprcs)) if auprcs else float("nan")
    metrics["macro_brier"] = float(np.mean(briers)) if briers else float("nan")
    return metrics


# --------------------------------------------------------------------------
# missing-modality ablation
# --------------------------------------------------------------------------

#: Named ablations and the channels each removes.
ABLATIONS: dict[str, tuple[str, ...]] = {
    "full": (),
    "no_wearable": ("resting_heart_rate", "wrist_temperature", "hrv_rmssd"),
    "no_cgm": ("mean_glucose",),
    "no_symptoms": ("__symptoms__",),
    "sparse_hormones": ("__sparse_hormones__",),
}


@dataclass
class AblationRow:
    """One row of the degradation table."""

    condition: str
    metrics: dict[str, float] = field(default_factory=dict)
    degradation: dict[str, float] = field(default_factory=dict)
    mean_input_coverage: float = 0.0


def ablate_days(
    days: Sequence[ParticipantDay],
    condition: str,
    *,
    sparse_keep_fraction: float = 0.2,
    seed: int = 0,
) -> list[ParticipantDay]:
    """Return a copy of ``days`` with one modality removed or thinned.

    Removal sets ``is_observed=False`` and ``value=None``, and *increases*
    ``time_since_last_observed``. It never writes zeros: a removed wearable must
    look missing to the model, not look like a resting heart rate of zero.

    Args:
        days: Source participant-days.
        condition: A key of :data:`ABLATIONS`.
        sparse_keep_fraction: Fraction of hormone observations retained under
            ``"sparse_hormones"``.
        seed: RNG seed for the sparse condition.

    Returns:
        A new list of :class:`ParticipantDay`.
    """
    if condition not in ABLATIONS:
        raise ValueError(f"Unknown ablation '{condition}'. Known: {sorted(ABLATIONS)}")
    rng = np.random.default_rng(seed)
    channels = ABLATIONS[condition]
    out: list[ParticipantDay] = []
    staleness: dict[tuple[str, str], float] = {}

    for day in days:
        copy = day.model_copy(deep=True)
        if condition == "no_symptoms":
            copy.daily_symptoms = {}
        elif condition == "sparse_hormones":
            for channel in ("lh", "e3g", "pdg"):
                if copy.is_observed.get(channel) and rng.random() > sparse_keep_fraction:
                    _drop(copy, channel, staleness)
        else:
            for channel in channels:
                if channel.startswith("__"):
                    continue
                _drop(copy, channel, staleness)
        out.append(copy)
    return out


def _drop(day: ParticipantDay, channel: str, staleness: dict[tuple[str, str], float]) -> None:
    """Mark one channel unobserved on one day, growing its staleness counter."""
    if channel not in day.values and channel not in day.is_observed:
        return
    day.values[channel] = None
    day.is_observed[channel] = False
    key = (day.participant_id, channel)
    staleness[key] = staleness.get(key, 0.0) + 1.0
    day.time_since_last_observed[channel] = staleness[key]


def missing_modality_ablation(
    days: Sequence[ParticipantDay],
    predict_fn: Callable[[list[ParticipantDay]], list[TemporalStateOutput]],
    *,
    conditions: Sequence[str] = tuple(ABLATIONS),
    reference: str = "full",
    seed: int = 0,
) -> list[AblationRow]:
    """Run the model under each missing-modality condition and tabulate the drop.

    Args:
        days: Evaluation participant-days (test participants only).
        predict_fn: Callable mapping days to state outputs — usually a fitted
            model's ``predict``.
        conditions: Which ablations to run.
        reference: The condition other rows are compared against.
        seed: Seed for the sparse-hormone condition.

    Returns:
        One :class:`AblationRow` per condition, ``reference`` first.
    """
    truth_by_key = {(d.participant_id, d.study_day): d for d in days}
    results: dict[str, dict[str, float]] = {}
    coverage: dict[str, float] = {}

    for condition in conditions:
        ablated = ablate_days(days, condition, seed=seed)
        outputs = predict_fn(ablated)
        if not outputs:
            results[condition] = {}
            coverage[condition] = 0.0
            continue

        true_phases: list[CyclePhase] = []
        next_symptoms: list[dict[str, bool]] = []
        for output in outputs:
            study_day = _study_day_of(output)
            source = truth_by_key.get((output.patient_id, study_day))
            true_phases.append(source.cycle_phase if source else "unknown")
            nxt = truth_by_key.get((output.patient_id, study_day + 1))
            next_symptoms.append(dict(nxt.daily_symptoms) if nxt else {})

        metrics = cycle_state_metrics(outputs, true_phases)
        metrics.update(symptom_metrics(outputs, next_symptoms))
        results[condition] = metrics
        coverage[condition] = float(np.mean([o.input_coverage for o in outputs]))

    baseline = results.get(reference, {})
    rows: list[AblationRow] = []
    for condition in conditions:
        metrics = results.get(condition, {})
        degradation = {
            key: float(baseline[key] - value)
            for key, value in metrics.items()
            if key in baseline and np.isfinite(baseline[key]) and np.isfinite(value)
        }
        rows.append(
            AblationRow(
                condition=condition,
                metrics=metrics,
                degradation={} if condition == reference else degradation,
                mean_input_coverage=coverage.get(condition, 0.0),
            )
        )
    rows.sort(key=lambda r: (r.condition != reference, r.condition))
    return rows


def _study_day_of(output: TemporalStateOutput) -> int:
    """Recover the study day from the ``as_of_date`` convention."""
    text = str(output.as_of_date)
    if text.startswith("study_day_"):
        try:
            return int(text.removeprefix("study_day_"))
        except ValueError:
            return -1
    return -1


def format_ablation_table(rows: Sequence[AblationRow], *, metric: str = "balanced_accuracy") -> str:
    """Render the degradation table as plain text for logs and reports."""
    lines = [f"{'condition':<20}{metric:>20}{'degradation':>14}{'coverage':>11}", "-" * 65]
    for row in rows:
        value = row.metrics.get(metric, float("nan"))
        drop = row.degradation.get(metric)
        lines.append(
            f"{row.condition:<20}{value:>20.4f}"
            f"{('—' if drop is None else f'{drop:+.4f}'):>14}"
            f"{row.mean_input_coverage:>11.3f}"
        )
    return "\n".join(lines)


def evaluate_temporal(
    outputs: Sequence[TemporalStateOutput],
    days: Sequence[ParticipantDay],
) -> dict[str, Any]:
    """Cycle-state and symptom metrics for a set of outputs against the truth."""
    truth = {(d.participant_id, d.study_day): d for d in days}
    phases: list[CyclePhase] = []
    next_symptoms: list[dict[str, bool]] = []
    for output in outputs:
        study_day = _study_day_of(output)
        source = truth.get((output.patient_id, study_day))
        phases.append(source.cycle_phase if source else "unknown")
        nxt = truth.get((output.patient_id, study_day + 1))
        next_symptoms.append(dict(nxt.daily_symptoms) if nxt else {})
    return {**cycle_state_metrics(outputs, phases), **symptom_metrics(outputs, next_symptoms)}
