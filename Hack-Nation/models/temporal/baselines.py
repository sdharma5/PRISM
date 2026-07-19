"""The temporal baseline ladder.

Before any sequence model may claim value, it has to beat these. On a
42-participant cohort that is a genuinely open question: the incumbent
echo-state model (fixed random recurrent features + trained linear heads) is
already simple, and the baselines below are simpler still.

The ladder, weakest first:

* :class:`GlobalMeanBaseline` -- the training-set mean. Anything that cannot beat
  this has learned nothing at all.
* :class:`LocfBaseline` -- last observation carried forward. Deceptively strong
  for slowly-varying physiology, and the one most sequence models actually have
  to beat rather than the global mean.
* :class:`ParticipantHistoryMeanBaseline` -- each participant's own running mean
  over their observed history. Captures inter-individual offset, which is large
  for hormones, without modelling dynamics at all.
* :class:`RidgeWindowBaseline` -- ridge on a flattened lookback window. The first
  model that can express dynamics.
* :class:`MajorityPhaseBaseline` / :class:`LogisticPhaseBaseline` -- the cycle-phase
  counterparts.

Two rules hold throughout and are the reason this module exists rather than a
few inline loops:

**Observed-only.** Every fit and every metric uses the ``is_observed`` mask.
PdG is present on ~33% of days, so a metric computed over imputed days would
mostly measure the imputation.

**Causal.** A prediction for day *t* may use only days < *t* (plus same-day
non-target channels). LOCF and running means are the easiest place to leak the
future by one index, so the window construction is shared rather than
reimplemented per baseline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "GlobalMeanBaseline",
    "LocfBaseline",
    "LogisticPhaseBaseline",
    "MajorityPhaseBaseline",
    "ParticipantHistoryMeanBaseline",
    "RidgeWindowBaseline",
    "build_sequences",
]

_EPS = 1e-9


def build_sequences(
    rows: list[dict[str, Any]],
    participant_ids: list[str],
    *,
    channels: tuple[str, ...],
    lookback: int = 14,
) -> dict[str, Any]:
    """Assemble per-participant day-ordered arrays and causal windows.

    Args:
        rows: Participant-day records.
        participant_ids: Participants to include (one split).
        channels: Value channels to extract, in a fixed order.
        lookback: Days of history available to a windowed model.

    Returns:
        A dict with ``values`` ``(n_days, n_channels)``, ``observed`` (same
        shape), ``participant`` ``(n_days,)``, ``phase`` ``(n_days,)`` and
        ``window`` ``(n_days, lookback, n_channels * 2)`` where the trailing
        channels are the observation mask.
    """
    allowed = set(participant_ids)
    by_participant: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        pid = str(row["participant_id"])
        if pid in allowed:
            by_participant.setdefault(pid, []).append(row)

    values_all, observed_all, participant_all, phase_all, windows_all = [], [], [], [], []

    for pid in sorted(by_participant):
        days = sorted(by_participant[pid], key=lambda r: r.get("study_day", 0))
        values = np.array(
            [[_as_float(d.get("values", {}).get(c)) for c in channels] for d in days], dtype=float
        )
        observed = np.array(
            [
                [
                    bool(
                        (d.get("is_observed") or {}).get(c, d.get("values", {}).get(c) is not None)
                    )
                    for c in channels
                ]
                for d in days
            ],
            dtype=bool,
        )
        values = np.where(observed, values, np.nan)

        # Causal windows: window[t] covers days [t-lookback, t-1]. Day t itself is
        # excluded so a model can never read the value it is predicting.
        filled = np.nan_to_num(values, nan=0.0)
        mask = observed.astype(float)
        padded_values = np.vstack([np.zeros((lookback, len(channels))), filled])
        padded_mask = np.vstack([np.zeros((lookback, len(channels))), mask])
        windows = np.stack(
            [
                np.concatenate(
                    [padded_values[t : t + lookback], padded_mask[t : t + lookback]], axis=1
                )
                for t in range(len(days))
            ]
        )

        values_all.append(values)
        observed_all.append(observed)
        windows_all.append(windows)
        participant_all.extend([pid] * len(days))
        phase_all.extend([_phase(d.get("cycle_phase")) for d in days])

    return {
        "values": np.vstack(values_all),
        "observed": np.vstack(observed_all),
        "window": np.vstack(windows_all),
        "participant": np.array(participant_all),
        "phase": np.array(phase_all),
        "channels": channels,
    }


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _phase(value: Any) -> str:
    return str(value).strip().lower() if value is not None else "unknown"


# -- hormone baselines -----------------------------------------------------


@dataclass
class GlobalMeanBaseline:
    """Predict the training mean of each channel. The floor."""

    name: str = "global_mean"
    means_: np.ndarray | None = None

    def fit(self, data: dict[str, Any]) -> GlobalMeanBaseline:
        values, observed = data["values"], data["observed"]
        self.means_ = np.array(
            [
                values[observed[:, c], c].mean() if observed[:, c].any() else 0.0
                for c in range(values.shape[1])
            ]
        )
        return self

    def predict(self, data: dict[str, Any]) -> np.ndarray:
        assert self.means_ is not None
        return np.tile(self.means_, (len(data["values"]), 1))


@dataclass
class LocfBaseline:
    """Last observation carried forward, per participant.

    The baseline most worth beating: hormones move slowly day to day, so
    yesterday's value is a strong predictor of today's. Falls back to the
    training mean before a participant's first observation -- carrying forward
    from nothing would otherwise silently predict zero.
    """

    name: str = "locf"
    fallback_: np.ndarray | None = None

    def fit(self, data: dict[str, Any]) -> LocfBaseline:
        self.fallback_ = GlobalMeanBaseline().fit(data).means_
        return self

    def predict(self, data: dict[str, Any]) -> np.ndarray:
        assert self.fallback_ is not None
        values, observed, participant = data["values"], data["observed"], data["participant"]
        predictions = np.tile(self.fallback_, (len(values), 1))

        for pid in np.unique(participant):
            index = np.flatnonzero(participant == pid)
            last = np.full(values.shape[1], np.nan)
            for position in index:
                # Predict BEFORE absorbing today's observation: using today's
                # value to predict today is a one-index future leak.
                for channel in range(values.shape[1]):
                    if np.isfinite(last[channel]):
                        predictions[position, channel] = last[channel]
                for channel in range(values.shape[1]):
                    if observed[position, channel]:
                        last[channel] = values[position, channel]
        return predictions


@dataclass
class ParticipantHistoryMeanBaseline:
    """Running mean of each participant's own observed history."""

    name: str = "participant_history_mean"
    fallback_: np.ndarray | None = None

    def fit(self, data: dict[str, Any]) -> ParticipantHistoryMeanBaseline:
        self.fallback_ = GlobalMeanBaseline().fit(data).means_
        return self

    def predict(self, data: dict[str, Any]) -> np.ndarray:
        assert self.fallback_ is not None
        values, observed, participant = data["values"], data["observed"], data["participant"]
        predictions = np.tile(self.fallback_, (len(values), 1))

        for pid in np.unique(participant):
            index = np.flatnonzero(participant == pid)
            total = np.zeros(values.shape[1])
            count = np.zeros(values.shape[1])
            for position in index:
                seen = count > 0
                predictions[position, seen] = total[seen] / count[seen]
                for channel in range(values.shape[1]):
                    if observed[position, channel]:
                        total[channel] += values[position, channel]
                        count[channel] += 1
        return predictions


@dataclass
class RidgeWindowBaseline:
    """Ridge regression on a flattened causal lookback window.

    The first baseline able to express dynamics. Fitted per channel on that
    channel's observed days only, so PdG's sparsity shrinks its training set
    rather than polluting it with imputed targets.
    """

    alpha: float = 1.0
    name: str = "ridge_window"
    coefficients_: dict[int, np.ndarray] = field(default_factory=dict)
    intercepts_: dict[int, float] = field(default_factory=dict)
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    def _design(self, data: dict[str, Any]) -> np.ndarray:
        flat = data["window"].reshape(len(data["window"]), -1)
        if self.mean_ is None:
            self.mean_ = flat.mean(axis=0)
            self.scale_ = np.where(flat.std(axis=0) > _EPS, flat.std(axis=0), 1.0)
        return (flat - self.mean_) / self.scale_

    def fit(self, data: dict[str, Any]) -> RidgeWindowBaseline:
        design = self._design(data)
        values, observed = data["values"], data["observed"]

        for channel in range(values.shape[1]):
            rows = observed[:, channel]
            if rows.sum() < 2:
                continue
            X = design[rows]
            y = values[rows, channel]
            gram = X.T @ X + self.alpha * np.eye(X.shape[1])
            self.coefficients_[channel] = np.linalg.solve(gram, X.T @ (y - y.mean()))
            self.intercepts_[channel] = float(y.mean())
        return self

    def predict(self, data: dict[str, Any]) -> np.ndarray:
        design = self._design(data)
        predictions = np.zeros((len(design), data["values"].shape[1]))
        for channel in range(predictions.shape[1]):
            if channel in self.coefficients_:
                predictions[:, channel] = (
                    design @ self.coefficients_[channel] + self.intercepts_[channel]
                )
        return predictions


# -- cycle-phase baselines -------------------------------------------------


@dataclass
class MajorityPhaseBaseline:
    """Always predict the most common training phase."""

    name: str = "majority_phase"
    majority_: str = "unknown"
    classes_: list[str] = field(default_factory=list)

    def fit(self, data: dict[str, Any]) -> MajorityPhaseBaseline:
        phases = [p for p in data["phase"] if p != "unknown"]
        self.classes_ = sorted(set(phases))
        if phases:
            unique, counts = np.unique(phases, return_counts=True)
            self.majority_ = str(unique[int(np.argmax(counts))])
        return self

    def predict(self, data: dict[str, Any]) -> np.ndarray:
        return np.array([self.majority_] * len(data["phase"]))


@dataclass
class LogisticPhaseBaseline:
    """Multinomial logistic regression on the causal window."""

    name: str = "logistic_phase"
    model_: Any = None
    classes_: list[str] = field(default_factory=list)
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    def _design(self, data: dict[str, Any]) -> np.ndarray:
        flat = data["window"].reshape(len(data["window"]), -1)
        if self.mean_ is None:
            self.mean_ = flat.mean(axis=0)
            self.scale_ = np.where(flat.std(axis=0) > _EPS, flat.std(axis=0), 1.0)
        return (flat - self.mean_) / self.scale_

    def fit(self, data: dict[str, Any]) -> LogisticPhaseBaseline:
        from sklearn.linear_model import LogisticRegression  # noqa: PLC0415

        design = self._design(data)
        labelled = data["phase"] != "unknown"
        if labelled.sum() < 2:
            return self
        self.model_ = LogisticRegression(max_iter=1000, class_weight="balanced")
        self.model_.fit(design[labelled], data["phase"][labelled])
        self.classes_ = list(self.model_.classes_)
        return self

    def predict(self, data: dict[str, Any]) -> np.ndarray:
        if self.model_ is None:
            return np.array(["unknown"] * len(data["phase"]))
        return self.model_.predict(self._design(data))

    def predict_proba(self, data: dict[str, Any]) -> np.ndarray | None:
        if self.model_ is None:
            return None
        return self.model_.predict_proba(self._design(data))
