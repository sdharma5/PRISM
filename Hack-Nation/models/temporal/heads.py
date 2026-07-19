"""Prediction heads over the shared state embedding.

Four heads sit on one embedding, and the multi-task structure is the point: a
representation that must simultaneously reconstruct hormones, place the cycle
phase, forecast tomorrow's symptoms and fill in hidden values cannot collapse
into any single one of those shortcuts.

Each head is a ridge-regularised linear (or multinomial-logistic) map fit in
closed form or by a short numpy optimisation. Linear heads are deliberate: the
recurrent encoder already supplies the nonlinearity, cohorts are small, and a
linear head keeps the mapping from state to prediction inspectable — you can read
off which embedding directions drive an LH surge prediction.

Every head reports uncertainty, because a state estimate on a participant who has
logged nothing for ten days must be visibly less certain than one on a densely
logged participant.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from schemas.temporal import CyclePhase

#: Cycle-phase classes in fixed index order. "unknown" is never predicted; it is
#: an input-side label meaning "we were not told", not a state the body is in.
CYCLE_CLASSES: tuple[CyclePhase, ...] = ("menstrual", "follicular", "peri_ovulatory", "luteal")

#: Hormone channels reconstructed by the hormone head.
HORMONE_TARGETS: tuple[str, ...] = ("lh", "e3g", "pdg")

EPS = 1e-9


def _add_bias(x: np.ndarray) -> np.ndarray:
    """Append a bias column."""
    x = np.atleast_2d(np.asarray(x, dtype=float))
    return np.hstack([x, np.ones((x.shape[0], 1))])


def _ridge_fit(
    x: np.ndarray, y: np.ndarray, alpha: float, weights: np.ndarray | None = None
) -> np.ndarray:
    """Closed-form (optionally weighted) ridge solution, bias unpenalised."""
    xb = _add_bias(x)
    y = np.atleast_2d(np.asarray(y, dtype=float))
    if y.shape[0] != xb.shape[0]:
        y = y.T
    if weights is not None:
        w = np.asarray(weights, dtype=float).reshape(-1, 1)
        xb = xb * np.sqrt(w)
        y = y * np.sqrt(w)
    penalty = alpha * np.eye(xb.shape[1])
    penalty[-1, -1] = 0.0
    return np.linalg.solve(xb.T @ xb + penalty, xb.T @ y)


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.clip(e.sum(axis=1, keepdims=True), EPS, None)


@dataclass
class HormoneReconstructionHead:
    """Predicts LH, E3G and PdG from the state embedding.

    Fit on observed targets only, per channel, so a channel that a participant
    rarely tests does not drag the other channels toward its sparse mean. The
    predicted variance is the residual variance of the fit, which gives honest
    (if homoscedastic) intervals.
    """

    targets: tuple[str, ...] = HORMONE_TARGETS
    alpha: float = 1.0
    coefficients: dict[str, np.ndarray] = field(default_factory=dict)
    residual_std: dict[str, float] = field(default_factory=dict)
    target_means: dict[str, float] = field(default_factory=dict)

    def fit(
        self, embeddings: np.ndarray, targets: np.ndarray, mask: np.ndarray | None = None
    ) -> HormoneReconstructionHead:
        """Fit one ridge regression per hormone channel.

        Args:
            embeddings: ``(N, H)`` state embeddings.
            targets: ``(N, C)`` hormone values; entries may be NaN when missing.
            mask: ``(N, C)`` 1 where the target is observed.
        """
        y = np.atleast_2d(np.asarray(targets, dtype=float))
        m = np.asarray(mask, dtype=bool) if mask is not None else np.isfinite(y)
        for index, name in enumerate(self.targets):
            rows = m[:, index] & np.isfinite(y[:, index])
            if rows.sum() < 3:
                self.coefficients[name] = np.zeros((embeddings.shape[1] + 1, 1))
                self.residual_std[name] = float("nan")
                finite = y[rows, index]
                finite = finite[np.isfinite(finite)]
                self.target_means[name] = float(finite.mean()) if finite.size else 0.0
                continue
            beta = _ridge_fit(embeddings[rows], y[rows, index : index + 1], self.alpha)
            self.coefficients[name] = beta
            residuals = y[rows, index : index + 1] - _add_bias(embeddings[rows]) @ beta
            self.residual_std[name] = float(residuals.std())
            self.target_means[name] = float(y[rows, index].mean())
        return self

    def predict(self, embeddings: np.ndarray) -> dict[str, np.ndarray]:
        """Predict each hormone channel."""
        xb = _add_bias(embeddings)
        out: dict[str, np.ndarray] = {}
        for name in self.targets:
            beta = self.coefficients.get(name)
            if beta is None:
                out[name] = np.full(xb.shape[0], self.target_means.get(name, 0.0))
            else:
                out[name] = (xb @ beta).ravel()
        return out

    def predict_with_uncertainty(
        self, embeddings: np.ndarray
    ) -> tuple[dict[str, np.ndarray], dict[str, float]]:
        """Predictions plus the per-channel residual standard deviation."""
        return self.predict(embeddings), dict(self.residual_std)


@dataclass
class CycleStateHead:
    """Multinomial logistic head over menstrual / follicular / peri_ovulatory / luteal.

    Trained with inverse-frequency class weights so the short peri-ovulatory
    window is not optimised away — a model that never predicts ovulation can still
    score ~85% accuracy and be useless.
    """

    classes: tuple[CyclePhase, ...] = CYCLE_CLASSES
    alpha: float = 1.0
    max_iter: int = 300
    learning_rate: float = 0.5
    weights: np.ndarray | None = None
    class_weights: np.ndarray | None = None

    def fit(self, embeddings: np.ndarray, labels: list[CyclePhase] | np.ndarray) -> CycleStateHead:
        """Fit by gradient descent on the class-weighted cross-entropy."""
        x = _add_bias(embeddings)
        y = np.asarray([self.classes.index(label) for label in labels], dtype=int)
        k = len(self.classes)
        counts = np.bincount(y, minlength=k).astype(float)
        self.class_weights = np.where(counts > 0, len(y) / (k * np.clip(counts, 1, None)), 1.0)
        sample_weights = self.class_weights[y]

        w = np.zeros((x.shape[1], k))
        onehot = np.eye(k)[y]
        for _ in range(self.max_iter):
            probs = _softmax(x @ w)
            grad = x.T @ ((probs - onehot) * sample_weights[:, None]) / max(len(y), 1)
            grad[:-1] += self.alpha * w[:-1] / max(len(y), 1)
            w -= self.learning_rate * grad
        self.weights = w
        return self

    def predict_proba(self, embeddings: np.ndarray) -> np.ndarray:
        """Class probabilities ``(N, K)``."""
        if self.weights is None:
            n = np.atleast_2d(np.asarray(embeddings)).shape[0]
            return np.full((n, len(self.classes)), 1.0 / len(self.classes))
        return _softmax(_add_bias(embeddings) @ self.weights)

    def predict(self, embeddings: np.ndarray) -> list[CyclePhase]:
        """Most likely phase per row."""
        return [self.classes[i] for i in self.predict_proba(embeddings).argmax(axis=1)]

    def predict_entropy(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalised predictive entropy in [0, 1]: the head's own uncertainty."""
        probs = np.clip(self.predict_proba(embeddings), EPS, 1.0)
        entropy = -(probs * np.log(probs)).sum(axis=1)
        return entropy / float(np.log(len(self.classes)))


@dataclass
class SymptomHead:
    """Predicts NEXT-day symptom probabilities (multilabel).

    Next-day rather than same-day on purpose: a same-day symptom head trained on
    a day whose symptoms are in the input is solving a copying task. The
    forecasting formulation is the one with clinical value and the one that can
    actually be wrong.
    """

    symptoms: tuple[str, ...] = ()
    alpha: float = 1.0
    max_iter: int = 200
    learning_rate: float = 0.5
    weights: dict[str, np.ndarray] = field(default_factory=dict)
    base_rates: dict[str, float] = field(default_factory=dict)

    def fit(
        self,
        embeddings: np.ndarray,
        targets: dict[str, np.ndarray],
        mask: dict[str, np.ndarray] | None = None,
    ) -> SymptomHead:
        """Fit one regularised logistic model per symptom."""
        self.symptoms = tuple(sorted(targets))
        x = _add_bias(embeddings)
        for name in self.symptoms:
            y = np.asarray(targets[name], dtype=float)
            rows = (
                np.asarray(mask[name], dtype=bool)
                if mask is not None and name in mask
                else np.isfinite(y)
            )
            self.base_rates[name] = float(y[rows].mean()) if rows.any() else 0.0
            if rows.sum() < 3 or len(np.unique(y[rows])) < 2:
                self.weights[name] = np.zeros(x.shape[1])
                continue
            w = np.zeros(x.shape[1])
            xr, yr = x[rows], y[rows]
            for _ in range(self.max_iter):
                p = 1.0 / (1.0 + np.exp(-np.clip(xr @ w, -30, 30)))
                grad = xr.T @ (p - yr) / max(len(yr), 1)
                grad[:-1] += self.alpha * w[:-1] / max(len(yr), 1)
                w -= self.learning_rate * grad
            self.weights[name] = w
        return self

    def predict_proba(self, embeddings: np.ndarray) -> dict[str, np.ndarray]:
        """Per-symptom probability of occurrence tomorrow."""
        x = _add_bias(embeddings)
        out: dict[str, np.ndarray] = {}
        for name in self.symptoms:
            w = self.weights.get(name)
            if w is None or not np.any(w):
                out[name] = np.full(x.shape[0], self.base_rates.get(name, 0.0))
            else:
                out[name] = 1.0 / (1.0 + np.exp(-np.clip(x @ w, -30, 30)))
        return out


@dataclass
class MaskedReconstructionHead:
    """Reconstructs artificially hidden channel values from the state embedding.

    This is the self-supervised head. Its value is that it needs no labels, so it
    can be trained on every participant-day including the many with no hormone
    test, which is where a hormone-only objective would have nothing to learn
    from.
    """

    n_channels: int = 0
    alpha: float = 1.0
    coefficients: np.ndarray | None = None

    def fit(
        self, embeddings: np.ndarray, targets: np.ndarray, mask: np.ndarray | None = None
    ) -> MaskedReconstructionHead:
        """Fit a multi-output ridge over the masked entries."""
        y = np.atleast_2d(np.asarray(targets, dtype=float))
        self.n_channels = y.shape[1]
        m = np.asarray(mask, dtype=bool) if mask is not None else np.isfinite(y)
        rows = m.any(axis=1)
        if rows.sum() < 3:
            self.coefficients = np.zeros((np.asarray(embeddings).shape[1] + 1, self.n_channels))
            return self
        filled = np.where(m, np.nan_to_num(y), 0.0)
        self.coefficients = _ridge_fit(np.asarray(embeddings)[rows], filled[rows], self.alpha)
        return self

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Reconstruct all channels ``(N, C)``."""
        if self.coefficients is None:
            n = np.atleast_2d(np.asarray(embeddings)).shape[0]
            return np.zeros((n, self.n_channels))
        return _add_bias(embeddings) @ self.coefficients


@dataclass
class TemporalHeads:
    """The four heads, fitted and applied together."""

    hormone: HormoneReconstructionHead = field(default_factory=HormoneReconstructionHead)
    cycle: CycleStateHead = field(default_factory=CycleStateHead)
    symptom: SymptomHead = field(default_factory=SymptomHead)
    masked: MaskedReconstructionHead = field(default_factory=MaskedReconstructionHead)

    def predict_all(self, embeddings: np.ndarray) -> dict[str, object]:
        """Run every head and return a dict keyed by head name."""
        hormones, uncertainty = self.hormone.predict_with_uncertainty(embeddings)
        return {
            "hormones": hormones,
            "hormone_uncertainty": uncertainty,
            "cycle_probabilities": self.cycle.predict_proba(embeddings),
            "cycle_entropy": self.cycle.predict_entropy(embeddings),
            "symptoms": self.symptom.predict_proba(embeddings),
            "masked_reconstruction": self.masked.predict(embeddings),
        }
