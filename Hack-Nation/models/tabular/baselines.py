"""Floor baselines. Any model that cannot beat these has learned nothing useful.

A majority-class predictor and a one-variable rule are cheap to run and expose
the two most common self-deceptions: a high accuracy that is just prevalence, and
a "complex model" that only rediscovered a single strong feature.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from evaluation.calibration import calibration_metrics_dict
from evaluation.classification import best_threshold, classification_metrics, threshold_metrics
from models.base import ArrayLike, BasePrismModel


class MajorityClassBaseline(BasePrismModel):
    """Always predicts the training-fold majority class at its base rate."""

    name = "baseline_majority_class"
    version = "0.1.0"

    def __init__(self, *, threshold: float = 0.5, **kwargs: Any) -> None:
        super().__init__(threshold=threshold, **kwargs)
        self.threshold = threshold
        self.majority_class_: int = 0
        self.positive_rate_: float = 0.0

    def fit(self, X: ArrayLike, y: ArrayLike | None = None, **kwargs: Any) -> MajorityClassBaseline:
        if y is None:
            raise ValueError(f"{self.name} requires labels.")
        self._record_features(X)
        y_arr = np.asarray(y, dtype=float)
        self.positive_rate_ = float(np.nanmean(y_arr)) if y_arr.size else 0.0
        self.majority_class_ = int(self.positive_rate_ >= 0.5)
        self.is_fitted = True
        return self

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        self._require_fitted()
        return np.full(self._as_array(X).shape[0], self.positive_rate_, dtype=float)

    def predict(self, X: ArrayLike) -> np.ndarray:
        self._require_fitted()
        return np.full(self._as_array(X).shape[0], self.majority_class_, dtype=int)

    def evaluate(self, X: ArrayLike, y: ArrayLike | None = None, **kwargs: Any) -> dict[str, float]:
        if y is None:
            raise ValueError(f"{self.name}.evaluate requires labels.")
        y_arr = np.asarray(y, dtype=float)
        probabilities = self.predict_proba(X)
        metrics = classification_metrics(y_arr, probabilities, threshold=self.threshold)
        metrics.update(calibration_metrics_dict(y_arr, probabilities))
        return metrics


class SingleFeatureRuleBaseline(BasePrismModel):
    """Threshold the single most univariately-informative feature.

    Feature and threshold are both chosen on the training fold only; picking them
    on the test fold would make this "baseline" beat real models for free.
    """

    name = "baseline_single_feature_rule"
    version = "0.1.0"

    def __init__(
        self,
        *,
        objective: str = "balanced_accuracy",
        feature_index: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(objective=objective, feature_index=feature_index, **kwargs)
        self.objective = objective
        self.fixed_feature_index = feature_index
        self.feature_index_: int = 0
        self.threshold_: float = 0.0
        self.sign_: int = 1

    def fit(
        self, X: ArrayLike, y: ArrayLike | None = None, **kwargs: Any
    ) -> SingleFeatureRuleBaseline:
        if y is None:
            raise ValueError(f"{self.name} requires labels.")
        X_arr = self._record_features(X)
        y_arr = np.asarray(y, dtype=float)

        if self.fixed_feature_index is not None:
            candidates = [int(self.fixed_feature_index)]
        else:
            candidates = list(range(X_arr.shape[1]))

        best_score, best = -np.inf, (0, 0.0, 1)
        for index in candidates:
            column = X_arr[:, index]
            if not np.isfinite(column).any() or np.nanstd(column) == 0:
                continue
            for sign in (1, -1):
                scores = sign * np.nan_to_num(column, nan=0.0)
                threshold = best_threshold(y_arr, scores, objective=self.objective)
                value = threshold_metrics(y_arr, (scores >= threshold).astype(int)).get(
                    self.objective, float("nan")
                )
                if np.isfinite(value) and value > best_score:
                    best_score, best = float(value), (index, float(threshold), sign)

        self.feature_index_, self.threshold_, self.sign_ = best
        self.is_fitted = True
        self.training_metadata_ = {
            "selected_feature_index": self.feature_index_,
            "selected_feature": (
                self.feature_names_[self.feature_index_] if self.feature_names_ else None
            ),
            "threshold": self.threshold_,
            "sign": self.sign_,
            f"train_{self.objective}": best_score if np.isfinite(best_score) else float("nan"),
        }
        return self

    def _scores(self, X: ArrayLike) -> np.ndarray:
        self._require_fitted()
        column = self._as_array(X)[:, self.feature_index_]
        return self.sign_ * np.nan_to_num(column, nan=0.0)

    def predict(self, X: ArrayLike) -> np.ndarray:
        return (self._scores(X) >= self.threshold_).astype(int)

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """A squashed distance from the rule's threshold — ranking, not calibration."""
        return 1.0 / (1.0 + np.exp(-(self._scores(X) - self.threshold_)))

    def evaluate(self, X: ArrayLike, y: ArrayLike | None = None, **kwargs: Any) -> dict[str, float]:
        if y is None:
            raise ValueError(f"{self.name}.evaluate requires labels.")
        y_arr = np.asarray(y, dtype=float)
        probabilities = self.predict_proba(X)
        metrics = classification_metrics(y_arr, probabilities, threshold=0.5)
        metrics.update(calibration_metrics_dict(y_arr, probabilities))
        return metrics


__all__ = ["MajorityClassBaseline", "SingleFeatureRuleBaseline"]
