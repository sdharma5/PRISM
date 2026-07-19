"""Regularized logistic regression — the reference model for static features.

Every more complex model must beat this one on the same folds before it earns a
place. A linear model with a readable coefficient vector is also far easier to
audit for "the model learned the missingness pattern, not the phenotype".
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

from evaluation.calibration import calibration_metrics_dict
from evaluation.classification import classification_metrics
from models.base import ArrayLike, BasePrismModel


class LogisticStaticModel(BasePrismModel):
    """L2-penalized logistic regression with optional class balancing."""

    name = "static_logistic"
    version = "0.1.0"

    def __init__(
        self,
        *,
        C: float = 1.0,
        penalty: str = "l2",
        solver: str = "lbfgs",
        max_iter: int = 2000,
        class_weight: str | None = "balanced",
        random_state: int = 0,
        threshold: float = 0.5,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            C=C,
            penalty=penalty,
            solver=solver,
            max_iter=max_iter,
            class_weight=class_weight,
            random_state=random_state,
            threshold=threshold,
            **kwargs,
        )
        self.threshold = threshold
        # sklearn >= 1.8 deprecates the `penalty` argument in favour of `l1_ratio`,
        # so the config keyword is translated rather than passed through.
        estimator_kwargs: dict[str, Any] = {
            "C": C,
            "solver": solver,
            "max_iter": max_iter,
            "class_weight": class_weight,
            "random_state": random_state,
        }
        l1_ratio = {"l2": 0.0, "l1": 1.0}.get(penalty)
        if l1_ratio is None and penalty is not None:
            raise ValueError(f"Unsupported penalty '{penalty}'; use 'l1' or 'l2'.")
        if l1_ratio:
            estimator_kwargs["l1_ratio"] = l1_ratio
        self.estimator = LogisticRegression(**estimator_kwargs)

    def fit(self, X: ArrayLike, y: ArrayLike | None = None, **kwargs: Any) -> LogisticStaticModel:
        if y is None:
            raise ValueError(f"{self.name} is supervised and requires labels.")
        X_arr = self._record_features(X)
        y_arr = np.asarray(y, dtype=float)
        self.estimator.fit(X_arr, y_arr.astype(int))
        self.is_fitted = True
        self.training_metadata_ = {
            "n_train": int(X_arr.shape[0]),
            "n_features": int(X_arr.shape[1]),
            "positive_rate": float(np.mean(y_arr)),
        }
        return self

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        self._require_fitted()
        return self.estimator.predict_proba(self._as_array(X))[:, 1]

    def predict(self, X: ArrayLike) -> np.ndarray:
        return (self.predict_proba(X) >= self.threshold).astype(int)

    def evaluate(self, X: ArrayLike, y: ArrayLike | None = None, **kwargs: Any) -> dict[str, float]:
        if y is None:
            raise ValueError(f"{self.name}.evaluate requires labels.")
        probabilities = self.predict_proba(X)
        y_arr = np.asarray(y, dtype=float)
        metrics = classification_metrics(y_arr, probabilities, threshold=self.threshold)
        metrics.update(calibration_metrics_dict(y_arr, probabilities))
        return metrics

    def coefficients(self) -> dict[str, float]:
        """Feature -> coefficient, for the "what did it actually use?" check."""
        self._require_fitted()
        names = self.feature_names_ or [f"f{i}" for i in range(self.estimator.coef_.shape[1])]
        return dict(zip(names, self.estimator.coef_[0].astype(float).tolist(), strict=False))


class RandomForestStaticModel(BasePrismModel):
    """Random forest baseline: nonlinear, low-tuning, and a useful sanity check."""

    name = "static_random_forest"
    version = "0.1.0"

    def __init__(
        self,
        *,
        n_estimators: int = 300,
        max_depth: int | None = None,
        min_samples_leaf: int = 2,
        class_weight: str | None = "balanced_subsample",
        random_state: int = 0,
        threshold: float = 0.5,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            random_state=random_state,
            threshold=threshold,
            **kwargs,
        )
        from sklearn.ensemble import RandomForestClassifier

        self.threshold = threshold
        self.estimator = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=1,
        )

    def fit(
        self, X: ArrayLike, y: ArrayLike | None = None, **kwargs: Any
    ) -> RandomForestStaticModel:
        if y is None:
            raise ValueError(f"{self.name} is supervised and requires labels.")
        X_arr = self._record_features(X)
        self.estimator.fit(X_arr, np.asarray(y, dtype=float).astype(int))
        self.is_fitted = True
        return self

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        self._require_fitted()
        return self.estimator.predict_proba(self._as_array(X))[:, 1]

    def predict(self, X: ArrayLike) -> np.ndarray:
        return (self.predict_proba(X) >= self.threshold).astype(int)

    def evaluate(self, X: ArrayLike, y: ArrayLike | None = None, **kwargs: Any) -> dict[str, float]:
        if y is None:
            raise ValueError(f"{self.name}.evaluate requires labels.")
        probabilities = self.predict_proba(X)
        y_arr = np.asarray(y, dtype=float)
        metrics = classification_metrics(y_arr, probabilities, threshold=self.threshold)
        metrics.update(calibration_metrics_dict(y_arr, probabilities))
        return metrics

    def feature_importances(self) -> dict[str, float]:
        self._require_fitted()
        names = self.feature_names_ or [
            f"f{i}" for i in range(len(self.estimator.feature_importances_))
        ]
        return dict(
            zip(names, self.estimator.feature_importances_.astype(float).tolist(), strict=False)
        )


__all__ = ["LogisticStaticModel", "RandomForestStaticModel"]
