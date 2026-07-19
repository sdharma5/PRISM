"""A small multilayer perceptron over the static feature matrix.

Included to test whether nonlinearity buys anything over logistic regression on
cohorts of a few hundred patients. Usually it does not, and demonstrating that is
itself a useful result.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.neural_network import MLPClassifier

from evaluation.calibration import calibration_metrics_dict
from evaluation.classification import classification_metrics
from models.base import ArrayLike, BasePrismModel


class MLPStaticModel(BasePrismModel):
    """sklearn ``MLPClassifier`` wrapped in the PRISM model contract."""

    name = "static_mlp"
    version = "0.1.0"

    def __init__(
        self,
        *,
        hidden_layer_sizes: tuple[int, ...] | list[int] = (64, 32),
        activation: str = "relu",
        alpha: float = 1e-3,
        learning_rate_init: float = 1e-3,
        max_iter: int = 500,
        early_stopping: bool = True,
        validation_fraction: float = 0.15,
        n_iter_no_change: int = 20,
        random_state: int = 0,
        threshold: float = 0.5,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            hidden_layer_sizes=list(hidden_layer_sizes),
            activation=activation,
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            random_state=random_state,
            threshold=threshold,
            **kwargs,
        )
        self.threshold = threshold
        self.estimator = MLPClassifier(
            hidden_layer_sizes=tuple(hidden_layer_sizes),
            activation=activation,
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            # Early stopping carves its validation split out of the *training*
            # rows it is handed, which keeps the test fold untouched.
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            random_state=random_state,
        )

    def fit(self, X: ArrayLike, y: ArrayLike | None = None, **kwargs: Any) -> MLPStaticModel:
        if y is None:
            raise ValueError(f"{self.name} is supervised and requires labels.")
        X_arr = self._record_features(X)
        y_arr = np.asarray(y, dtype=float).astype(int)

        # Early stopping needs enough rows to hold a stratified validation slice.
        if self.estimator.early_stopping and X_arr.shape[0] < 50:
            self.estimator.set_params(early_stopping=False)

        import warnings

        from sklearn.exceptions import ConvergenceWarning

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.estimator.fit(X_arr, y_arr)

        self.is_fitted = True
        self.training_metadata_ = {
            "n_train": int(X_arr.shape[0]),
            "n_features": int(X_arr.shape[1]),
            "n_iter": int(getattr(self.estimator, "n_iter_", 0)),
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
        y_arr = np.asarray(y, dtype=float)
        probabilities = self.predict_proba(X)
        metrics = classification_metrics(y_arr, probabilities, threshold=self.threshold)
        metrics.update(calibration_metrics_dict(y_arr, probabilities))
        return metrics


__all__ = ["MLPStaticModel"]
