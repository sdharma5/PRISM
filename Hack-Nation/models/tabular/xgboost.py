"""Gradient-boosted trees, with a graceful fallback.

xgboost is an optional dependency. Rather than making the whole pipeline
uninstallable without it, this model falls back to sklearn's
``HistGradientBoostingClassifier`` and *says so loudly* — a silent substitution
would make two runs incomparable while both claim the name "xgboost".
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from evaluation.calibration import calibration_metrics_dict
from evaluation.classification import classification_metrics
from models.base import ArrayLike, BasePrismModel


def xgboost_available() -> bool:
    """Whether the real xgboost backend can be imported."""
    try:
        import xgboost  # noqa: F401
    except ImportError:
        return False
    return True


class XGBoostStaticModel(BasePrismModel):
    """Gradient boosting over the static feature matrix."""

    name = "static_xgboost"
    version = "0.1.0"

    def __init__(
        self,
        *,
        n_estimators: int = 300,
        max_depth: int = 3,
        learning_rate: float = 0.05,
        subsample: float = 0.9,
        colsample_bytree: float = 0.9,
        reg_lambda: float = 1.0,
        min_child_weight: float = 1.0,
        random_state: int = 0,
        threshold: float = 0.5,
        scale_pos_weight: float | None = None,
        allow_fallback: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_lambda=reg_lambda,
            min_child_weight=min_child_weight,
            random_state=random_state,
            threshold=threshold,
            scale_pos_weight=scale_pos_weight,
            **kwargs,
        )
        self.threshold = threshold
        self.allow_fallback = allow_fallback
        self.backend: str = "xgboost" if xgboost_available() else "sklearn_hist_gradient_boosting"
        self.estimator = self._build_estimator()

    def _build_estimator(self) -> Any:
        """Construct the backend lazily so importing this module never needs xgboost."""
        p = self.params
        if self.backend == "xgboost":
            import xgboost as xgb

            return xgb.XGBClassifier(
                n_estimators=p["n_estimators"],
                max_depth=p["max_depth"],
                learning_rate=p["learning_rate"],
                subsample=p["subsample"],
                colsample_bytree=p["colsample_bytree"],
                reg_lambda=p["reg_lambda"],
                min_child_weight=p["min_child_weight"],
                random_state=p["random_state"],
                scale_pos_weight=p.get("scale_pos_weight") or 1.0,
                eval_metric="logloss",
                tree_method="hist",
                n_jobs=1,
            )

        if not self.allow_fallback:
            raise ImportError(
                "xgboost is not installed and allow_fallback=False. Install the 'tabular' extra."
            )

        warnings.warn(
            "xgboost is not installed; falling back to sklearn "
            "HistGradientBoostingClassifier. Results are NOT comparable to an xgboost run — "
            "the recorded backend is 'sklearn_hist_gradient_boosting'.",
            RuntimeWarning,
            stacklevel=3,
        )
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(
            max_iter=p["n_estimators"],
            max_depth=p["max_depth"],
            learning_rate=p["learning_rate"],
            l2_regularization=p["reg_lambda"],
            random_state=p["random_state"],
        )

    def fit(self, X: ArrayLike, y: ArrayLike | None = None, **kwargs: Any) -> XGBoostStaticModel:
        if y is None:
            raise ValueError(f"{self.name} is supervised and requires labels.")
        X_arr = self._record_features(X)
        y_arr = np.asarray(y, dtype=float).astype(int)
        self.estimator.fit(X_arr, y_arr)
        self.is_fitted = True
        self.training_metadata_ = {
            "backend": self.backend,
            "n_train": int(X_arr.shape[0]),
            "n_features": int(X_arr.shape[1]),
            "positive_rate": float(np.mean(y_arr)),
        }
        return self

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        self._require_fitted()
        return np.asarray(self.estimator.predict_proba(self._as_array(X)))[:, 1]

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

    def export_model_card_metadata(self, **kwargs: Any) -> Any:
        """Record which backend actually ran, so cards cannot misattribute results."""
        extra = list(kwargs.pop("extra_limitations", []) or [])
        if self.backend != "xgboost":
            extra.append(
                f"Fitted with fallback backend '{self.backend}' rather than xgboost; "
                "hyperparameters were mapped approximately."
            )
        return super().export_model_card_metadata(extra_limitations=extra, **kwargs)


__all__ = ["XGBoostStaticModel", "xgboost_available"]
