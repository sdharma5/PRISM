"""The contract every PRISM model implements.

One ABC keeps the training engine, the evaluation layer and the model-card
exporter interchangeable across modalities: the engine never needs to know
whether it is driving a logistic regression or a masked autoencoder.
"""

from __future__ import annotations

import json
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
import pandas as pd

from schemas.model_output import ModelCardMetadata

#: Explicitly a ``TypeAlias``: without the annotation this reads as a plain
#: module-level variable and every ``X: ArrayLike`` below is silently unchecked.
ArrayLike: TypeAlias = np.ndarray | pd.DataFrame


class BasePrismModel(ABC):
    """Abstract base for every model in PRISM.

    Subclasses must be constructible from config alone, so an experiment is fully
    described by its resolved YAML plus a seed.
    """

    #: Human-readable model name, used in artifacts and model cards.
    name: str = "base_prism_model"
    #: Bumped whenever fitted behaviour changes in a way that invalidates old runs.
    version: str = "0.1.0"
    #: Whether ``predict_proba`` is meaningful for this model.
    is_classifier: bool = True

    def __init__(self, **params: Any) -> None:
        self.params: dict[str, Any] = dict(params)
        self.is_fitted: bool = False
        self.feature_names_: list[str] | None = None
        self.training_metadata_: dict[str, Any] = {}

    # -- Core API ----------------------------------------------------------

    @abstractmethod
    def fit(self, X: ArrayLike, y: ArrayLike | None = None, **kwargs: Any) -> BasePrismModel:
        """Fit on a *training fold only*. Returns ``self`` for chaining."""

    @abstractmethod
    def predict(self, X: ArrayLike) -> np.ndarray:
        """Return hard predictions (classifiers) or reconstructions (autoencoders)."""

    @abstractmethod
    def evaluate(self, X: ArrayLike, y: ArrayLike | None = None, **kwargs: Any) -> dict[str, float]:
        """Return a flat ``{metric_name: value}`` mapping for this model's task."""

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """Positive-class probabilities. Classifiers must override this."""
        raise NotImplementedError(f"{type(self).__name__} does not expose probabilities.")

    # -- Persistence -------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        """Persist the fitted model plus a sidecar JSON describing it."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(self, fh)
        sidecar = path.with_suffix(path.suffix + ".json")
        sidecar.write_text(
            json.dumps(
                {
                    "name": self.name,
                    "version": self.version,
                    "class": type(self).__name__,
                    "module": type(self).__module__,
                    "params": _jsonable(self.params),
                    "is_fitted": self.is_fitted,
                    "feature_names": self.feature_names_,
                    "training_metadata": _jsonable(self.training_metadata_),
                },
                indent=2,
            )
            + "\n"
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> BasePrismModel:
        """Load a model previously written by :meth:`save`."""
        with Path(path).open("rb") as fh:
            model = pickle.load(fh)
        if not isinstance(model, BasePrismModel):
            raise TypeError(f"{path} does not contain a BasePrismModel.")
        return model

    # -- Documentation -----------------------------------------------------

    def export_model_card_metadata(
        self,
        *,
        intended_use: str | None = None,
        training_datasets: list[str] | None = None,
        evaluation_datasets: list[str] | None = None,
        metrics: dict[str, float] | None = None,
        extra_limitations: list[str] | None = None,
    ) -> ModelCardMetadata:
        """Emit the model card metadata block for this model.

        Defaults deliberately over-state the limitations: an unqualified card is
        the easiest way for a research artifact to be mistaken for a product.
        """
        limitations = [
            "Trained and evaluated on research cohorts that are not representative of any "
            "general population.",
            "Missingness is informative in these cohorts; measured performance partly reflects "
            "who was tested rather than who has the phenotype.",
            "Labels are dataset-provided and were not re-adjudicated by clinicians.",
            "No prospective, external or temporal validation has been performed.",
            *(extra_limitations or []),
        ]
        return ModelCardMetadata(
            model_name=self.name,
            model_version=self.version,
            intended_use=(
                intended_use
                or "Research-only exploration of hormonal-health phenotype profiles on "
                "de-identified cohort data."
            ),
            out_of_scope_uses=[
                "Any clinical decision-making, triage, screening or referral.",
                "Communicating a result to a patient as a finding about their health.",
                "Insurance, employment, or any other consequential eligibility decision.",
                "Deployment on populations unlike the research cohorts used here.",
            ],
            training_datasets=list(training_datasets or []),
            evaluation_datasets=list(evaluation_datasets or []),
            metrics=dict(metrics or {}),
            limitations=limitations,
            ethical_considerations=[
                "Cohort composition is skewed; subgroup performance must be reported before "
                "any conclusion is drawn about a group.",
                "Symptom-only evidence is qualified as such and must never be presented as a "
                "biochemical or imaging finding.",
                "Outputs are phenotype profiles a patient may resemble, not categories a "
                "patient belongs to.",
            ],
        )

    # -- Helpers for subclasses -------------------------------------------

    def _record_features(self, X: ArrayLike) -> np.ndarray:
        """Remember the column order seen at fit time and return a plain array."""
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = [str(c) for c in X.columns]
            return X.to_numpy(dtype=float)
        self.feature_names_ = self.feature_names_ or [f"f{i}" for i in range(np.shape(X)[1])]
        return np.asarray(X, dtype=float)

    def _as_array(self, X: ArrayLike) -> np.ndarray:
        """Coerce to an array, enforcing the fitted column order when known."""
        if isinstance(X, pd.DataFrame):
            if self.feature_names_ is not None and list(X.columns) != self.feature_names_:
                missing = [c for c in self.feature_names_ if c not in X.columns]
                if missing:
                    raise KeyError(f"{self.name}: input is missing fitted feature(s) {missing}.")
                X = X[self.feature_names_]
            return X.to_numpy(dtype=float)
        return np.asarray(X, dtype=float)

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError(f"{self.name}: call fit() before predict()/evaluate().")


def _jsonable(obj: Any) -> Any:
    """Best-effort conversion of arbitrary params into JSON-serializable values."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, str | int | float | bool) or obj is None:
        return obj
    return repr(obj)


__all__ = ["ArrayLike", "BasePrismModel"]
