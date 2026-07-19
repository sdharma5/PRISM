"""The fold loop. Preprocessing is fitted inside each training fold, always.

This is the cardinal rule of the project: an imputer that saw a test-fold median,
or a scaler that saw a test-fold mean, has leaked the evaluation distribution
into the model. The resulting metrics are optimistic and the bug is invisible in
the output. Hence one function — :func:`fit_preprocessing_on_fold` — is the only
place a transform is ever fitted, and it only ever receives training rows.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features.static_features import indicator_columns_of
from models.base import BasePrismModel
from schemas.patient import SplitManifest
from training.callbacks import Callback, CallbackList
from training.seeding import derive_seed, seed_everything
from training.splits import fold_row_indices


@dataclass
class PreprocessingSpec:
    """Which columns get which treatment. Resolved from the config, never guessed."""

    numeric_columns: list[str] = field(default_factory=list)
    categorical_columns: list[str] = field(default_factory=list)
    passthrough_columns: list[str] = field(default_factory=list)
    numeric_impute_strategy: str = "median"
    categorical_impute_strategy: str = "most_frequent"
    scale: bool = True
    add_indicator: bool = True


def infer_preprocessing_spec(
    X: pd.DataFrame,
    *,
    numeric_impute_strategy: str = "median",
    scale: bool = True,
    add_indicator: bool = True,
) -> PreprocessingSpec:
    """Split columns into numeric / categorical / passthrough.

    Missingness indicators are passed through untouched: they are already 0/1 and
    complete, and imputing or scaling them would destroy their meaning.
    """
    indicators = set(indicator_columns_of(X))
    numeric: list[str] = []
    categorical: list[str] = []
    for column in X.columns:
        if column in indicators:
            continue
        if pd.api.types.is_numeric_dtype(X[column]):
            numeric.append(str(column))
        else:
            categorical.append(str(column))
    return PreprocessingSpec(
        numeric_columns=numeric,
        categorical_columns=categorical,
        passthrough_columns=sorted(indicators),
        numeric_impute_strategy=numeric_impute_strategy,
        scale=scale,
        add_indicator=add_indicator,
    )


def build_preprocessing_pipeline(spec: PreprocessingSpec) -> Pipeline:
    """impute -> scale -> encode, plus explicit missingness indicators.

    Returned *unfitted*. Fitting is the caller's responsibility and must happen on
    training rows only.
    """
    numeric_steps: list[tuple[str, Any]] = [
        (
            "impute",
            SimpleImputer(
                strategy=spec.numeric_impute_strategy,
                add_indicator=spec.add_indicator,
                keep_empty_features=True,
            ),
        )
    ]
    if spec.scale:
        numeric_steps.append(("scale", StandardScaler()))
    numeric_pipeline = Pipeline(numeric_steps)

    categorical_pipeline = Pipeline(
        [
            (
                "impute",
                SimpleImputer(
                    strategy=spec.categorical_impute_strategy,
                    fill_value="__missing__",
                    add_indicator=spec.add_indicator,
                ),
            ),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    transformers: list[tuple[str, Any, list[str]]] = []
    if spec.numeric_columns:
        transformers.append(("numeric", numeric_pipeline, spec.numeric_columns))
    if spec.categorical_columns:
        transformers.append(("categorical", categorical_pipeline, spec.categorical_columns))
    if spec.passthrough_columns:
        transformers.append(("indicators", "passthrough", spec.passthrough_columns))

    if not transformers:
        raise ValueError("PreprocessingSpec selected no columns at all.")

    return Pipeline(
        [
            (
                "columns",
                ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0),
            )
        ]
    )


def fit_preprocessing_on_fold(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[Pipeline, np.ndarray, np.ndarray]:
    """Fit on training rows, then transform both sides.

    ``X_test`` is passed only to :meth:`transform`. If you ever find yourself
    wanting to pass it to ``fit``, the answer is no.
    """
    fitted = pipeline.fit(X_train)
    return fitted, np.asarray(fitted.transform(X_train)), np.asarray(fitted.transform(X_test))


def transformed_feature_names(pipeline: Pipeline, fallback_width: int) -> list[str]:
    """Best-effort output column names for the fitted pipeline."""
    try:
        return [str(name) for name in pipeline.get_feature_names_out()]
    except (AttributeError, ValueError):
        return [f"x{i}" for i in range(fallback_width)]


@dataclass
class FoldResult:
    """Everything one fold produced."""

    fold: int
    seed: int
    n_train: int
    n_test: int
    metrics: dict[str, float]
    test_patient_ids: list[str]
    y_true: np.ndarray
    y_prob: np.ndarray | None
    model: BasePrismModel | None = None
    pipeline: Pipeline | None = None


@dataclass
class CrossValidationResult:
    """The fold loop's output, ready for ``evaluation.reports``."""

    fold_results: list[FoldResult] = field(default_factory=list)
    predictions: pd.DataFrame = field(default_factory=pd.DataFrame)
    feature_names: list[str] = field(default_factory=list)
    preprocessing_spec: PreprocessingSpec | None = None
    last_pipeline: Pipeline | None = None

    def fold_records(self) -> list[dict[str, Any]]:
        return [
            {
                "fold": r.fold,
                "seed": r.seed,
                "n_train": r.n_train,
                "n_test": r.n_test,
                "metrics": r.metrics,
            }
            for r in self.fold_results
        ]


ModelFactory = Callable[[int], BasePrismModel]


def run_cross_validation(
    X: pd.DataFrame,
    y: pd.Series,
    patient_ids: Sequence[str],
    manifest: SplitManifest,
    model_factory: ModelFactory,
    *,
    spec: PreprocessingSpec | None = None,
    base_seed: int = 0,
    keep_models: bool = False,
    callbacks: Sequence[Callback] | None = None,
) -> CrossValidationResult:
    """Run the fold loop with fold-local preprocessing and fold-local model fitting."""
    manifest.assert_disjoint()

    spec = spec or infer_preprocessing_spec(X)
    y_series = pd.to_numeric(pd.Series(y).reset_index(drop=True), errors="coerce")
    y_arr = y_series.to_numpy(dtype=float)
    ids = [str(p) for p in patient_ids]
    X = X.reset_index(drop=True)

    cb = CallbackList(list(callbacks or []))
    cb.on_experiment_start({"n_folds": len(manifest.folds), "n_rows": len(X)})

    results: list[FoldResult] = []
    prediction_rows: list[pd.DataFrame] = []
    feature_names: list[str] = []
    last_pipeline: Pipeline | None = None

    for fold_index, fold in enumerate(manifest.folds):
        fold_seed = derive_seed(base_seed, "fold", fold_index)
        seed_everything(fold_seed)

        train_idx, test_idx = fold_row_indices(fold, ids)
        if train_idx.size == 0 or test_idx.size == 0:
            continue

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_arr[train_idx], y_arr[test_idx]

        cb.on_fold_start(fold_index, {"n_train": int(train_idx.size), "n_test": int(test_idx.size)})

        # -- The leakage boundary. Nothing below this line may see X_test values
        #    during a fit() call.
        pipeline = build_preprocessing_pipeline(spec)
        pipeline, X_train_t, X_test_t = fit_preprocessing_on_fold(pipeline, X_train, X_test)
        last_pipeline = pipeline
        if not feature_names:
            feature_names = transformed_feature_names(pipeline, X_train_t.shape[1])

        model = model_factory(fold_seed)
        model.fit(X_train_t, y_train)

        y_prob: np.ndarray | None = None
        if model.is_classifier:
            try:
                y_prob = np.asarray(model.predict_proba(X_test_t), dtype=float)
            except NotImplementedError:
                y_prob = None

        metrics = model.evaluate(X_test_t, y_test)

        results.append(
            FoldResult(
                fold=fold_index,
                seed=fold_seed,
                n_train=int(train_idx.size),
                n_test=int(test_idx.size),
                metrics=metrics,
                test_patient_ids=[ids[i] for i in test_idx],
                y_true=y_test,
                y_prob=y_prob,
                model=model if keep_models else None,
                pipeline=pipeline if keep_models else None,
            )
        )

        if y_prob is not None:
            prediction_rows.append(
                pd.DataFrame(
                    {
                        "fold": fold_index,
                        "seed": fold_seed,
                        "patient_id": [ids[i] for i in test_idx],
                        "y_true": y_test,
                        "y_prob": y_prob,
                    }
                )
            )

        cb.on_fold_end(fold_index, metrics)

    predictions = (
        pd.concat(prediction_rows, ignore_index=True) if prediction_rows else pd.DataFrame()
    )
    cb.on_experiment_end({"n_folds": len(results)})

    return CrossValidationResult(
        fold_results=results,
        predictions=predictions,
        feature_names=feature_names,
        preprocessing_spec=spec,
        last_pipeline=last_pipeline,
    )


def fit_final_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_factory: ModelFactory,
    *,
    spec: PreprocessingSpec | None = None,
    seed: int = 0,
) -> tuple[BasePrismModel, Pipeline]:
    """Refit on all available rows, for export only — never for reporting metrics."""
    spec = spec or infer_preprocessing_spec(X)
    seed_everything(seed)
    pipeline = build_preprocessing_pipeline(spec).fit(X)
    model = model_factory(seed)
    model.fit(
        np.asarray(pipeline.transform(X)),
        pd.to_numeric(pd.Series(y), errors="coerce").to_numpy(dtype=float),
    )
    return model, pipeline


__all__ = [
    "CrossValidationResult",
    "FoldResult",
    "PreprocessingSpec",
    "build_preprocessing_pipeline",
    "fit_final_model",
    "fit_preprocessing_on_fold",
    "infer_preprocessing_spec",
    "run_cross_validation",
    "transformed_feature_names",
]
