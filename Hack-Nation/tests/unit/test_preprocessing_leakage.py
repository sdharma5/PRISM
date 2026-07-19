"""Proof that preprocessing statistics come from the training fold and nothing else.

These tests are written so they FAIL if someone "helpfully" fits the imputer or
scaler on the full matrix. That refactor is easy to make, looks harmless in
review, and silently inflates every downstream metric.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features.static_features import build_static_features
from models.tabular.logistic import LogisticStaticModel
from schemas.patient import SplitManifest
from tests.fixtures.synthetic_tabular import make_synthetic_cohort
from training.engine import (
    build_preprocessing_pipeline,
    fit_preprocessing_on_fold,
    infer_preprocessing_spec,
    run_cross_validation,
)


def _frame(values: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"a": values})


def test_imputer_uses_training_median_only():
    """Poisoning the test fold with extreme values must not move the fitted median."""
    X_train = _frame([1.0, 2.0, 3.0, np.nan, 5.0])
    X_test_normal = _frame([2.0, np.nan, 4.0])
    X_test_extreme = _frame([1e9, np.nan, 1e9])

    spec = infer_preprocessing_spec(X_train, scale=False)

    fitted_a, _, _ = fit_preprocessing_on_fold(
        build_preprocessing_pipeline(spec), X_train, X_test_normal
    )
    fitted_b, _, _ = fit_preprocessing_on_fold(
        build_preprocessing_pipeline(spec), X_train, X_test_extreme
    )

    imputer_a = fitted_a.named_steps["columns"].named_transformers_["numeric"].named_steps["impute"]
    imputer_b = fitted_b.named_steps["columns"].named_transformers_["numeric"].named_steps["impute"]

    assert imputer_a.statistics_ == pytest.approx(imputer_b.statistics_)
    # The median of the observed training values 1,2,3,5 is 2.5.
    assert imputer_a.statistics_[0] == pytest.approx(2.5)


def test_scaler_statistics_are_unaffected_by_the_test_fold():
    X_train = _frame([1.0, 2.0, 3.0, 4.0])
    X_test_small = _frame([2.0, 3.0])
    X_test_huge = _frame([1e6, -1e6])

    spec = infer_preprocessing_spec(X_train, scale=True)

    fitted_small, _, _ = fit_preprocessing_on_fold(
        build_preprocessing_pipeline(spec), X_train, X_test_small
    )
    fitted_huge, _, _ = fit_preprocessing_on_fold(
        build_preprocessing_pipeline(spec), X_train, X_test_huge
    )

    scaler_small = (
        fitted_small.named_steps["columns"].named_transformers_["numeric"].named_steps["scale"]
    )
    scaler_huge = (
        fitted_huge.named_steps["columns"].named_transformers_["numeric"].named_steps["scale"]
    )

    assert scaler_small.mean_ == pytest.approx(scaler_huge.mean_)
    assert scaler_small.scale_ == pytest.approx(scaler_huge.scale_)
    assert scaler_small.mean_[0] == pytest.approx(2.5)


def test_transformed_test_fold_is_not_standardized_to_zero_mean():
    """A test fold scaled to mean 0 is the fingerprint of a scaler fitted on everything."""
    X_train = _frame([1.0, 2.0, 3.0, 4.0])
    X_test = _frame([10.0, 11.0, 12.0])

    spec = infer_preprocessing_spec(X_train, scale=True)
    _, _, X_test_t = fit_preprocessing_on_fold(build_preprocessing_pipeline(spec), X_train, X_test)

    assert abs(float(np.mean(X_test_t[:, 0]))) > 1.0


def test_cross_validation_pipeline_differs_between_folds():
    """Fold-local fitting implies fold-specific statistics; identical ones are suspicious."""
    df = make_synthetic_cohort(n=120, seed=5)
    matrix = build_static_features(df)
    manifest = SplitManifest(
        manifest_id="m",
        dataset_id="synthetic",
        dataset_version="v1",
        strategy="repeated_stratified_kfold",
        n_splits=2,
        folds=[
            {
                "train": matrix.patient_ids.tolist()[:60],
                "test": matrix.patient_ids.tolist()[60:],
            },
            {
                "train": matrix.patient_ids.tolist()[60:],
                "test": matrix.patient_ids.tolist()[:60],
            },
        ],
    )

    result = run_cross_validation(
        matrix.X,
        matrix.y,
        matrix.patient_ids.tolist(),
        manifest,
        lambda seed: LogisticStaticModel(random_state=seed),
        keep_models=True,
    )

    pipelines = [r.pipeline for r in result.fold_results]
    assert len(pipelines) == 2

    def _medians(pipeline):
        return (
            pipeline.named_steps["columns"]
            .named_transformers_["numeric"]
            .named_steps["impute"]
            .statistics_
        )

    assert not np.allclose(_medians(pipelines[0]), _medians(pipelines[1]))


def test_full_matrix_fit_would_produce_different_statistics():
    """Explicitly contrast fold-local fitting with the leaky alternative."""
    df = make_synthetic_cohort(n=100, seed=7)
    matrix = build_static_features(df, add_missingness_indicators=False)
    spec = infer_preprocessing_spec(matrix.X)

    train = matrix.X.iloc[:50]
    test = matrix.X.iloc[50:]

    fold_local, _, _ = fit_preprocessing_on_fold(build_preprocessing_pipeline(spec), train, test)
    leaky = build_preprocessing_pipeline(spec).fit(matrix.X)

    def _scale(pipeline):
        return (
            pipeline.named_steps["columns"]
            .named_transformers_["numeric"]
            .named_steps["scale"]
            .mean_
        )

    # If these ever became equal, fold-local fitting stopped happening.
    assert not np.allclose(_scale(fold_local), _scale(leaky))


def test_model_never_sees_untransformed_test_rows_during_fit(monkeypatch):
    """Guard the boundary directly: record every array handed to a fit() call."""
    df = make_synthetic_cohort(n=80, seed=11)
    matrix = build_static_features(df, add_missingness_indicators=False)
    ids = matrix.patient_ids.tolist()
    manifest = SplitManifest(
        manifest_id="m",
        dataset_id="synthetic",
        dataset_version="v1",
        strategy="holdout",
        n_splits=1,
        folds=[{"train": ids[:60], "test": ids[60:]}],
    )

    seen_row_counts: list[int] = []
    original_fit = LogisticStaticModel.fit

    def spy(self, X, y=None, **kwargs):
        seen_row_counts.append(np.shape(X)[0])
        return original_fit(self, X, y, **kwargs)

    monkeypatch.setattr(LogisticStaticModel, "fit", spy)

    run_cross_validation(
        matrix.X,
        matrix.y,
        ids,
        manifest,
        lambda seed: LogisticStaticModel(random_state=seed),
    )

    assert seen_row_counts == [60]
