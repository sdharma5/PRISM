"""Calibration metrics must recognize a well-calibrated model and catch a bad one."""

from __future__ import annotations

import numpy as np
import pytest

from evaluation.calibration import (
    brier_score,
    calibration_curve_points,
    calibration_metrics_dict,
    calibration_report,
    calibration_slope_intercept,
    expected_calibration_error,
)
from evaluation.classification import aggregate_fold_metrics, classification_metrics


@pytest.fixture
def calibrated() -> tuple[np.ndarray, np.ndarray]:
    """Probabilities that are true by construction: outcomes drawn from them."""
    rng = np.random.default_rng(0)
    probabilities = rng.uniform(0.02, 0.98, size=4000)
    outcomes = (rng.uniform(size=4000) < probabilities).astype(float)
    return outcomes, probabilities


def test_brier_of_a_perfect_predictor_is_zero():
    y = np.array([0.0, 1.0, 1.0, 0.0])
    assert brier_score(y, y) == pytest.approx(0.0)


def test_brier_of_a_maximally_wrong_predictor_is_one():
    y = np.array([0.0, 1.0, 1.0, 0.0])
    assert brier_score(y, 1 - y) == pytest.approx(1.0)


def test_brier_of_a_base_rate_predictor(calibrated):
    y_true, _ = calibrated
    rate = float(y_true.mean())
    assert brier_score(y_true, np.full_like(y_true, rate)) == pytest.approx(
        rate * (1 - rate), abs=0.01
    )


def test_ece_is_near_zero_for_a_calibrated_model(calibrated):
    y_true, y_prob = calibrated
    assert expected_calibration_error(y_true, y_prob, n_bins=10) < 0.03


def test_ece_is_large_for_an_overconfident_model(calibrated):
    """Pushing probabilities to the extremes must be detected."""
    y_true, y_prob = calibrated
    overconfident = np.clip((y_prob - 0.5) * 4 + 0.5, 0.001, 0.999)
    assert expected_calibration_error(y_true, overconfident, n_bins=10) > 0.1


def test_calibration_slope_is_one_for_a_calibrated_model(calibrated):
    y_true, y_prob = calibrated
    slope, intercept = calibration_slope_intercept(y_true, y_prob)
    assert slope == pytest.approx(1.0, abs=0.12)
    assert intercept == pytest.approx(0.0, abs=0.12)


def test_calibration_slope_is_below_one_when_predictions_are_too_extreme(calibrated):
    y_true, y_prob = calibrated
    logits = np.log(y_prob / (1 - y_prob))
    too_extreme = 1.0 / (1.0 + np.exp(-2.5 * logits))
    slope, _ = calibration_slope_intercept(y_true, too_extreme)
    assert slope < 0.7


def test_calibration_slope_is_above_one_when_predictions_are_too_timid(calibrated):
    y_true, y_prob = calibrated
    logits = np.log(y_prob / (1 - y_prob))
    too_timid = 1.0 / (1.0 + np.exp(-0.4 * logits))
    slope, _ = calibration_slope_intercept(y_true, too_timid)
    assert slope > 1.5


def test_intercept_detects_systematic_over_prediction(calibrated):
    y_true, y_prob = calibrated
    logits = np.log(y_prob / (1 - y_prob))
    inflated = 1.0 / (1.0 + np.exp(-(logits + 1.5)))
    _, intercept = calibration_slope_intercept(y_true, inflated)
    assert intercept < -0.5


def test_calibration_curve_bins_are_ordered_and_populated(calibrated):
    y_true, y_prob = calibrated
    points = calibration_curve_points(y_true, y_prob, n_bins=10)
    assert len(points) == 10
    assert all(p["count"] > 0 for p in points)
    predicted = [p["mean_predicted"] for p in points]
    assert predicted == sorted(predicted)
    for point in points:
        assert abs(point["mean_predicted"] - point["observed_rate"]) < 0.12


def test_quantile_binning_produces_balanced_bins(calibrated):
    y_true, y_prob = calibrated
    points = calibration_curve_points(y_true, y_prob, n_bins=5, strategy="quantile")
    counts = [p["count"] for p in points]
    assert max(counts) - min(counts) <= 2


def test_report_returns_the_schema_contract(calibrated):
    y_true, y_prob = calibrated
    report = calibration_report(y_true, y_prob, n_bins=15)
    assert report.n_bins == 15
    assert 0.0 <= report.brier <= 1.0
    assert report.calibration_slope is not None
    assert report.ece is not None


def test_report_handles_empty_input():
    report = calibration_report([], [])
    assert report.calibration_slope is None
    assert report.calibration_intercept is None


def test_single_class_input_yields_undefined_slope():
    """With one class present a recalibration fit is meaningless, not zero."""
    y_true = np.zeros(50)
    y_prob = np.linspace(0.1, 0.9, 50)
    slope, intercept = calibration_slope_intercept(y_true, y_prob)
    assert np.isnan(slope) and np.isnan(intercept)


def test_metrics_dict_keys_match_the_experiment_contract(calibrated):
    y_true, y_prob = calibrated
    metrics = calibration_metrics_dict(y_true, y_prob)
    assert set(metrics) == {"brier", "ece", "calibration_slope", "calibration_intercept"}


def test_aggregate_across_folds_reports_mean_and_std():
    folds = [
        {"auroc": 0.70, "brier": 0.20},
        {"auroc": 0.80, "brier": 0.22},
        {"auroc": 0.75, "brier": float("nan")},
    ]
    mean, std = aggregate_fold_metrics(folds)
    assert mean["auroc"] == pytest.approx(0.75)
    assert std["auroc"] == pytest.approx(0.05)
    # A NaN fold is skipped rather than poisoning the aggregate.
    assert mean["brier"] == pytest.approx(0.21)


def test_classification_metrics_include_calibration_free_panel(calibrated):
    y_true, y_prob = calibrated
    metrics = classification_metrics(y_true, y_prob)
    for key in ("auroc", "auprc", "balanced_accuracy", "sensitivity", "specificity", "f1"):
        assert key in metrics and np.isfinite(metrics[key])
    assert metrics["auroc"] > 0.7
