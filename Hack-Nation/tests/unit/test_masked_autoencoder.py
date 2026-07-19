"""The masked autoencoder must earn its place against mean imputation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features.static_features import build_static_features, value_columns_of
from models.tabular.masked_autoencoder import MaskedTabularAutoencoder
from tests.fixtures.synthetic_tabular import make_synthetic_cohort


@pytest.fixture(scope="module")
def matrices() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = make_synthetic_cohort(n=240, seed=4, missing_rate=0.2)
    matrix = build_static_features(df, add_missingness_indicators=False)
    X = matrix.X[value_columns_of(matrix.X)]
    return X.iloc[:180], X.iloc[180:]


@pytest.fixture(scope="module")
def fitted(matrices) -> MaskedTabularAutoencoder:
    train, _ = matrices
    return MaskedTabularAutoencoder(latent_dim=12, hidden_dim=48, epochs=150, random_state=0).fit(
        train
    )


def test_beats_mean_imputation_on_withheld_values(fitted, matrices):
    """The whole point: recovering hidden values better than guessing the mean."""
    _, test = matrices
    metrics = fitted.evaluate(test, seed=123)

    assert metrics["masked_reconstruction_mse"] < metrics["mean_imputation_mse"]
    assert metrics["beats_mean_imputation"] == 1.0
    assert metrics["mse_improvement_over_mean"] > 0.02
    assert metrics["n_masked"] > 0


def test_beats_mean_imputation_on_training_data_too(fitted, matrices):
    train, _ = matrices
    metrics = fitted.evaluate(train, seed=7)
    assert metrics["masked_reconstruction_mse"] < metrics["mean_imputation_mse"]


def test_embedding_has_the_requested_shape(fitted, matrices):
    _, test = matrices
    embedding = fitted.embed(test)
    assert embedding.shape == (len(test), fitted.latent_dim)
    assert np.isfinite(embedding).all()


def test_embedding_is_deterministic(fitted, matrices):
    _, test = matrices
    assert np.allclose(fitted.embed(test), fitted.embed(test))


def test_embedding_varies_between_patients(fitted, matrices):
    """A collapsed embedding would be useless downstream, so check it moved."""
    _, test = matrices
    embedding = fitted.embed(test)
    assert float(embedding.std(axis=0).mean()) > 1e-3


def test_training_is_reproducible_under_a_fixed_seed(matrices):
    train, test = matrices
    kwargs = {"latent_dim": 8, "hidden_dim": 32, "epochs": 30, "random_state": 5}
    a = MaskedTabularAutoencoder(**kwargs).fit(train)
    b = MaskedTabularAutoencoder(**kwargs).fit(train)
    assert np.allclose(a.embed(test), b.embed(test))


def test_training_loss_decreases(fitted):
    history = fitted.history_
    assert len(history) > 10
    early = np.mean([h["masked_mse"] for h in history[:5]])
    late = np.mean([h["masked_mse"] for h in history[-5:]])
    assert late < early


def test_does_not_require_torch(monkeypatch, matrices):
    """Torch is optional; the model must train with it entirely unavailable."""
    import builtins

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("torch is unavailable in this test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)

    train, test = matrices
    model = MaskedTabularAutoencoder(latent_dim=6, hidden_dim=16, epochs=20).fit(train)
    assert model.is_fitted
    assert model.embed(test).shape == (len(test), 6)


def test_reconstruction_returns_original_scale(fitted, matrices):
    _, test = matrices
    reconstructed = fitted.reconstruct(test, original_scale=True)
    assert reconstructed.shape == test.shape
    assert np.isfinite(reconstructed).all()

    observed = np.isfinite(test.to_numpy(dtype=float))
    columns = test.columns.tolist()
    bmi_index = columns.index("bmi")
    if observed[:, bmi_index].any():
        values = reconstructed[observed[:, bmi_index], bmi_index]
        # Original units, not z-scores: BMI should land in a plausible band.
        assert 10 < float(np.median(values)) < 60


def test_missing_values_are_not_treated_as_measured_zeros():
    """A column of all-NaN must not masquerade as a column of zeros."""
    X = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "never_measured": [np.nan] * 6,
        }
    )
    model = MaskedTabularAutoencoder(latent_dim=2, hidden_dim=4, epochs=5).fit(X)
    assert model.stds_[1] == 1.0
    assert model.means_[1] == 0.0
    assert np.isfinite(model.embed(X)).all()


def test_fit_rejects_a_fully_unobserved_matrix():
    X = pd.DataFrame({"a": [np.nan, np.nan], "b": [np.nan, np.nan]})
    with pytest.raises(ValueError, match="no observed values"):
        MaskedTabularAutoencoder(epochs=2).fit(X)


def test_predict_before_fit_raises():
    model = MaskedTabularAutoencoder()
    with pytest.raises(RuntimeError, match="fit()"):
        model.embed(pd.DataFrame({"a": [1.0]}))


def test_invalid_mask_rate_range_is_rejected():
    with pytest.raises(ValueError, match="mask_rate_range"):
        MaskedTabularAutoencoder(mask_rate_range=(0.5, 0.2))


def test_save_and_load_round_trip(tmp_path, fitted, matrices):
    _, test = matrices
    path = fitted.save(tmp_path / "ae.pkl")
    reloaded = MaskedTabularAutoencoder.load(path)
    assert np.allclose(reloaded.embed(test), fitted.embed(test))
    assert (tmp_path / "ae.pkl.json").exists()
