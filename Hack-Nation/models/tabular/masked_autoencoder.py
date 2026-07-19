"""A masked/denoising tabular autoencoder with a numpy-only training loop.

Why not torch: this model must run in every environment the project supports, and
a two-layer autoencoder over a few dozen columns does not need a deep-learning
framework. The forward/backward passes below are explicit, which also makes the
masking semantics auditable.

Why masking rather than plain reconstruction: hormonal panels are missing
non-randomly. Training the network to *recover deliberately hidden observed
values* forces it to learn the correlation structure between variables instead of
memorizing an identity map, and it gives an honest evaluation target — masked
reconstruction error against a mean-imputation baseline.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from models.base import ArrayLike, BasePrismModel

_EPS = 1e-8


class MaskedTabularAutoencoder(BasePrismModel):
    """Denoising autoencoder over standardized tabular features.

    ``embed`` exposes the latent vector used as the static-clinical patient
    embedding downstream.
    """

    name = "tabular_masked_autoencoder"
    version = "0.1.0"
    is_classifier = False

    def __init__(
        self,
        *,
        latent_dim: int = 16,
        hidden_dim: int | None = 64,
        mask_rate_range: tuple[float, float] = (0.1, 0.3),
        epochs: int = 200,
        batch_size: int = 32,
        learning_rate: float = 3e-3,
        weight_decay: float = 1e-4,
        random_state: int = 0,
        activation: str = "tanh",
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        low, high = mask_rate_range
        if not 0.0 < low <= high < 1.0:
            raise ValueError("mask_rate_range must satisfy 0 < low <= high < 1.")
        super().__init__(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            mask_rate_range=[low, high],
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            random_state=random_state,
            activation=activation,
            **kwargs,
        )
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.mask_rate_range = (low, high)
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.random_state = random_state
        self.activation = activation
        self.verbose = verbose

        self.means_: np.ndarray | None = None
        self.stds_: np.ndarray | None = None
        self.weights_: dict[str, np.ndarray] = {}
        self.history_: list[dict[str, float]] = []

    # -- Standardization ---------------------------------------------------

    def _standardize(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (standardized values with NaN preserved, observed mask)."""
        if self.means_ is None or self.stds_ is None:
            raise RuntimeError(f"{self.name}: fit() must run before standardizing.")
        observed = np.isfinite(X)
        Z = np.where(observed, (X - self.means_) / self.stds_, np.nan)
        return Z, observed

    # -- Network -----------------------------------------------------------

    def _activate(self, A: np.ndarray) -> np.ndarray:
        if self.activation == "relu":
            return np.maximum(A, 0.0)
        return np.tanh(A)

    def _activate_grad(self, A: np.ndarray, H: np.ndarray) -> np.ndarray:
        if self.activation == "relu":
            return (A > 0).astype(float)
        return 1.0 - H**2

    def _init_weights(self, d: int, rng: np.random.Generator) -> None:
        """Xavier-ish init; input is (values, observed-mask) so it is 2*d wide."""
        hidden = self.hidden_dim or self.latent_dim
        sizes = [2 * d, hidden, self.latent_dim, hidden, d]
        self.weights_ = {}
        for i in range(len(sizes) - 1):
            fan_in, fan_out = sizes[i], sizes[i + 1]
            scale = np.sqrt(2.0 / (fan_in + fan_out))
            self.weights_[f"W{i}"] = rng.normal(scale=scale, size=(fan_in, fan_out))
            self.weights_[f"b{i}"] = np.zeros(fan_out)

    def _forward(self, X_in: np.ndarray, mask_in: np.ndarray) -> dict[str, np.ndarray]:
        """Forward pass. Returns every intermediate the backward pass needs."""
        W, cache = self.weights_, {}
        cache["input"] = np.hstack([X_in, mask_in])
        A0 = cache["input"] @ W["W0"] + W["b0"]
        H0 = self._activate(A0)
        A1 = H0 @ W["W1"] + W["b1"]
        Z = self._activate(A1)  # latent
        A2 = Z @ W["W2"] + W["b2"]
        H2 = self._activate(A2)
        out = H2 @ W["W3"] + W["b3"]
        cache.update({"A0": A0, "H0": H0, "A1": A1, "Z": Z, "A2": A2, "H2": H2, "out": out})
        return cache

    def _backward(
        self, cache: dict[str, np.ndarray], target: np.ndarray, loss_mask: np.ndarray
    ) -> dict[str, np.ndarray]:
        """Gradients of the masked MSE, computed only over supervised entries."""
        W = self.weights_
        n_supervised = max(float(loss_mask.sum()), 1.0)

        d_out = 2.0 * (cache["out"] - target) * loss_mask / n_supervised

        grads: dict[str, np.ndarray] = {}
        grads["W3"] = cache["H2"].T @ d_out
        grads["b3"] = d_out.sum(axis=0)

        d_H2 = d_out @ W["W3"].T
        d_A2 = d_H2 * self._activate_grad(cache["A2"], cache["H2"])
        grads["W2"] = cache["Z"].T @ d_A2
        grads["b2"] = d_A2.sum(axis=0)

        d_Z = d_A2 @ W["W2"].T
        d_A1 = d_Z * self._activate_grad(cache["A1"], cache["Z"])
        grads["W1"] = cache["H0"].T @ d_A1
        grads["b1"] = d_A1.sum(axis=0)

        d_H0 = d_A1 @ W["W1"].T
        d_A0 = d_H0 * self._activate_grad(cache["A0"], cache["H0"])
        grads["W0"] = cache["input"].T @ d_A0
        grads["b0"] = d_A0.sum(axis=0)
        return grads

    # -- Training ----------------------------------------------------------

    def fit(
        self, X: ArrayLike, y: ArrayLike | None = None, **kwargs: Any
    ) -> MaskedTabularAutoencoder:
        """Fit on training rows only. ``y`` is ignored — this model is unsupervised."""
        X_arr = self._record_features(X)
        rng = np.random.default_rng(self.random_state)

        observed = np.isfinite(X_arr)
        if not observed.any():
            raise ValueError(f"{self.name}: the training matrix has no observed values.")

        # Column statistics from observed entries only: a mean over silently
        # zero-filled cells would be a different number entirely. Computed by hand
        # rather than with nanmean so an entirely-unobserved column is a defined
        # case (mean 0, std 1) instead of a warning and a NaN.
        counts = observed.sum(axis=0).astype(float)
        safe_counts = np.maximum(counts, 1.0)
        values = np.where(observed, X_arr, 0.0)
        means = values.sum(axis=0) / safe_counts
        variances = ((np.where(observed, X_arr - means, 0.0)) ** 2).sum(axis=0) / safe_counts
        stds = np.sqrt(variances)
        self.means_ = np.where(counts > 0, means, 0.0)
        self.stds_ = np.where((counts > 1) & (stds > _EPS), stds, 1.0)

        Z, observed = self._standardize(X_arr)
        n, d = Z.shape
        self._init_weights(d, rng)

        adam = {k: (np.zeros_like(v), np.zeros_like(v)) for k, v in self.weights_.items()}
        beta1, beta2 = 0.9, 0.999
        step = 0
        self.history_ = []

        for epoch in range(self.epochs):
            order = rng.permutation(n)
            epoch_loss, n_batches = 0.0, 0

            for start in range(0, n, self.batch_size):
                batch = order[start : start + self.batch_size]
                Z_b, obs_b = Z[batch], observed[batch]
                if not obs_b.any():
                    continue

                # Hide 10-30% of the observed entries; those are the only cells the
                # loss is computed on, so the network cannot cheat by copying input.
                rate = rng.uniform(*self.mask_rate_range)
                hide = obs_b & (rng.random(obs_b.shape) < rate)
                if not hide.any():
                    continue

                visible = obs_b & ~hide
                # Unobserved and hidden cells enter the network as 0 (= the training
                # mean in standardized space) *accompanied by* their mask channel,
                # so "missing" is never confused with "measured zero".
                X_in = np.where(visible, np.nan_to_num(Z_b, nan=0.0), 0.0)
                mask_in = visible.astype(float)
                target = np.nan_to_num(Z_b, nan=0.0)
                loss_mask = hide.astype(float)

                cache = self._forward(X_in, mask_in)
                loss = float(
                    (((cache["out"] - target) ** 2) * loss_mask).sum() / max(loss_mask.sum(), 1.0)
                )
                grads = self._backward(cache, target, loss_mask)

                step += 1
                for key, grad in grads.items():
                    if key.startswith("W"):
                        grad = grad + self.weight_decay * self.weights_[key]
                    m, v = adam[key]
                    m = beta1 * m + (1 - beta1) * grad
                    v = beta2 * v + (1 - beta2) * grad**2
                    adam[key] = (m, v)
                    m_hat = m / (1 - beta1**step)
                    v_hat = v / (1 - beta2**step)
                    self.weights_[key] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + _EPS)

                epoch_loss += loss
                n_batches += 1

            if n_batches:
                record = {"epoch": float(epoch), "masked_mse": epoch_loss / n_batches}
                self.history_.append(record)
                if self.verbose and epoch % 20 == 0:
                    print(f"epoch {epoch:4d}  masked_mse={record['masked_mse']:.4f}")

        self.is_fitted = True
        self.training_metadata_ = {
            "n_train": int(n),
            "n_features": int(d),
            "latent_dim": int(self.latent_dim),
            "final_masked_mse": self.history_[-1]["masked_mse"] if self.history_ else float("nan"),
            "observed_fraction": float(observed.mean()),
        }
        return self

    # -- Inference ---------------------------------------------------------

    def _encode_inputs(self, X_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        Z, observed = self._standardize(X_arr)
        X_in = np.where(observed, np.nan_to_num(Z, nan=0.0), 0.0)
        return Z, observed, X_in

    def embed(self, X: ArrayLike) -> np.ndarray:
        """Latent patient embedding, shape ``(n, latent_dim)``."""
        self._require_fitted()
        X_arr = self._as_array(X)
        _, observed, X_in = self._encode_inputs(X_arr)
        return self._forward(X_in, observed.astype(float))["Z"]

    def reconstruct(self, X: ArrayLike, *, original_scale: bool = True) -> np.ndarray:
        """Reconstruct every column, including the ones that were never observed."""
        self._require_fitted()
        X_arr = self._as_array(X)
        _, observed, X_in = self._encode_inputs(X_arr)
        out = self._forward(X_in, observed.astype(float))["out"]
        if original_scale:
            return out * self.stds_ + self.means_
        return out

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Reconstruction in the original units — this model has no class output."""
        return self.reconstruct(X, original_scale=True)

    def evaluate(
        self,
        X: ArrayLike,
        y: ArrayLike | None = None,
        *,
        mask_rate: float = 0.2,
        seed: int | None = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Hide observed values, reconstruct them, and compare to mean imputation.

        The mean-imputation baseline is the honest comparator: if the autoencoder
        cannot beat "guess the training mean", its embedding carries no structure
        worth propagating downstream.
        """
        self._require_fitted()
        X_arr = self._as_array(X)
        rng = np.random.default_rng(self.random_state if seed is None else seed)

        Z, observed = self._standardize(X_arr)
        hide = observed & (rng.random(observed.shape) < mask_rate)
        if not hide.any():
            return {"masked_reconstruction_mse": float("nan"), "n_masked": 0.0}

        visible = observed & ~hide
        X_in = np.where(visible, np.nan_to_num(Z, nan=0.0), 0.0)
        out = self._forward(X_in, visible.astype(float))["out"]

        target = np.nan_to_num(Z, nan=0.0)
        n_masked = float(hide.sum())
        model_mse = float((((out - target) ** 2) * hide).sum() / n_masked)
        # Mean imputation predicts the training mean, which is 0 after standardizing.
        baseline_mse = float(((target**2) * hide).sum() / n_masked)

        improvement = (baseline_mse - model_mse) / baseline_mse if baseline_mse > _EPS else 0.0
        return {
            "masked_reconstruction_mse": model_mse,
            "mean_imputation_mse": baseline_mse,
            "mse_improvement_over_mean": float(improvement),
            "beats_mean_imputation": float(model_mse < baseline_mse),
            "masked_reconstruction_rmse": float(np.sqrt(model_mse)),
            "n_masked": n_masked,
            "mask_rate": float(mask_rate),
        }


__all__ = ["MaskedTabularAutoencoder"]
