"""Temporal convolutional encoder: an alternative to the GRU.

A dilated causal TCN is offered alongside the GRU because the two fail
differently. A GRU can quietly forget an event 20 days back; a TCN with
exponentially dilated kernels has a fixed, auditable receptive field, so you can
state exactly how many days of history influence today's state. On short
cycle-length windows with heavy missingness the TCN is often the better-behaved
of the two, and having both lets the choice be an empirical one.

Causality is strict: the encoder never sees a future day. Non-causal convolution
would leak tomorrow's hormone value into today's state estimate, which would
inflate every metric in this repository.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _causal_pad(x: np.ndarray, pad: int) -> np.ndarray:
    """Left-pad the time axis so no output timestep sees the future."""
    if pad <= 0:
        return x
    return np.pad(x, ((0, 0), (pad, 0), (0, 0)), mode="edge")


class NumpyTCN:
    """Dilated causal 1D convolutions in numpy, with fixed random filters.

    Like :class:`~models.temporal.gru.NumpyGRU` this is a random-feature encoder
    whose trained component lives in the heads. It exists so the TCN path is
    exercised in a torch-free environment rather than being dead code.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 32,
        *,
        n_levels: int = 3,
        kernel_size: int = 3,
        seed: int = 0,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_levels = n_levels
        self.kernel_size = kernel_size
        self.dilations = [2**level for level in range(n_levels)]
        self.filters: list[np.ndarray] = []
        channels = input_size
        for _ in range(n_levels):
            scale = 1.0 / np.sqrt(max(channels * kernel_size, 1))
            self.filters.append(rng.normal(0, scale, (kernel_size, channels, hidden_size)))
            channels = hidden_size

    @property
    def receptive_field(self) -> int:
        """Number of past days that can influence the final output."""
        return 1 + sum((self.kernel_size - 1) * d for d in self.dilations)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Run the stack over ``(N, T, F)`` and return the last step ``(N, H)``."""
        h = np.atleast_3d(np.asarray(x, dtype=float))
        for weights, dilation in zip(self.filters, self.dilations, strict=True):
            pad = (self.kernel_size - 1) * dilation
            padded = _causal_pad(h, pad)
            t_out = h.shape[1]
            out = np.zeros((h.shape[0], t_out, self.hidden_size))
            for k in range(self.kernel_size):
                offset = k * dilation
                out += padded[:, offset : offset + t_out, :] @ weights[k]
            h = np.tanh(out)
        return h[:, -1, :]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


class TorchTCN:
    """Lazy torch dilated causal TCN with the same interface."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 32,
        *,
        n_levels: int = 3,
        kernel_size: int = 3,
    ) -> None:
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_levels = n_levels
        self.kernel_size = kernel_size
        self._module: Any | None = None

    @staticmethod
    def is_available() -> bool:
        try:
            import torch  # noqa: F401, PLC0415
        except ImportError:
            return False
        return True

    @property
    def receptive_field(self) -> int:
        return 1 + sum((self.kernel_size - 1) * (2**level) for level in range(self.n_levels))

    def build(self) -> Any:
        """Instantiate and cache the underlying module."""
        if self._module is not None:
            return self._module
        try:
            from torch import nn  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError("TorchTCN requires the optional 'torch' extra.") from exc

        kernel_size, hidden = self.kernel_size, self.hidden_size
        n_levels = self.n_levels

        class _CausalBlock(nn.Module):
            """Dilated causal conv with the right-hand padding trimmed off."""

            def __init__(self, cin: int, cout: int, dilation: int) -> None:
                super().__init__()
                self.pad = (kernel_size - 1) * dilation
                self.conv = nn.Conv1d(cin, cout, kernel_size, padding=self.pad, dilation=dilation)
                self.act = nn.ReLU()

            def forward(self, x: Any) -> Any:
                out = self.conv(x)
                return self.act(out[:, :, : -self.pad] if self.pad else out)

        class _TCN(nn.Module):
            def __init__(self, input_size: int) -> None:
                super().__init__()
                blocks = []
                channels = input_size
                for level in range(n_levels):
                    blocks.append(_CausalBlock(channels, hidden, 2**level))
                    channels = hidden
                self.blocks = nn.Sequential(*blocks)

            def forward(self, x: Any) -> Any:
                # (N, T, F) -> (N, F, T) -> conv -> last timestep.
                return self.blocks(x.transpose(1, 2))[:, :, -1]

        self._module = _TCN(self.input_size)
        return self._module

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Return the final representation as ``(N, H)`` numpy."""
        import torch  # noqa: PLC0415

        module = self.build()
        module.eval()
        with torch.no_grad():
            out = module(torch.as_tensor(np.asarray(x), dtype=torch.float32))
        return out.numpy()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


def build_tcn(
    input_size: int,
    *,
    hidden_size: int = 32,
    backend: str = "auto",
    seed: int = 0,
    **kwargs: Any,
) -> NumpyTCN | TorchTCN:
    """Return a TCN encoder, preferring torch when installed."""
    if backend == "numpy":
        return NumpyTCN(input_size, hidden_size, seed=seed, **kwargs)
    if backend == "torch":
        return TorchTCN(input_size, hidden_size, **kwargs)
    if backend == "auto":
        return (
            TorchTCN(input_size, hidden_size, **kwargs)
            if TorchTCN.is_available()
            else NumpyTCN(input_size, hidden_size, seed=seed, **kwargs)
        )
    raise ValueError(f"Unknown backend '{backend}'.")
