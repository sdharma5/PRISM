"""Training callbacks: observation hooks that never alter the fold loop's logic.

Keeping logging, timing and early stopping out of ``engine.py`` keeps the leakage
boundary in that file easy to read and audit.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import numpy as np


class Callback:
    """No-op base. Subclasses override only the hooks they care about."""

    def on_experiment_start(self, context: dict[str, Any]) -> None:
        """Called once, before the first fold."""

    def on_fold_start(self, fold: int, context: dict[str, Any]) -> None:
        """Called before a fold's preprocessing is fitted."""

    def on_fold_end(self, fold: int, metrics: dict[str, float]) -> None:
        """Called after a fold's metrics are computed."""

    def on_epoch_end(self, epoch: int, metrics: dict[str, float]) -> None:
        """Called by iterative models that report per-epoch progress."""

    def on_experiment_end(self, context: dict[str, Any]) -> None:
        """Called once, after the last fold."""


class CallbackList(Callback):
    """Fan a single hook call out to several callbacks."""

    def __init__(self, callbacks: Sequence[Callback] | None = None) -> None:
        self.callbacks: list[Callback] = list(callbacks or [])

    def append(self, callback: Callback) -> None:
        self.callbacks.append(callback)

    def on_experiment_start(self, context: dict[str, Any]) -> None:
        for cb in self.callbacks:
            cb.on_experiment_start(context)

    def on_fold_start(self, fold: int, context: dict[str, Any]) -> None:
        for cb in self.callbacks:
            cb.on_fold_start(fold, context)

    def on_fold_end(self, fold: int, metrics: dict[str, float]) -> None:
        for cb in self.callbacks:
            cb.on_fold_end(fold, metrics)

    def on_epoch_end(self, epoch: int, metrics: dict[str, float]) -> None:
        for cb in self.callbacks:
            cb.on_epoch_end(epoch, metrics)

    def on_experiment_end(self, context: dict[str, Any]) -> None:
        for cb in self.callbacks:
            cb.on_experiment_end(context)


class JsonlLoggingCallback(Callback):
    """Write every hook into the experiment's ``training_log.jsonl``."""

    def __init__(self, logger: Any) -> None:
        self.logger = logger

    def on_experiment_start(self, context: dict[str, Any]) -> None:
        self.logger.log("experiment_start", **context)

    def on_fold_start(self, fold: int, context: dict[str, Any]) -> None:
        self.logger.log("fold_start", fold=fold, **context)

    def on_fold_end(self, fold: int, metrics: dict[str, float]) -> None:
        self.logger.log("fold_end", fold=fold, metrics=metrics)

    def on_epoch_end(self, epoch: int, metrics: dict[str, float]) -> None:
        self.logger.log("epoch_end", epoch=epoch, metrics=metrics)

    def on_experiment_end(self, context: dict[str, Any]) -> None:
        self.logger.log("experiment_end", **context)


class TimingCallback(Callback):
    """Record wall-clock seconds per fold — cheap, and it catches pathological runs."""

    def __init__(self) -> None:
        self.fold_seconds: dict[int, float] = {}
        self._started: dict[int, float] = {}
        self.total_seconds: float = 0.0
        self._experiment_start: float | None = None

    def on_experiment_start(self, context: dict[str, Any]) -> None:
        self._experiment_start = time.perf_counter()

    def on_fold_start(self, fold: int, context: dict[str, Any]) -> None:
        self._started[fold] = time.perf_counter()

    def on_fold_end(self, fold: int, metrics: dict[str, float]) -> None:
        start = self._started.pop(fold, None)
        if start is not None:
            self.fold_seconds[fold] = time.perf_counter() - start

    def on_experiment_end(self, context: dict[str, Any]) -> None:
        if self._experiment_start is not None:
            self.total_seconds = time.perf_counter() - self._experiment_start


class EarlyStopping(Callback):
    """Stop when a monitored metric stops improving.

    Only ever driven by a *validation split carved out of the training fold* —
    monitoring the test fold would turn early stopping into leakage.
    """

    def __init__(
        self,
        monitor: str = "loss",
        *,
        patience: int = 10,
        mode: str = "min",
        min_delta: float = 0.0,
    ) -> None:
        if mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'.")
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best: float = np.inf if mode == "min" else -np.inf
        self.best_epoch: int = -1
        self.wait: int = 0
        self.should_stop: bool = False

    def _improved(self, value: float) -> bool:
        if self.mode == "min":
            return value < self.best - self.min_delta
        return value > self.best + self.min_delta

    def on_epoch_end(self, epoch: int, metrics: dict[str, float]) -> None:
        value = metrics.get(self.monitor)
        if value is None or not np.isfinite(value):
            return
        if self._improved(float(value)):
            self.best, self.best_epoch, self.wait = float(value), epoch, 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.should_stop = True

    def reset(self) -> None:
        self.best = np.inf if self.mode == "min" else -np.inf
        self.best_epoch, self.wait, self.should_stop = -1, 0, False


__all__ = [
    "Callback",
    "CallbackList",
    "EarlyStopping",
    "JsonlLoggingCallback",
    "TimingCallback",
]
