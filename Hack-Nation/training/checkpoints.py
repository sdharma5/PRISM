"""Checkpointing models together with their fold-local preprocessing pipeline.

A model saved without the pipeline that produced its inputs is unusable: the
column order, the imputation medians and the scaling constants are all part of
the fitted artifact. They are therefore always written as one bundle.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from models.base import BasePrismModel

CHECKPOINT_SUFFIX = ".pkl"
METADATA_SUFFIX = ".meta.json"


@dataclass
class Checkpoint:
    """A model plus everything required to apply it to new rows."""

    model: BasePrismModel
    pipeline: Any = None
    feature_names: list[str] | None = None
    fold: int | None = None
    seed: int | None = None
    metrics: dict[str, float] | None = None
    created_at: str = ""

    def metadata(self) -> dict[str, Any]:
        return {
            "model_name": self.model.name,
            "model_version": self.model.version,
            "model_class": type(self.model).__name__,
            "pipeline_class": type(self.pipeline).__name__ if self.pipeline is not None else None,
            "feature_names": self.feature_names,
            "fold": self.fold,
            "seed": self.seed,
            "metrics": self.metrics or {},
            "created_at": self.created_at,
        }


def save_checkpoint(
    path: str | Path,
    model: BasePrismModel,
    *,
    pipeline: Any = None,
    feature_names: list[str] | None = None,
    fold: int | None = None,
    seed: int | None = None,
    metrics: dict[str, float] | None = None,
) -> Path:
    """Write ``<path>.pkl`` and a readable ``<path>.meta.json`` beside it."""
    path = Path(path)
    if path.suffix != CHECKPOINT_SUFFIX:
        path = path.with_suffix(CHECKPOINT_SUFFIX)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = Checkpoint(
        model=model,
        pipeline=pipeline,
        feature_names=feature_names,
        fold=fold,
        seed=seed,
        metrics=metrics,
        created_at=datetime.now(UTC).isoformat(),
    )
    with path.open("wb") as fh:
        pickle.dump(checkpoint, fh)

    meta_path = path.with_suffix("")
    meta_path = meta_path.with_name(meta_path.name + METADATA_SUFFIX)
    meta_path.write_text(json.dumps(checkpoint.metadata(), indent=2, default=str) + "\n")
    return path


def load_checkpoint(path: str | Path) -> Checkpoint:
    """Load a bundle written by :func:`save_checkpoint`."""
    path = Path(path)
    with path.open("rb") as fh:
        checkpoint = pickle.load(fh)
    if not isinstance(checkpoint, Checkpoint):
        raise TypeError(f"{path} does not contain a PRISM Checkpoint.")
    return checkpoint


def checkpoint_dir(experiment_dir: str | Path) -> Path:
    """The conventional per-experiment checkpoint subdirectory."""
    path = Path(experiment_dir) / "checkpoints"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_fold_checkpoints(
    experiment_dir: str | Path,
    fold_results: list[Any],
) -> list[Path]:
    """Persist whichever folds retained their fitted model."""
    target = checkpoint_dir(experiment_dir)
    written: list[Path] = []
    for result in fold_results:
        model = getattr(result, "model", None)
        if model is None:
            continue
        written.append(
            save_checkpoint(
                target / f"fold_{result.fold:03d}",
                model,
                pipeline=getattr(result, "pipeline", None),
                fold=result.fold,
                seed=getattr(result, "seed", None),
                metrics=getattr(result, "metrics", None),
            )
        )
    return written


__all__ = [
    "Checkpoint",
    "checkpoint_dir",
    "load_checkpoint",
    "save_checkpoint",
    "save_fold_checkpoints",
]
