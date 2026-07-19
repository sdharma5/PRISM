"""Patient-level splitting. Every strategy returns a saveable ``SplitManifest``.

Splits are *always* by patient id. Splitting by row would put two rows from the
same person on both sides of the evaluation boundary, which inflates every metric
downstream and is invisible in the results.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, RepeatedStratifiedKFold, train_test_split

from schemas.patient import SplitManifest


def _patient_level_frame(
    patient_ids: Sequence[str],
    y: Sequence[float] | None = None,
    groups: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Collapse rows to one record per patient.

    The stratification label is the patient's max label: if a patient has any
    positive row they count as positive for balancing purposes.
    """
    frame = pd.DataFrame({"patient_id": [str(p) for p in patient_ids]})
    if y is not None:
        frame["y"] = pd.to_numeric(pd.Series(list(y)), errors="coerce").to_numpy()
    if groups is not None:
        frame["group"] = [str(g) for g in groups]

    agg: dict[str, str] = {}
    if y is not None:
        agg["y"] = "max"
    if groups is not None:
        agg["group"] = "first"

    if not agg:
        return frame.drop_duplicates("patient_id").reset_index(drop=True)
    collapsed = frame.groupby("patient_id", as_index=False).agg(agg)
    return collapsed


def _finalize(manifest: SplitManifest) -> SplitManifest:
    """Every manifest leaves this module already proven disjoint."""
    manifest.assert_disjoint()
    return manifest


def make_repeated_stratified_kfold_manifest(
    patient_ids: Sequence[str],
    y: Sequence[float],
    *,
    manifest_id: str,
    dataset_id: str,
    dataset_version: str = "unknown",
    n_splits: int = 5,
    n_repeats: int = 1,
    seeds: Sequence[int] = (0,),
    holdout_ids: Sequence[str] | None = None,
) -> SplitManifest:
    """Repeated stratified K-fold over unique patients, repeated across seeds."""
    holdout = {str(p) for p in (holdout_ids or [])}
    people = _patient_level_frame(patient_ids, y)
    people = people[~people["patient_id"].isin(holdout)].reset_index(drop=True)

    ids = people["patient_id"].to_numpy()
    labels = people["y"].fillna(0).astype(int).to_numpy()
    _check_stratification(labels, n_splits)

    folds: list[dict[str, list[str]]] = []
    for seed in seeds:
        splitter = RepeatedStratifiedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=int(seed)
        )
        for train_idx, test_idx in splitter.split(ids.reshape(-1, 1), labels):
            folds.append(
                {"train": sorted(ids[train_idx].tolist()), "test": sorted(ids[test_idx].tolist())}
            )

    return _finalize(
        SplitManifest(
            manifest_id=manifest_id,
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            strategy="repeated_stratified_kfold",
            n_splits=n_splits,
            seeds=[int(s) for s in seeds],
            folds=folds,
            holdout_ids=sorted(holdout),
        )
    )


def make_grouped_kfold_manifest(
    patient_ids: Sequence[str],
    groups: Sequence[str],
    *,
    manifest_id: str,
    dataset_id: str,
    dataset_version: str = "unknown",
    n_splits: int = 5,
    seeds: Sequence[int] = (0,),
    holdout_ids: Sequence[str] | None = None,
) -> SplitManifest:
    """Grouped K-fold, so an entire site/family/batch is held out together."""
    holdout = {str(p) for p in (holdout_ids or [])}
    people = _patient_level_frame(patient_ids, groups=groups)
    people = people[~people["patient_id"].isin(holdout)].reset_index(drop=True)

    ids = people["patient_id"].to_numpy()
    group_labels = people["group"].to_numpy()
    n_groups = len(set(group_labels.tolist()))
    if n_groups < n_splits:
        raise ValueError(
            f"grouped_kfold needs at least n_splits={n_splits} groups, found {n_groups}."
        )

    splitter = GroupKFold(n_splits=n_splits)
    folds = [
        {"train": sorted(ids[train_idx].tolist()), "test": sorted(ids[test_idx].tolist())}
        for train_idx, test_idx in splitter.split(ids.reshape(-1, 1), groups=group_labels)
    ]

    return _finalize(
        SplitManifest(
            manifest_id=manifest_id,
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            strategy="grouped_kfold",
            n_splits=n_splits,
            seeds=[int(s) for s in seeds],
            folds=folds,
            holdout_ids=sorted(holdout),
        )
    )


def make_leave_one_participant_out_manifest(
    patient_ids: Sequence[str],
    *,
    manifest_id: str,
    dataset_id: str,
    dataset_version: str = "unknown",
    seeds: Sequence[int] = (0,),
    holdout_ids: Sequence[str] | None = None,
    max_participants: int | None = None,
) -> SplitManifest:
    """One fold per participant — the standard protocol for small longitudinal cohorts."""
    holdout = {str(p) for p in (holdout_ids or [])}
    ids = sorted({str(p) for p in patient_ids} - holdout)
    ids_used = ids[:max_participants] if max_participants is not None else ids

    folds = [{"train": [p for p in ids if p != held], "test": [held]} for held in ids_used]

    return _finalize(
        SplitManifest(
            manifest_id=manifest_id,
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            strategy="leave_one_participant_out",
            n_splits=len(folds),
            seeds=[int(s) for s in seeds],
            folds=folds,
            holdout_ids=sorted(holdout),
        )
    )


def make_holdout_manifest(
    patient_ids: Sequence[str],
    y: Sequence[float] | None = None,
    *,
    manifest_id: str,
    dataset_id: str,
    dataset_version: str = "unknown",
    test_size: float = 0.2,
    seed: int = 0,
    holdout_ids: Sequence[str] | None = None,
) -> SplitManifest:
    """A single stratified train/test split at patient level."""
    holdout = {str(p) for p in (holdout_ids or [])}
    people = _patient_level_frame(patient_ids, y)
    people = people[~people["patient_id"].isin(holdout)].reset_index(drop=True)

    ids = people["patient_id"].to_numpy()
    stratify = None
    if "y" in people.columns:
        labels = people["y"].fillna(0).astype(int).to_numpy()
        if len(np.unique(labels)) > 1 and np.bincount(labels).min() >= 2:
            stratify = labels

    train_ids, test_ids = train_test_split(
        ids, test_size=test_size, random_state=int(seed), stratify=stratify
    )

    return _finalize(
        SplitManifest(
            manifest_id=manifest_id,
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            strategy="holdout",
            n_splits=1,
            seeds=[int(seed)],
            folds=[{"train": sorted(train_ids.tolist()), "test": sorted(test_ids.tolist())}],
            holdout_ids=sorted(holdout),
        )
    )


def reserve_holdout_patients(
    patient_ids: Sequence[str],
    y: Sequence[float] | None = None,
    *,
    fraction: float = 0.15,
    seed: int = 0,
) -> list[str]:
    """Carve off an untouched holdout set *before* any cross-validation is built."""
    if fraction <= 0:
        return []
    people = _patient_level_frame(patient_ids, y)
    ids = people["patient_id"].to_numpy()
    stratify = None
    if "y" in people.columns:
        labels = people["y"].fillna(0).astype(int).to_numpy()
        if len(np.unique(labels)) > 1 and np.bincount(labels).min() >= 2:
            stratify = labels
    _, held = train_test_split(ids, test_size=fraction, random_state=int(seed), stratify=stratify)
    return sorted(held.tolist())


def _check_stratification(labels: np.ndarray, n_splits: int) -> None:
    """Refuse to build folds that cannot contain both classes."""
    classes, counts = np.unique(labels, return_counts=True)
    if len(classes) < 2:
        raise ValueError("Stratified splitting requires at least two label classes.")
    if counts.min() < n_splits:
        raise ValueError(
            f"Cannot build {n_splits} stratified folds: the rarest class has only "
            f"{int(counts.min())} patient(s)."
        )


def build_split_manifest(
    strategy: str,
    patient_ids: Sequence[str],
    y: Sequence[float] | None = None,
    groups: Sequence[str] | None = None,
    **kwargs: object,
) -> SplitManifest:
    """Config-driven entry point used by the training scripts."""
    if strategy == "repeated_stratified_kfold":
        if y is None:
            raise ValueError("repeated_stratified_kfold requires labels.")
        return make_repeated_stratified_kfold_manifest(patient_ids, y, **kwargs)  # type: ignore[arg-type]
    if strategy == "grouped_kfold":
        if groups is None:
            raise ValueError("grouped_kfold requires a grouping column.")
        return make_grouped_kfold_manifest(patient_ids, groups, **kwargs)  # type: ignore[arg-type]
    if strategy == "leave_one_participant_out":
        return make_leave_one_participant_out_manifest(patient_ids, **kwargs)  # type: ignore[arg-type]
    if strategy == "holdout":
        return make_holdout_manifest(patient_ids, y, **kwargs)  # type: ignore[arg-type]
    raise ValueError(f"Unknown split strategy '{strategy}'.")


def save_split_manifest(manifest: SplitManifest, path: str | Path) -> Path:
    """Persist a manifest, re-checking disjointness on the way out."""
    manifest.assert_disjoint()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest.model_dump(mode="json"), indent=2) + "\n")
    return path


def load_split_manifest(path: str | Path) -> SplitManifest:
    """Load a manifest, re-checking disjointness on the way in."""
    manifest = SplitManifest.model_validate(json.loads(Path(path).read_text()))
    manifest.assert_disjoint()
    return manifest


def fold_row_indices(
    fold: dict[str, list[str]],
    patient_ids: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Map a fold's patient ids onto row positions in the feature matrix."""
    series = pd.Series([str(p) for p in patient_ids])
    train = series.isin(set(fold.get("train", []))).to_numpy().nonzero()[0]
    test = series.isin(set(fold.get("test", []))).to_numpy().nonzero()[0]
    overlap = set(train.tolist()) & set(test.tolist())
    if overlap:
        raise ValueError(f"{len(overlap)} row(s) resolved into both train and test.")
    return train, test


__all__ = [
    "build_split_manifest",
    "fold_row_indices",
    "load_split_manifest",
    "make_grouped_kfold_manifest",
    "make_holdout_manifest",
    "make_leave_one_participant_out_manifest",
    "make_repeated_stratified_kfold_manifest",
    "reserve_holdout_patients",
    "save_split_manifest",
]
