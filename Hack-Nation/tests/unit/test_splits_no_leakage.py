"""Splits must never place a patient on both sides of the evaluation boundary."""

from __future__ import annotations

import pytest

from schemas.patient import SplitManifest
from tests.fixtures.synthetic_tabular import make_synthetic_cohort_with_groups
from training.splits import (
    build_split_manifest,
    fold_row_indices,
    load_split_manifest,
    make_grouped_kfold_manifest,
    make_holdout_manifest,
    make_leave_one_participant_out_manifest,
    make_repeated_stratified_kfold_manifest,
    reserve_holdout_patients,
    save_split_manifest,
)


@pytest.fixture
def cohort():
    return make_synthetic_cohort_with_groups(n=120, seed=3)


def test_repeated_stratified_kfold_is_patient_disjoint(cohort):
    manifest = make_repeated_stratified_kfold_manifest(
        cohort["patient_id"],
        cohort["pmos_binary"],
        manifest_id="m",
        dataset_id="synthetic",
        n_splits=5,
        seeds=(0, 1),
    )
    assert manifest.strategy == "repeated_stratified_kfold"
    assert len(manifest.folds) == 10
    manifest.assert_disjoint()
    for fold in manifest.folds:
        assert not set(fold["train"]) & set(fold["test"])


def test_every_patient_is_tested_exactly_once_per_seed(cohort):
    manifest = make_repeated_stratified_kfold_manifest(
        cohort["patient_id"],
        cohort["pmos_binary"],
        manifest_id="m",
        dataset_id="synthetic",
        n_splits=4,
        seeds=(0,),
    )
    tested = [pid for fold in manifest.folds for pid in fold["test"]]
    assert sorted(tested) == sorted(cohort["patient_id"].tolist())


def test_grouped_kfold_keeps_whole_groups_together(cohort):
    manifest = make_grouped_kfold_manifest(
        cohort["patient_id"],
        cohort["site"],
        manifest_id="m",
        dataset_id="synthetic",
        n_splits=3,
    )
    site_of = dict(zip(cohort["patient_id"], cohort["site"], strict=True))
    for fold in manifest.folds:
        train_sites = {site_of[p] for p in fold["train"]}
        test_sites = {site_of[p] for p in fold["test"]}
        assert not train_sites & test_sites


def test_leave_one_participant_out_holds_out_one_person(cohort):
    manifest = make_leave_one_participant_out_manifest(
        cohort["patient_id"], manifest_id="m", dataset_id="synthetic", max_participants=6
    )
    assert len(manifest.folds) == 6
    for fold in manifest.folds:
        assert len(fold["test"]) == 1
        assert fold["test"][0] not in fold["train"]


def test_holdout_split_is_disjoint(cohort):
    manifest = make_holdout_manifest(
        cohort["patient_id"],
        cohort["pmos_binary"],
        manifest_id="m",
        dataset_id="synthetic",
        test_size=0.25,
    )
    fold = manifest.folds[0]
    assert not set(fold["train"]) & set(fold["test"])
    assert len(fold["test"]) == pytest.approx(30, abs=2)


def test_reserved_holdout_never_appears_in_any_fold(cohort):
    held = reserve_holdout_patients(cohort["patient_id"], cohort["pmos_binary"], fraction=0.2)
    manifest = make_repeated_stratified_kfold_manifest(
        cohort["patient_id"],
        cohort["pmos_binary"],
        manifest_id="m",
        dataset_id="synthetic",
        n_splits=4,
        holdout_ids=held,
    )
    assert held
    for fold in manifest.folds:
        assert not set(held) & (set(fold["train"]) | set(fold["test"]))


def test_assert_disjoint_fires_on_a_leaking_manifest():
    """The guard must actually raise — a silent check protects nothing."""
    leaking = SplitManifest(
        manifest_id="bad",
        dataset_id="synthetic",
        dataset_version="v1",
        strategy="holdout",
        n_splits=1,
        folds=[{"train": ["P1", "P2", "P3"], "test": ["P3", "P4"]}],
    )
    with pytest.raises(ValueError, match="leaks 1 patient"):
        leaking.assert_disjoint()


def test_assert_disjoint_fires_when_a_fold_touches_the_holdout():
    leaking = SplitManifest(
        manifest_id="bad",
        dataset_id="synthetic",
        dataset_version="v1",
        strategy="holdout",
        n_splits=1,
        folds=[{"train": ["P1", "P2"], "test": ["P3"]}],
        holdout_ids=["P2"],
    )
    with pytest.raises(ValueError, match="untouched holdout"):
        leaking.assert_disjoint()


def test_save_and_load_round_trip(tmp_path, cohort):
    manifest = make_repeated_stratified_kfold_manifest(
        cohort["patient_id"],
        cohort["pmos_binary"],
        manifest_id="round-trip",
        dataset_id="synthetic",
        n_splits=3,
    )
    path = save_split_manifest(manifest, tmp_path / "split_manifest.json")
    reloaded = load_split_manifest(path)
    assert reloaded.manifest_id == manifest.manifest_id
    assert reloaded.folds == manifest.folds


def test_fold_row_indices_do_not_overlap(cohort):
    manifest = build_split_manifest(
        "repeated_stratified_kfold",
        cohort["patient_id"].tolist(),
        cohort["pmos_binary"].tolist(),
        None,
        manifest_id="m",
        dataset_id="synthetic",
        n_splits=4,
    )
    for fold in manifest.folds:
        train_idx, test_idx = fold_row_indices(fold, cohort["patient_id"].tolist())
        assert not set(train_idx.tolist()) & set(test_idx.tolist())
        assert train_idx.size + test_idx.size == len(cohort)


def test_repeated_rows_for_one_patient_stay_on_one_side():
    """The real leakage risk: several rows per patient must not be split apart."""
    patient_ids = [f"P{i // 3:03d}" for i in range(90)]
    labels = [(i // 3) % 2 for i in range(90)]  # label is a property of the patient
    manifest = make_repeated_stratified_kfold_manifest(
        patient_ids, labels, manifest_id="m", dataset_id="synthetic", n_splits=3
    )
    for fold in manifest.folds:
        train_idx, test_idx = fold_row_indices(fold, patient_ids)
        train_patients = {patient_ids[i] for i in train_idx}
        test_patients = {patient_ids[i] for i in test_idx}
        assert not train_patients & test_patients
