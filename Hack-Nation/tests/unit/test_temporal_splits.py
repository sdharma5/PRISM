"""Days from one participant must NEVER be split across train and test.

This test exists to fail loudly if someone regresses the split. A random
day-level split would put day 40 of a participant in train and day 41 in test —
same lookback window, same cycle, same body — and the reported score would
measure interpolation within a known person rather than generalisation to a new
one. That is the single most likely way this repository could produce a
scientifically worthless but impressive-looking number.
"""

from __future__ import annotations

import numpy as np
import pytest

from models.temporal.state_model import TemporalStateModel, grouped_participant_split
from schemas.patient import SplitManifest
from tests.fixtures.synthetic_cycles import generate_cohort


@pytest.fixture(scope="module")
def cohort():
    return generate_cohort(n_participants=10, n_days=60, seed=0)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_no_participant_appears_in_both_train_and_test(cohort, seed):
    """The core invariant, checked across several seeds."""
    groups = [day.participant_id for day in cohort.days]
    train_index, test_index = grouped_participant_split(groups, test_fraction=0.3, seed=seed)

    train_ids = {groups[i] for i in train_index}
    test_ids = {groups[i] for i in test_index}
    assert train_ids & test_ids == set(), "participant leaked across the split"
    assert train_ids | test_ids == set(groups), "every participant must be assigned"


def test_indices_partition_the_dataset(cohort):
    """No day may be dropped or duplicated by the split."""
    groups = [day.participant_id for day in cohort.days]
    train_index, test_index = grouped_participant_split(groups, test_fraction=0.3, seed=0)
    combined = np.concatenate([train_index, test_index])
    assert len(combined) == len(groups)
    assert set(combined.tolist()) == set(range(len(groups)))


def test_all_days_of_a_participant_stay_together(cohort):
    """Every one of a participant's days lands on the same side."""
    groups = [day.participant_id for day in cohort.days]
    train_index, _ = grouped_participant_split(groups, test_fraction=0.3, seed=1)
    train_ids = {groups[i] for i in train_index}

    per_participant: dict[str, set[bool]] = {}
    train_set = set(train_index.tolist())
    for index, participant in enumerate(groups):
        per_participant.setdefault(participant, set()).add(index in train_set)
    for participant, sides in per_participant.items():
        assert len(sides) == 1, f"{participant} has days on both sides of the split"
    assert train_ids


def test_a_deliberately_broken_random_day_split_is_detectable(cohort):
    """A naive row-level split leaks, and this test proves the check catches it.

    If this ever stops failing, the leakage detector itself has regressed.
    """
    groups = [day.participant_id for day in cohort.days]
    rng = np.random.default_rng(0)
    shuffled = rng.permutation(len(groups))
    bad_train = shuffled[: int(0.7 * len(groups))]
    bad_test = shuffled[int(0.7 * len(groups)) :]

    train_ids = {groups[i] for i in bad_train}
    test_ids = {groups[i] for i in bad_test}
    assert train_ids & test_ids, "a random day split must leak participants"


def test_split_manifest_rejects_a_leaking_fold():
    """The SplitManifest contract fails closed on overlap."""
    manifest = SplitManifest(
        manifest_id="leaky",
        dataset_id="synthetic_cycles",
        dataset_version="unversioned",
        strategy="grouped_kfold",
        n_splits=1,
        seeds=[0],
        folds=[{"train": ["A", "B"], "test": ["B", "C"]}],
    )
    with pytest.raises(ValueError, match="leaks"):
        manifest.assert_disjoint()


def test_model_trained_on_train_predicts_only_unseen_participants(cohort):
    """End to end: predictions on the test side involve no training participant."""
    groups = [day.participant_id for day in cohort.days]
    train_index, test_index = grouped_participant_split(groups, test_fraction=0.3, seed=0)
    train_days = [cohort.days[i] for i in train_index]
    test_days = [cohort.days[i] for i in test_index]

    model = TemporalStateModel(lookback_days=21, hidden_size=16, seed=0).fit(train_days)
    outputs = model.predict(test_days)

    assert outputs
    train_ids = {d.participant_id for d in train_days}
    predicted_ids = {o.patient_id for o in outputs}
    assert predicted_ids & train_ids == set()


def test_test_fraction_is_applied_to_participants_not_rows(cohort):
    """A 30% split means 30% of people, not 30% of days."""
    groups = [day.participant_id for day in cohort.days]
    _, test_index = grouped_participant_split(groups, test_fraction=0.3, seed=0)
    n_test_participants = len({groups[i] for i in test_index})
    assert n_test_participants == pytest.approx(0.3 * len(set(groups)), abs=1)
