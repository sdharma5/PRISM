"""NHANES and mcPHASES adapter tests on tiny synthetic tables."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from ingestion.mcphases.alignment import (
    UNOBSERVED_SENTINEL,
    assign_cycle_day,
    assign_cycle_phase,
    build_participant_days,
    derive_time_since_last_observed,
)
from ingestion.mcphases.daily_aggregation import (
    aggregate_day,
    day_night_difference,
    missing_fraction,
    rate_of_change_per_hour,
    time_in_range,
)
from ingestion.mcphases.loader import (
    McPhasesAdapter,
    McPhasesDataNotFoundError,
    resolve_stream_files,
)
from ingestion.mcphases.validation import validate_no_fabricated_values, validate_participant_days
from ingestion.nhanes.loader import NhanesAdapter
from ingestion.nhanes.merge import SEQN, collect_survey_metadata, merge_components
from ingestion.nhanes.survey_design import (
    weighted_mean,
    weighted_quantile,
    weighted_reference_range,
    weighted_std,
)
from ingestion.nhanes.validation import validate_merged_frame

# -- NHANES: merge ---------------------------------------------------------


@pytest.fixture
def components() -> dict[str, pd.DataFrame]:
    demo = pd.DataFrame(
        {
            SEQN: [1, 2, 3],
            "RIDAGEYR": [28.0, 34.0, 41.0],
            "WTMEC2YR": [12000.0, 8000.0, 25000.0],
            "SDMVSTRA": [101, 101, 102],
            "SDMVPSU": [1, 2, 1],
        }
    )
    labs = pd.DataFrame(
        {SEQN: [1, 2, 4], "LBXTST": [38.0, 72.0, 51.0], "LBXSHBG": [60.0, 22.0, 45.0]}
    )
    body = pd.DataFrame({SEQN: [1, 2, 3], "BMXBMI": [23.1, 34.0, 27.5]})
    return {"DEMO": demo, "TST": labs, "BMX": body}


def test_components_merge_on_seqn_as_an_outer_join(components):
    merged = merge_components(components)
    # SEQN 4 exists only in the lab file; an inner join would silently drop it.
    assert list(merged[SEQN]) == [1, 2, 3, 4]
    assert merged.loc[merged[SEQN] == 3, "LBXTST"].isna().all()


def test_merge_rejects_a_component_without_seqn():
    with pytest.raises(ValueError, match="no SEQN"):
        merge_components({"BAD": pd.DataFrame({"x": [1]})})


def test_merge_rejects_duplicate_seqn():
    with pytest.raises(ValueError, match="duplicate"):
        merge_components({"BAD": pd.DataFrame({SEQN: [1, 1], "RIDAGEYR": [20.0, 30.0]})})


def test_required_component_restricts_the_population(components):
    merged = merge_components(components, required=["DEMO"])
    assert list(merged[SEQN]) == [1, 2, 3]


def test_overlapping_columns_are_suffixed_not_overwritten():
    merged = merge_components(
        {
            "A": pd.DataFrame({SEQN: [1], "BMXBMI": [22.0]}),
            "B": pd.DataFrame({SEQN: [1], "BMXBMI": [99.0]}),
        }
    )
    assert set(merged.columns) == {SEQN, "BMXBMI", "BMXBMI__B"}


def test_survey_metadata_is_collected_per_respondent(components):
    merged = merge_components(components)
    metadata = collect_survey_metadata(merged.iloc[0])
    assert metadata["WTMEC2YR"] == 12000.0
    assert metadata["SDMVSTRA"] == 101.0


def test_survey_design_columns_may_not_be_mapped_as_variables(components):
    merged = merge_components(components)
    errors = validate_merged_frame(merged, {"WTMEC2YR": "age"})
    assert any("survey-design column" in e for e in errors)


# -- NHANES: adapter -------------------------------------------------------


def test_nhanes_refuses_a_prohibited_use():
    with pytest.raises(PermissionError):
        NhanesAdapter(use="pcos_diagnosis")
    with pytest.raises(PermissionError):
        NhanesAdapter(use="longitudinal_state_modeling")


def test_nhanes_allows_a_registered_use():
    assert NhanesAdapter(use="population_reference").use == "population_reference"


def test_nhanes_emits_scoped_ids_and_carries_weights(components):
    adapter = NhanesAdapter()
    events = adapter.run(components)
    assert {e.patient_id for e in events} == {
        "nhanes_2021_2023:1",
        "nhanes_2021_2023:2",
        "nhanes_2021_2023:3",
        "nhanes_2021_2023:4",
    }
    testosterone = [
        e
        for e in events
        if e.canonical_variable_code == "total_testosterone"
        and e.patient_id == "nhanes_2021_2023:1"
    ][0]
    assert testosterone.value == 38.0
    assert testosterone.modality == "laboratory"
    assert "WTMEC2YR=12000.0" in testosterone.evidence_text


def test_nhanes_missing_lab_becomes_not_collected_not_zero(components):
    events = NhanesAdapter().run(components)
    missing = [
        e
        for e in events
        if e.canonical_variable_code == "total_testosterone"
        and e.patient_id == "nhanes_2021_2023:3"
    ][0]
    assert missing.value is None
    assert missing.missingness_status == "not_collected"


def test_nhanes_manifest_documents_design_columns(components):
    adapter = NhanesAdapter()
    adapter.run(components)
    manifest = adapter.build_manifest()
    assert "WTMEC2YR" in manifest.excluded_source_columns
    assert manifest.n_events_emitted > 0


# -- NHANES: survey design -------------------------------------------------


def test_weighted_mean_differs_from_the_unweighted_mean():
    values = [10.0, 20.0]
    weights = [1.0, 9.0]
    assert weighted_mean(values, weights) == pytest.approx(19.0)
    assert weighted_mean(values, weights) != np.mean(values)


def test_equal_weights_reproduce_the_plain_mean_and_std():
    values = [1.0, 2.0, 3.0, 4.0]
    weights = [1.0] * 4
    assert weighted_mean(values, weights) == pytest.approx(np.mean(values))
    assert weighted_std(values, weights) == pytest.approx(np.std(values, ddof=1))


def test_weighted_quantile_matches_numpy_for_equal_weights():
    values = list(np.linspace(0, 100, 101))
    weights = [1.0] * 101
    assert weighted_quantile(values, weights, 0.5) == pytest.approx(50.0, abs=0.6)


def test_weighted_quantile_shifts_toward_heavily_weighted_values():
    median = weighted_quantile([1.0, 2.0, 100.0], [1.0, 1.0, 1000.0], 0.5)
    assert median > 90


def test_weighted_quantile_accepts_a_sequence():
    result = weighted_quantile([1.0, 2.0, 3.0], [1.0, 1.0, 1.0], [0.0, 1.0])
    assert result == [1.0, 3.0]


def test_weighted_quantile_rejects_a_quantile_outside_zero_one():
    with pytest.raises(ValueError, match="outside"):
        weighted_quantile([1.0], [1.0], 1.5)


def test_non_positive_weights_and_nans_are_dropped():
    assert weighted_mean([1.0, float("nan"), 5.0], [1.0, 1.0, 1.0]) == pytest.approx(3.0)
    assert weighted_mean([1.0, 99.0], [1.0, 0.0]) == pytest.approx(1.0)


def test_no_usable_pairs_raises_rather_than_returning_zero():
    with pytest.raises(ValueError, match="No usable"):
        weighted_mean([float("nan")], [1.0])


def test_reference_range_reports_a_central_interval():
    rng = np.random.default_rng(0)
    values = rng.normal(50, 10, 5000)
    result = weighted_reference_range(values, np.ones(5000))
    assert result["lower"] < result["median"] < result["upper"]
    assert result["median"] == pytest.approx(50, abs=1.0)
    assert result["n_unweighted"] == 5000


# -- mcPHASES: aggregation -------------------------------------------------


def stamps(day: date, hours: list[int]) -> list[datetime]:
    return [datetime(day.year, day.month, day.day, h) for h in hours]


def test_aggregate_reports_every_documented_statistic():
    day = date(2026, 1, 1)
    aggregate = aggregate_day(
        "cgm_mean_glucose",
        stamps(day, [8, 12, 20, 23, 2]),
        [100.0, 150.0, 200.0, 90.0, 80.0],
        expected_per_day=288,
    )
    assert aggregate.is_observed is True
    assert aggregate.n_observations == 5
    assert aggregate.mean == pytest.approx(124.0)
    assert aggregate.median == pytest.approx(100.0)
    assert aggregate.std == pytest.approx(np.std([100, 150, 200, 90, 80], ddof=1))
    assert aggregate.minimum == 80.0
    assert aggregate.maximum == 200.0
    assert aggregate.day_night_difference is not None
    assert 0.0 <= aggregate.time_in_range <= 1.0
    assert aggregate.rate_of_change_per_hour is not None
    assert aggregate.missing_fraction == pytest.approx(1 - 5 / 288)


def test_a_day_with_no_readings_is_unobserved_not_zero_filled():
    aggregate = aggregate_day("cgm_mean_glucose", [], [], expected_per_day=288)
    assert aggregate.is_observed is False
    assert aggregate.mean is None
    assert aggregate.std is None
    assert aggregate.missing_fraction == 1.0


def test_a_single_reading_has_undefined_not_zero_variability():
    aggregate = aggregate_day("cgm_mean_glucose", stamps(date(2026, 1, 1), [9]), [110.0])
    assert aggregate.is_observed is True
    assert aggregate.std is None, "one reading cannot assert perfect stability"
    assert aggregate.warnings


def test_day_night_difference_needs_both_windows():
    day = date(2026, 1, 1)
    assert day_night_difference(stamps(day, [8, 12]), [10.0, 20.0]) is None
    diff = day_night_difference(stamps(day, [8, 12, 23]), [10.0, 20.0, 5.0])
    assert diff == pytest.approx(15.0 - 5.0)


def test_time_in_range_uses_the_cgm_consensus_target():
    assert time_in_range([60.0, 100.0, 150.0, 250.0], code="cgm_mean_glucose") == pytest.approx(0.5)


def test_time_in_range_falls_back_to_the_registry_range():
    assert time_in_range([30.0, 200.0], code="resting_heart_rate") == pytest.approx(1.0)


def test_time_in_range_is_none_without_readings():
    assert time_in_range([], code="cgm_mean_glucose") is None


def test_rate_of_change_uses_real_timestamp_gaps():
    day = date(2026, 1, 1)
    roc = rate_of_change_per_hour(stamps(day, [0, 2, 4]), [100.0, 120.0, 140.0])
    assert roc == pytest.approx(10.0)


def test_rate_of_change_needs_two_points():
    assert rate_of_change_per_hour(stamps(date(2026, 1, 1), [0]), [100.0]) is None


def test_missing_fraction_is_none_without_an_expectation():
    assert missing_fraction(5, None) is None
    assert missing_fraction(288, 288) == 0.0


# -- mcPHASES: alignment ---------------------------------------------------


def test_cycle_day_counts_from_the_most_recent_onset():
    onsets = [date(2026, 1, 1), date(2026, 1, 29)]
    assert assign_cycle_day(date(2026, 1, 1), onsets) == 1
    assert assign_cycle_day(date(2026, 1, 10), onsets) == 10
    assert assign_cycle_day(date(2026, 1, 30), onsets) == 2


def test_cycle_day_is_unknown_before_the_first_onset():
    assert assign_cycle_day(date(2025, 12, 31), [date(2026, 1, 1)]) is None


@pytest.mark.parametrize(
    ("cycle_day", "expected"),
    [(1, "menstrual"), (5, "menstrual"), (9, "follicular"), (15, "peri_ovulatory"), (25, "luteal")],
)
def test_cycle_phase_assignment(cycle_day, expected):
    assert assign_cycle_phase(cycle_day, cycle_length=28) == expected


def test_cycle_phase_is_unknown_without_enough_information():
    assert assign_cycle_phase(None, 28) == "unknown"
    assert assign_cycle_phase(20, None) == "unknown"


def test_time_since_last_observed_grows_across_gaps():
    recency = derive_time_since_last_observed([False, True, False, False, True])
    assert recency == [UNOBSERVED_SENTINEL, 0.0, 1.0, 2.0, 0.0]


def test_unobserved_days_carry_no_value():
    days = build_participant_days(
        "mcphases:P1",
        [date(2026, 1, 1), date(2026, 1, 2), date(2026, 1, 3)],
        {
            date(2026, 1, 1): {"cgm_mean_glucose_mean": 105.0},
            date(2026, 1, 3): {"cgm_mean_glucose_mean": 111.0},
        },
        menses_onsets=[date(2026, 1, 1)],
        cycle_length=28,
    )
    assert [d.is_observed["cgm_mean_glucose_mean"] for d in days] == [True, False, True]
    assert days[1].values["cgm_mean_glucose_mean"] is None
    assert days[1].time_since_last_observed["cgm_mean_glucose_mean"] == 1.0
    assert [d.cycle_day for d in days] == [1, 2, 3]
    assert validate_no_fabricated_values(days) == []


def test_fabricated_value_on_an_unobserved_day_is_caught():
    days = build_participant_days("mcphases:P1", [date(2026, 1, 1)], {date(2026, 1, 1): {"x": 1.0}})
    days[0].is_observed["x"] = False  # simulate an upstream forward-fill bug
    assert any("fabricated data" in e for e in validate_participant_days(days))


# -- mcPHASES: adapter -----------------------------------------------------


@pytest.fixture
def stream() -> pd.DataFrame:
    base = datetime(2026, 3, 1, 0, 0)
    rows = []
    for participant in ("P1", "P2"):
        for hour in range(0, 48, 4):
            rows.append(
                {
                    "participant_id": participant,
                    "timestamp": base + timedelta(hours=hour),
                    "glucose_mgdl": 100.0 + hour,
                    "heart_rate_bpm": 62.0 + (hour % 5),
                }
            )
    return pd.DataFrame(rows)


def test_mcphases_refuses_a_prohibited_use():
    with pytest.raises(PermissionError):
        McPhasesAdapter(use="binary_baseline")
    with pytest.raises(PermissionError):
        McPhasesAdapter(use="pcos_diagnosis")


def test_mcphases_builds_participant_days(stream):
    days = McPhasesAdapter().run(stream)
    assert {d.participant_id for d in days} == {"mcphases:P1", "mcphases:P2"}
    per_participant = [d for d in days if d.participant_id == "mcphases:P1"]
    assert len(per_participant) == 2
    assert per_participant[0].values["cgm_mean_glucose_mean"] is not None
    assert per_participant[0].is_observed["cgm_mean_glucose_mean"] is True
    assert per_participant[0].source_dataset == "mcphases"


def test_mcphases_emits_no_point_events(stream):
    adapter = McPhasesAdapter()
    raw = adapter.load_raw(stream)
    assert adapter.transform(raw) == []
    assert adapter.participant_days


def test_mcphases_rejects_a_stream_without_a_measurement_column():
    adapter = McPhasesAdapter()
    frame = pd.DataFrame({"participant_id": ["P1"], "timestamp": [datetime(2026, 1, 1)]})
    with pytest.raises(ValueError, match="no recognised measurement column"):
        adapter.run(frame)


def test_mcphases_manifest_is_buildable(stream):
    adapter = McPhasesAdapter()
    adapter.run(stream)
    manifest = adapter.build_manifest()
    assert manifest.dataset_id == "mcphases"
    assert "glucose_mgdl" in manifest.variable_mapping


def test_mcphases_refuses_to_guess_among_unrecognised_csvs(tmp_path):
    """A directory of per-stream CSVs is refused, not silently sampled.

    The official mcPHASES distribution ships one CSV per stream under its own
    column names. Resolution used to fall back to the alphabetically first CSV,
    which selected ``active_minutes.csv`` and ingested it as the stream table.
    """
    (tmp_path / "active_minutes.csv").write_text("id,day_in_study,sedentary\nP1,1,10\n")
    (tmp_path / "glucose.csv").write_text("id,day_in_study,timestamp,glucose_value\nP1,1,x,5\n")

    with pytest.raises(McPhasesDataNotFoundError) as excinfo:
        resolve_stream_files(tmp_path)

    message = str(excinfo.value)
    # The diagnosis must name each rejected file and why, so the user is not
    # left guessing which of their CSVs was wrong.
    assert "active_minutes.csv" in message
    assert "glucose.csv" in message
    assert "participant_id" in message


def test_mcphases_resolves_a_structurally_valid_csv_by_any_name(tmp_path):
    """Recognition is by columns, not filename."""
    oddly_named = tmp_path / "not_a_conventional_name.csv"
    oddly_named.write_text("participant_id,timestamp,glucose_mgdl\nP1,2026-01-01T00:00:00,95\n")
    assert resolve_stream_files(tmp_path) == [oddly_named]


def test_mcphases_load_raw_names_the_file_it_rejected(tmp_path):
    """A non-stream CSV fails with guidance, not a bare pandas KeyError."""
    path = tmp_path / "active_minutes.csv"
    path.write_text("id,day_in_study,sedentary\nP1,1,10\n")
    with pytest.raises(McPhasesDataNotFoundError, match="active_minutes.csv"):
        McPhasesAdapter().load_raw(path)


def _write_mcphases_fixture(root):
    """A miniature mcPHASES directory in the distribution's real column shape."""
    root.mkdir(parents=True, exist_ok=True)
    # Day 1-3 in one block, then day 900 after the multi-year study break.
    (root / "hormones_and_selfreport.csv").write_text(
        "id,study_interval,day_in_study,phase,lh,estrogen,pdg,cramps,bloating,moodswing,sorebreasts\n"
        "1,2022,1,Follicular,2.9,94.2,,High,Not at all,Low,Low\n"
        "1,2022,3,Fertility,30.0,240.0,1.2,Low,Low,Low,Low\n"
        "1,2024,900,Luteal,4.0,120.0,8.0,Very High,Low,Low,Low\n"
    )
    # 5.0 mmol/L is ~90 mg/dL. Unconverted it would be a plausible-looking 5.
    (root / "glucose.csv").write_text(
        "id,study_interval,day_in_study,timestamp,glucose_value\n"
        "1,2022,1,00:04:06,5.0\n"
        "1,2022,1,00:09:06,5.0\n"
    )


def test_consolidator_converts_glucose_to_mg_dl(tmp_path):
    """mmol/L in, mg/dL out, using the registry factor rather than a local copy."""
    from registry.loader import convert_to_canonical
    from scripts.consolidate_mcphases import build_participant_days

    _write_mcphases_fixture(tmp_path)
    days = build_participant_days(tmp_path)
    day_one = next(d for d in days if d.study_day == 1)
    expected = convert_to_canonical("cgm_mean_glucose", 5.0, "mmol/L").value
    assert day_one.values["mean_glucose"] == pytest.approx(expected)
    assert 80 < day_one.values["mean_glucose"] < 100


def test_consolidator_does_not_bridge_the_study_break(tmp_path):
    """Days 4..899 must not be invented to join the 2022 and 2024 blocks."""
    from scripts.consolidate_mcphases import build_participant_days

    _write_mcphases_fixture(tmp_path)
    emitted = {d.study_day for d in build_participant_days(tmp_path)}
    assert emitted == {1, 2, 3, 900}, "gap between study blocks must stay empty"
    # Day 2 is a real within-block gap and IS emitted, unobserved rather than filled.
    day_two = next(d for d in build_participant_days(tmp_path) if d.study_day == 2)
    assert day_two.is_observed["lh"] is False
    assert day_two.values["lh"] is None


def test_consolidator_maps_phases_and_symptom_threshold(tmp_path):
    """Fertility maps to peri_ovulatory; only Moderate+ counts as present."""
    from scripts.consolidate_mcphases import build_participant_days

    _write_mcphases_fixture(tmp_path)
    days = {d.study_day: d for d in build_participant_days(tmp_path)}
    assert days[3].cycle_phase == "peri_ovulatory"
    assert days[1].cycle_phase == "follicular"
    assert days[1].daily_symptoms["cramps"] is True  # "High"
    assert days[1].daily_symptoms["mood_low"] is False  # "Low"


def test_consolidated_days_are_preferred_over_stream_tables(tmp_path):
    """A consolidated file short-circuits stream-table resolution."""
    from ingestion.mcphases.loader import CONSOLIDATED_DAYS_FILENAME, load_participant_days
    from scripts.consolidate_mcphases import build_participant_days

    _write_mcphases_fixture(tmp_path)
    days = build_participant_days(tmp_path)
    target = tmp_path / CONSOLIDATED_DAYS_FILENAME
    target.write_text("\n".join(d.model_dump_json() for d in days))

    loaded = load_participant_days(tmp_path)
    assert len(loaded) == len(days)
    assert loaded[0].values.keys() == days[0].values.keys()


def test_consolidator_detects_mixed_glucose_units_per_participant(tmp_path):
    """mcPHASES reports glucose in mmol/L for most participants and mg/dL for some.

    Treating the file as single-unit multiplies the mg/dL participants by ~18
    into impossible ~2000 mg/dL readings.
    """
    from scripts.consolidate_mcphases import build_participant_days

    _write_mcphases_fixture(tmp_path)
    # Participant 2 reports mg/dL; participant 1 (in the fixture) reports mmol/L.
    (tmp_path / "glucose.csv").write_text(
        "id,study_interval,day_in_study,timestamp,glucose_value\n"
        "1,2022,1,00:04:06,5.0\n"
        "2,2022,1,00:04:06,110.0\n"
        "2,2022,1,00:09:06,112.0\n"
    )
    (tmp_path / "hormones_and_selfreport.csv").write_text(
        "id,study_interval,day_in_study,phase,lh,estrogen,pdg,cramps,bloating,moodswing,sorebreasts\n"
        "1,2022,1,Follicular,2.9,94.2,,High,Not at all,Low,Low\n"
        "2,2022,1,Follicular,3.0,95.0,,Low,Low,Low,Low\n"
    )
    days = {d.participant_id: d for d in build_participant_days(tmp_path)}
    mmol_participant = days["mcphases:1"].values["mean_glucose"]
    mgdl_participant = days["mcphases:2"].values["mean_glucose"]
    # Both land in physiological mg/dL range; the mg/dL one is NOT scaled again.
    assert 80 < mmol_participant < 100
    assert 100 < mgdl_participant < 120


def test_consolidator_drops_zero_resting_heart_rate_sentinels(tmp_path):
    """value == 0.0 with error == 0.0 is this file's missing marker, not 0 bpm."""
    from scripts.consolidate_mcphases import build_participant_days

    _write_mcphases_fixture(tmp_path)
    (tmp_path / "resting_heart_rate.csv").write_text(
        "id,study_interval,day_in_study,value,error\n1,2022,1,72.5,6.79\n1,2022,3,0.0,0.0\n"
    )
    days = {d.study_day: d for d in build_participant_days(tmp_path)}
    assert days[1].values["resting_heart_rate"] == pytest.approx(72.5)
    # The sentinel day is unobserved, never a recorded zero.
    assert days[3].is_observed["resting_heart_rate"] is False
    assert days[3].values["resting_heart_rate"] is None


def test_consolidator_refuses_a_wrong_source_unit(tmp_path, monkeypatch):
    """A mis-declared source unit fails loudly instead of writing the cohort."""
    import scripts.consolidate_mcphases as mod

    _write_mcphases_fixture(tmp_path)
    # Claim the mmol/L glucose column is already mg/dL: values stay ~5, far below
    # the registry's 20-600 valid range for cgm_mean_glucose.
    monkeypatch.setitem(mod.CHANNEL_SPECS, "mean_glucose", ("cgm_mean_glucose", "mg/dL"))
    monkeypatch.setattr(mod, "detect_glucose_units", lambda frame: {})
    with pytest.raises(SystemExit, match="contradict their declared units"):
        mod.consolidate(tmp_path)
