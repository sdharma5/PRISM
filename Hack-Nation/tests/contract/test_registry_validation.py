"""Contract tests for the YAML registries.

The registries are the project's shared vocabulary. A malformed entry here
propagates silently into every adapter and model, so the invariants are pinned
rather than assumed.
"""

from __future__ import annotations

import pytest

from registry.loader import (
    REGISTRY_DIR,
    in_valid_range,
    load_dataset_registry,
    load_phenotype_domains,
    load_schema_versions,
    load_units,
    load_variable_registry,
)
from schemas.dataset import DatasetSpec, ValidRange, VariableSpec

pytestmark = pytest.mark.contract

VARIABLES = load_variable_registry().variables
DATASETS = load_dataset_registry().datasets
UNITS = load_units()


# -- Loading ---------------------------------------------------------------


def test_every_registry_file_loads():
    assert VARIABLES and DATASETS and UNITS
    assert load_phenotype_domains()
    assert load_schema_versions()
    assert REGISTRY_DIR.is_dir()


# -- Variable registry -----------------------------------------------------


@pytest.mark.parametrize("code", sorted(VARIABLES))
def test_variable_has_a_canonical_name_and_valid_type(code):
    spec = VARIABLES[code]
    assert spec.canonical_name.strip()
    assert spec.type in {
        "continuous",
        "integer",
        "binary",
        "categorical",
        "ordinal",
        "datetime",
        "text",
    }


@pytest.mark.parametrize("code", sorted(VARIABLES))
def test_valid_range_is_ordered(code):
    valid_range = VARIABLES[code].valid_range
    if valid_range and valid_range.min is not None and valid_range.max is not None:
        assert valid_range.min < valid_range.max, f"{code}: valid_range is inverted"


@pytest.mark.parametrize("code", sorted(VARIABLES))
def test_categorical_variables_declare_categories(code):
    spec = VARIABLES[code]
    if spec.type == "categorical":
        assert spec.categories, f"{code}: categorical variable without categories"
        assert len(set(spec.categories)) == len(spec.categories)


def test_categorical_without_categories_is_rejected():
    with pytest.raises(ValueError, match="must list categories"):
        VariableSpec(canonical_name="Bad", type="categorical")


@pytest.mark.parametrize("code", sorted(VARIABLES))
def test_continuous_variables_declare_a_unit_or_are_explicitly_unitless(code):
    spec = VARIABLES[code]
    if spec.type == "continuous":
        assert spec.canonical_unit or spec.unit, f"{code}: continuous variable with no unit"


def test_variable_registry_raises_on_an_unknown_code():
    with pytest.raises(KeyError, match="Unknown canonical variable"):
        load_variable_registry().get("not_a_real_variable")


def test_in_valid_range_flags_implausible_values():
    assert in_valid_range("age", 30) is True
    assert in_valid_range("age", 999) is False
    assert in_valid_range("bmi", 24.0) is True
    assert in_valid_range("bmi", 500.0) is False
    # A variable with no declared range never rejects anything.
    assert in_valid_range("urinary_lh", 1e6) is True


def test_valid_range_boundaries_are_inclusive():
    assert ValidRange(min=0, max=10).contains(0)
    assert ValidRange(min=0, max=10).contains(10)
    assert not ValidRange(min=0, max=10).contains(10.001)


# -- Dataset registry ------------------------------------------------------


@pytest.mark.parametrize("dataset_id", sorted(DATASETS))
def test_dataset_declares_allowed_uses_and_a_modality(dataset_id):
    spec = DATASETS[dataset_id]
    assert spec.allowed_uses, f"{dataset_id}: no allowed_uses"
    assert spec.modality, f"{dataset_id}: no modality"
    assert spec.name.strip()


@pytest.mark.parametrize("dataset_id", sorted(DATASETS))
def test_allowed_and_prohibited_never_overlap(dataset_id):
    spec = DATASETS[dataset_id]
    assert not set(spec.allowed_uses) & set(spec.prohibited_claims)


def test_dataset_without_allowed_uses_is_rejected():
    with pytest.raises(ValueError, match="must declare allowed_uses"):
        DatasetSpec(name="Bad", modality=["tabular"])


def test_overlapping_allowed_and_prohibited_is_rejected():
    with pytest.raises(ValueError, match="both allowed and prohibited"):
        DatasetSpec(
            name="Bad",
            modality=["tabular"],
            allowed_uses=["pmos_diagnosis"],
            prohibited_claims=["pmos_diagnosis"],
        )


def test_require_returns_the_spec_for_a_permitted_use():
    spec = load_dataset_registry().require("pmos_tabular_public", "binary_baseline")
    assert spec.name.startswith("Public PMOS")


def test_require_refuses_a_prohibited_use():
    with pytest.raises(PermissionError, match="does not allow use"):
        load_dataset_registry().require("pmos_tabular_public", "prospective_clinical_deployment")


def test_require_refuses_an_unlisted_use():
    # Fail closed: anything not explicitly allowed is refused, not tolerated.
    with pytest.raises(PermissionError):
        load_dataset_registry().require("nhanes_2021_2023", "something_nobody_declared")


def test_require_raises_for_an_unknown_dataset():
    with pytest.raises(KeyError, match="Unknown dataset"):
        load_dataset_registry().require("no_such_dataset", "binary_baseline")


def test_nhanes_may_not_be_used_for_longitudinal_or_diagnostic_claims():
    for prohibited in ("longitudinal_state_modeling", "pmos_diagnosis"):
        with pytest.raises(PermissionError):
            load_dataset_registry().require("nhanes_2021_2023", prohibited)


def test_mcphases_may_not_be_used_as_a_pmos_baseline():
    with pytest.raises(PermissionError):
        load_dataset_registry().require("mcphases", "binary_baseline")


# -- Unit registry cross-checks -------------------------------------------


@pytest.mark.parametrize("code", sorted(UNITS["unit_conversions"]))
def test_conversion_target_matches_the_variable_registry(code):
    spec = VARIABLES.get(code)
    if spec is None:
        pytest.skip(f"{code} has no variable entry")
    canonical = UNITS["unit_conversions"][code]["canonical_unit"]
    assert (spec.canonical_unit or spec.unit) == canonical


@pytest.mark.parametrize("code", sorted(UNITS["affine_conversions"]))
def test_affine_conversion_target_matches_the_variable_registry(code):
    spec = VARIABLES.get(code)
    if spec is None:
        pytest.skip(f"{code} has no variable entry")
    canonical = UNITS["affine_conversions"][code]["canonical_unit"]
    assert (spec.canonical_unit or spec.unit) == canonical


def test_unit_aliases_are_self_consistent():
    aliases = UNITS["aliases"]
    for target in set(aliases.values()):
        assert aliases.get(target, target) == target, f"alias target '{target}' remaps elsewhere"


def test_every_conversion_source_unit_has_a_positive_factor():
    for code, spec in UNITS["unit_conversions"].items():
        for source, factor in spec["from"].items():
            assert float(factor) > 0, f"{code}: non-positive factor for '{source}'"
