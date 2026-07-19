"""Every conversion in registry/units.yaml is exercised here.

A wrong conversion factor is silent and catastrophic — it produces plausible
numbers in the wrong scale — so each factor is round-tripped and the
clinically important ones are pinned to known values.
"""

from __future__ import annotations

import math

import pytest

from registry.loader import (
    UnitConversionError,
    convert_to_canonical,
    load_units,
    normalize_unit,
)

UNITS = load_units()


def _multiplicative_cases() -> list[tuple[str, str, float]]:
    cases = []
    for code, spec in UNITS["unit_conversions"].items():
        for source_unit, factor in spec["from"].items():
            cases.append((code, source_unit, float(factor)))
    return cases


def _affine_cases() -> list[tuple[str, str, float, float]]:
    cases = []
    for code, spec in UNITS["affine_conversions"].items():
        for source_unit, rule in spec["from"].items():
            cases.append((code, source_unit, float(rule["scale"]), float(rule["offset"])))
    return cases


@pytest.mark.parametrize(("code", "source_unit", "factor"), _multiplicative_cases())
def test_every_multiplicative_conversion_applies_its_factor(code, source_unit, factor):
    result = convert_to_canonical(code, 1.0, source_unit)
    assert result.canonical_unit == UNITS["unit_conversions"][code]["canonical_unit"]
    assert result.value == pytest.approx(factor)
    assert result.conversion_applied is (factor != 1.0)


@pytest.mark.parametrize(("code", "source_unit", "factor"), _multiplicative_cases())
def test_every_multiplicative_conversion_round_trips(code, source_unit, factor):
    original = 7.5
    canonical = convert_to_canonical(code, original, source_unit).value
    assert canonical / factor == pytest.approx(original, rel=1e-9)


@pytest.mark.parametrize(("code", "source_unit", "scale", "offset"), _affine_cases())
def test_every_affine_conversion_round_trips(code, source_unit, scale, offset):
    original = 36.6
    result = convert_to_canonical(code, original, source_unit)
    assert result.canonical_unit == UNITS["affine_conversions"][code]["canonical_unit"]
    assert (result.value - offset) / scale == pytest.approx(original, rel=1e-6)


def test_canonical_unit_is_identity_for_every_variable():
    for code, spec in UNITS["unit_conversions"].items():
        canonical = spec["canonical_unit"]
        assert spec["from"][canonical] == 1.0, f"{code}: canonical unit must have factor 1.0"


# -- Known-value pins -----------------------------------------------------


def test_testosterone_nmol_per_litre_to_ng_per_decilitre():
    assert convert_to_canonical("total_testosterone", 1.0, "nmol/L").value == pytest.approx(28.818)


def test_glucose_mmol_per_litre_to_mg_per_decilitre():
    result = convert_to_canonical("fasting_glucose", 5.5, "mmol/L")
    assert result.value == pytest.approx(99.088, abs=0.01)
    assert result.canonical_unit == "mg/dL"


def test_fahrenheit_to_celsius_freezing_and_body_temperature():
    freezing = convert_to_canonical("skin_temperature", 32.0, "degF")
    assert freezing.value == pytest.approx(0.0, abs=1e-6)
    body = convert_to_canonical("skin_temperature", 98.6, "degF")
    assert body.value == pytest.approx(37.0, abs=1e-4)


def test_celsius_is_left_untouched():
    result = convert_to_canonical("skin_temperature", 36.5, "degC")
    assert result.value == pytest.approx(36.5)
    assert result.conversion_applied is False


def test_weight_pounds_to_kilograms():
    assert convert_to_canonical("weight", 154.0, "lb").value == pytest.approx(69.85, abs=0.01)


def test_height_inches_to_centimetres():
    assert convert_to_canonical("height", 64.0, "in").value == pytest.approx(162.56)


def test_lh_iu_per_litre_equals_miu_per_millilitre():
    assert convert_to_canonical("luteinizing_hormone", 12.0, "IU/L").value == pytest.approx(12.0)


def test_dheas_micromol_to_microgram_per_decilitre():
    assert convert_to_canonical("dheas", 5.0, "umol/L").value == pytest.approx(184.23, abs=0.01)


def test_estradiol_pmol_to_pg_per_millilitre():
    assert convert_to_canonical("estradiol", 100.0, "pmol/L").value == pytest.approx(27.24)


# -- Aliases and failure modes --------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("ng/dl", "ng/dL"),
        ("NG/DL", "ng/dL"),
        (" nmol/l ", "nmol/L"),
        ("µIU/mL", "uIU/mL"),
        ("°F", "degF"),
        ("%", "percent"),
        ("kg/m2", "kg/m^2"),
    ],
)
def test_unit_aliases_normalize(raw, expected):
    assert normalize_unit(raw) == expected


def test_blank_unit_normalizes_to_none():
    assert normalize_unit("   ") is None
    assert normalize_unit(None) is None


def test_unknown_unit_raises_rather_than_guessing():
    with pytest.raises(UnitConversionError):
        convert_to_canonical("total_testosterone", 1.0, "furlongs/fortnight")


def test_missing_unit_assumes_canonical_but_records_no_conversion():
    result = convert_to_canonical("total_testosterone", 40.0, None)
    assert result.value == 40.0
    assert result.conversion_applied is False
    assert result.source_unit is None


def test_variable_without_conversion_table_passes_value_through():
    result = convert_to_canonical("ferriman_gallwey_score", 8.0, None)
    assert result.value == 8.0
    assert math.isfinite(result.value)
