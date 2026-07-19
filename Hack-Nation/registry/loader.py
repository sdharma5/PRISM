"""Loaders for the YAML registries, plus the unit-conversion engine.

Everything that touches units goes through :func:`convert_to_canonical`. There
are no ad-hoc conversion constants anywhere else in the codebase.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from schemas.dataset import DatasetRegistry, VariableRegistry

REGISTRY_DIR = Path(__file__).resolve().parent


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open() as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping at the top level.")
    return data


@lru_cache(maxsize=1)
def load_dataset_registry(path: Path | None = None) -> DatasetRegistry:
    return DatasetRegistry.model_validate(_read_yaml(path or REGISTRY_DIR / "datasets.yaml"))


@lru_cache(maxsize=1)
def load_variable_registry(path: Path | None = None) -> VariableRegistry:
    return VariableRegistry.model_validate(_read_yaml(path or REGISTRY_DIR / "variables.yaml"))


@lru_cache(maxsize=1)
def load_units(path: Path | None = None) -> dict[str, Any]:
    return _read_yaml(path or REGISTRY_DIR / "units.yaml")


@lru_cache(maxsize=1)
def load_phenotype_domains(path: Path | None = None) -> dict[str, Any]:
    return _read_yaml(path or REGISTRY_DIR / "phenotype_domains.yaml")


@lru_cache(maxsize=1)
def load_schema_versions(path: Path | None = None) -> dict[str, Any]:
    return _read_yaml(path or REGISTRY_DIR / "schema_versions.yaml")


class UnitConversionError(ValueError):
    """Raised when a unit cannot be converted. Never silently passed through."""


@dataclass(frozen=True)
class ConversionResult:
    value: float
    canonical_unit: str
    source_unit: str | None
    conversion_applied: bool


def normalize_unit(unit: str | None) -> str | None:
    """Map a messy source unit string onto its registry alias."""
    if unit is None:
        return None
    stripped = unit.strip()
    if not stripped:
        return None
    aliases: dict[str, str] = load_units().get("aliases", {})
    if stripped in aliases:
        return aliases[stripped]
    lowered = stripped.lower()
    for raw, canonical in aliases.items():
        if raw.lower() == lowered:
            return canonical
    return stripped


def convert_to_canonical(code: str, value: float, unit: str | None) -> ConversionResult:
    """Convert ``value`` for canonical variable ``code`` into its canonical unit.

    Raises :class:`UnitConversionError` rather than guessing. A wrong silent
    conversion is far more damaging than a loud failure.
    """
    units = load_units()
    source_unit = normalize_unit(unit)

    affine = units.get("affine_conversions", {}).get(code)
    if affine is not None:
        canonical_unit = affine["canonical_unit"]
        if source_unit is None:
            return ConversionResult(float(value), canonical_unit, None, False)
        rule = affine["from"].get(source_unit)
        if rule is None:
            raise UnitConversionError(
                f"{code}: no affine conversion from '{source_unit}' to '{canonical_unit}'."
            )
        converted = float(value) * float(rule["scale"]) + float(rule["offset"])
        return ConversionResult(
            converted, canonical_unit, source_unit, source_unit != canonical_unit
        )

    spec = units.get("unit_conversions", {}).get(code)
    if spec is None:
        # No conversion table: the variable is unitless or carries its unit as-is.
        variables = load_variable_registry().variables
        canonical_unit = None
        if code in variables:
            canonical_unit = variables[code].canonical_unit or variables[code].unit
        return ConversionResult(
            float(value), canonical_unit or (source_unit or ""), source_unit, False
        )

    canonical_unit = spec["canonical_unit"]
    if source_unit is None:
        # Assume canonical, but the caller records that no conversion happened.
        return ConversionResult(float(value), canonical_unit, None, False)

    factor = spec["from"].get(source_unit)
    if factor is None:
        raise UnitConversionError(
            f"{code}: no conversion from '{source_unit}' to '{canonical_unit}'. "
            f"Known source units: {sorted(spec['from'])}."
        )
    return ConversionResult(
        float(value) * float(factor), canonical_unit, source_unit, float(factor) != 1.0
    )


def in_valid_range(code: str, value: float) -> bool:
    """True when ``value`` falls inside the registry's plausible range."""
    spec = load_variable_registry().variables.get(code)
    if spec is None or spec.valid_range is None:
        return True
    return spec.valid_range.contains(float(value))
