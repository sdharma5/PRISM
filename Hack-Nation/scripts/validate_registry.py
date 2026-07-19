#!/usr/bin/env python3
"""Validate every registry file and their cross-references.

This runs in CI on every change to ``schemas/``, ``registry/`` or ``ingestion/``.
It is intentionally strict: a registry that quietly disagrees with itself is how
a unit error or an unsupported claim reaches a result table.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from registry.loader import (  # noqa: E402
    load_dataset_registry,
    load_phenotype_domains,
    load_schema_versions,
    load_units,
    load_variable_registry,
)


class RegistryValidationError(Exception):
    pass


def _check_units_reference_variables(errors: list[str]) -> None:
    units = load_units()
    variables = load_variable_registry().variables

    for code, spec in units.get("unit_conversions", {}).items():
        if code not in variables:
            errors.append(f"units.yaml: '{code}' has conversions but is not in variables.yaml")
            continue
        declared = variables[code].canonical_unit or variables[code].unit
        if declared and declared != spec["canonical_unit"]:
            errors.append(
                f"units.yaml: '{code}' canonical_unit '{spec['canonical_unit']}' disagrees with "
                f"variables.yaml '{declared}'"
            )
        if spec["canonical_unit"] not in spec["from"]:
            errors.append(
                f"units.yaml: '{code}' must include an identity conversion for its canonical unit "
                f"'{spec['canonical_unit']}'"
            )
        elif float(spec["from"][spec["canonical_unit"]]) != 1.0:
            errors.append(f"units.yaml: '{code}' identity conversion must have factor 1.0")

    for code, spec in units.get("affine_conversions", {}).items():
        if code not in variables:
            errors.append(f"units.yaml: affine '{code}' is not in variables.yaml")
        identity = spec["from"].get(spec["canonical_unit"])
        if not identity or identity["scale"] != 1.0 or identity["offset"] != 0.0:
            errors.append(
                f"units.yaml: affine '{code}' needs an identity rule for its canonical unit"
            )


def _check_domains_reference_variables(errors: list[str]) -> None:
    domains = load_phenotype_domains()
    variables = load_variable_registry().variables

    for domain_name, domain in domains.get("domains", {}).items():
        features = domain.get("features", [])
        if not features:
            errors.append(f"phenotype_domains.yaml: domain '{domain_name}' has no features")
        for feature in features:
            code = feature["code"]
            if code not in variables:
                errors.append(
                    f"phenotype_domains.yaml: domain '{domain_name}' references unknown "
                    f"variable '{code}'"
                )
            if feature.get("direction") not in (1, -1):
                errors.append(
                    f"phenotype_domains.yaml: '{domain_name}.{code}' direction must be 1 or -1"
                )
            if float(feature.get("weight", 0)) <= 0:
                errors.append(f"phenotype_domains.yaml: '{domain_name}.{code}' weight must be > 0")
            if feature.get("evidence_class") not in {
                "biochemical",
                "report",
                "measurement",
                "imaging",
                "anthropometric",
            }:
                errors.append(
                    f"phenotype_domains.yaml: '{domain_name}.{code}' has an unknown evidence_class"
                )

        # A domain that can be scored from symptoms alone must say how to label it.
        classes = {f.get("evidence_class") for f in features}
        if "report" in classes and len(classes) > 1 and not domain.get("symptom_only_qualifier"):
            errors.append(
                f"phenotype_domains.yaml: '{domain_name}' mixes reported and measured evidence "
                "and must define symptom_only_qualifier, so a score that happened to come from "
                "symptoms alone is never presented as measured evidence"
            )

        # The androgenic axis is split precisely so that an absent assay cannot
        # drag symptom evidence below its coverage floor. That guarantee only
        # holds while each half stays pure: one mixed feature would put assay
        # weight back into the clinical denominator.
        if domain_name == "clinical_androgenic_evidence" and classes != {"report"}:
            errors.append(
                "phenotype_domains.yaml: 'clinical_androgenic_evidence' must contain only "
                f"report-class features, found {sorted(c for c in classes if c)}. Any assay "
                "weight here re-creates the bug where a missing androgen panel made observed "
                "cutaneous signs unassessable."
            )
        if domain_name == "biochemical_androgenic_evidence" and classes != {"biochemical"}:
            errors.append(
                "phenotype_domains.yaml: 'biochemical_androgenic_evidence' must contain only "
                f"biochemical-class features, found {sorted(c for c in classes if c)}. "
                "Biochemical hyperandrogenism is never inferred from cutaneous signs."
            )

        if str(domain.get("evidence_source", "")) not in {
            "symptoms",
            "biochemical",
            "imaging",
            "mixed",
        }:
            errors.append(
                f"phenotype_domains.yaml: '{domain_name}' must declare evidence_source as one "
                "of symptoms|biochemical|imaging|mixed, so a consumer can tell what kind of "
                "evidence a score is made of without re-deriving it from the feature list"
            )


def _check_dataset_claims(errors: list[str]) -> None:
    registry = load_dataset_registry()
    for dataset_id, spec in registry.datasets.items():
        if not spec.prohibited_claims:
            errors.append(
                f"datasets.yaml: '{dataset_id}' must state prohibited_claims — an empty list "
                "reads as 'anything goes'"
            )
        # Cross-sectional datasets can never support longitudinal state modeling.
        if not spec.longitudinal and "temporal_state_model" in spec.allowed_uses:
            errors.append(
                f"datasets.yaml: '{dataset_id}' is not longitudinal but allows "
                "'temporal_state_model'"
            )


def _check_schema_versions(errors: list[str]) -> None:
    import importlib

    ledger = load_schema_versions()
    for key, entry in ledger.get("schemas", {}).items():
        module_name = entry["module"]
        try:
            importlib.import_module(module_name)
        except ImportError as exc:  # pragma: no cover - import failure is the finding
            errors.append(f"schema_versions.yaml: cannot import '{module_name}' for '{key}': {exc}")

    event_module = importlib.import_module("schemas.event")
    declared = ledger["schemas"]["event"]["version"]
    if declared != event_module.SCHEMA_VERSION:
        errors.append(
            f"schema_versions.yaml declares event schema {declared} but schemas/event.py says "
            f"{event_module.SCHEMA_VERSION}. Bump both and add a CHANGELOG entry."
        )


def validate() -> list[str]:
    """Run every registry check and return the collected error messages."""
    errors: list[str] = []

    # Pydantic validation happens on load; surface it as a readable error.
    try:
        load_dataset_registry()
        load_variable_registry()
    except Exception as exc:
        errors.append(f"registry failed schema validation: {exc}")
        return errors

    _check_units_reference_variables(errors)
    _check_domains_reference_variables(errors)
    _check_dataset_claims(errors)
    _check_schema_versions(errors)
    return errors


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser. Exposed so the CLI contract test can inspect it."""
    return argparse.ArgumentParser(description=__doc__)


def main(argv: list[str] | None = None) -> int:
    build_parser().parse_args(argv)
    errors = validate()
    if errors:
        print(f"\033[31mRegistry validation failed with {len(errors)} error(s):\033[0m")
        for err in errors:
            print(f"  - {err}")
        return 1

    datasets = load_dataset_registry().datasets
    variables = load_variable_registry().variables
    domains = load_phenotype_domains()["domains"]
    print("Registry validation passed.")
    print(f"  datasets:  {len(datasets)}")
    print(f"  variables: {len(variables)}")
    print(f"  domains:   {len(domains)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
