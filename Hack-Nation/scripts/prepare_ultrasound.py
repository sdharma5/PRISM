#!/usr/bin/env python
"""Prepare ovarian ultrasound studies: load, de-identify, validate, preprocess.

Config-driven. When the configured root does not exist — which is the normal case
in this repository, since no imaging data is committed — synthetic phantoms are
generated instead, so the preparation path is always exercised.

Writes ``prepared_manifest.json`` recording, per study, whether it was
de-identified, whether spacing was known, and every validation warning. That
manifest is the audit trail for which studies were eligible for measurement at
all.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running as a script from a checkout without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from typing import Any

import numpy as np
import yaml

from ingestion.ultrasound.loader import load_study
from ingestion.ultrasound.preprocessing import assert_transforms_allowed, preprocess_volume
from ingestion.ultrasound.validation import validate_study
from schemas.imaging import UltrasoundStudyMetadata
from scripts._cli import add_standard_arguments, make_parser, resolve_output_dir
from scripts._experiment_io import resolve_data_root

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "data" / "ultrasound.yaml"


def load_config(path: Path) -> dict[str, Any]:
    """Read a YAML config into a dict."""
    with Path(path).open() as fh:
        return yaml.safe_load(fh) or {}


LoadedStudies = list[tuple[np.ndarray, UltrasoundStudyMetadata, Any]]


def synthetic_studies(config: dict[str, Any]) -> LoadedStudies:
    """Generate phantom studies when no real dataset is present."""
    from tests.fixtures.synthetic_ultrasound import make_phantom  # noqa: PLC0415

    settings = config.get("synthetic_fallback", {}) or {}
    n = int(settings.get("n_studies", 6))
    seed = int(settings.get("seed", 0))
    out: LoadedStudies = []
    for index in range(n):
        # The volumetric phantom is the OPTIONAL enhanced mode; its settings
        # live under `volume:` now that 2D is the primary pathway.
        volume_cfg = settings.get("volume", settings) or {}
        phantom = make_phantom(
            shape=tuple(volume_cfg.get("shape", (48, 64, 64))),  # type: ignore[arg-type]
            spacing=tuple(volume_cfg.get("spacing_mm", (1.0, 0.6, 0.6))),  # type: ignore[arg-type]
            semi_axes_mm=tuple(volume_cfg.get("semi_axes_mm", (11.0, 15.0, 12.0))),  # type: ignore[arg-type]
            follicle_diameters_mm=tuple(
                volume_cfg.get("follicle_diameters_mm", (4.0, 5.0, 6.0, 7.0, 8.0, 9.0))
            ),
            seed=seed + index,
        )
        _, metadata = load_study(
            phantom.volume,
            patient_id=f"PHANTOM{index:03d}",
            study_id=f"PHANTOM{index:03d}_L",
            spacing_mm=phantom.spacing,
            laterality="left" if index % 2 == 0 else "right",
            route="transvaginal",
            source_dataset="synthetic_phantom",
            acquisition_mode="volume_3d",
        )
        out.append((phantom.volume, metadata, phantom))
    return out


def real_studies(root: Path, config: dict[str, Any]) -> LoadedStudies:
    """Load every study under ``root``, skipping ones that fail to load."""
    loading = config.get("loading", {}) or {}
    out: LoadedStudies = []
    for path in sorted(root.iterdir()):
        if path.name.startswith(".") or path.suffix.lower() == ".json":
            continue
        try:
            volume, metadata = load_study(
                path,
                source_dataset=config.get("source_dataset"),
                require_deidentified=bool(loading.get("require_deidentified", True)),
            )
        except Exception as exc:  # noqa: BLE001 - recorded, not silently swallowed
            print(f"  skipped {path.name}: {type(exc).__name__}: {exc}")
            continue
        out.append((volume, metadata, None))
    return out


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser. Exposed so the CLI contract test can inspect it."""
    parser = make_parser(description=__doc__)
    add_standard_arguments(
        parser, config_default=DEFAULT_CONFIG, seed=False, experiment_id=False, quiet=False
    )
    parser.add_argument(
        "--force-synthetic",
        action="store_true",
        help="Generate phantoms even when a real dataset is present.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    args = build_parser().parse_args(argv)

    config = load_config(args.config)
    preprocessing = config.get("preprocessing", {}) or {}
    transforms = preprocessing.get("transforms")
    if transforms:
        # Fails loudly on any count-destroying augmentation enabled in YAML.
        assert_transforms_allowed(transforms)

    output_dir = resolve_output_dir(
        config, args.output_dir, experiment_id="ultrasound", config_keys=("output.dir",)
    )
    root = resolve_data_root(config, args.data_root)
    use_synthetic = args.force_synthetic or root is None or not root.exists()
    if use_synthetic:
        print(f"No dataset at {root}; generating synthetic phantoms.")
        studies = synthetic_studies(config)
    else:
        studies = real_studies(root, config)

    loading = config.get("loading", {}) or {}
    records: list[dict[str, Any]] = []
    for volume, metadata, _ in studies:
        report = validate_study(
            volume,
            metadata,
            require_deidentified=bool(loading.get("require_deidentified", True)),
            require_spacing=bool(loading.get("require_spacing", False)),
        )
        prepared = preprocess_volume(
            volume,
            spacing_mm=metadata.spacing_mm,
            target_spacing_mm=preprocessing.get("target_spacing_mm"),
            target_shape=preprocessing.get("target_shape"),
            normalization=str(preprocessing.get("normalization", "percentile")),
        )
        records.append(
            {
                "study_id": metadata.study_id,
                "patient_id": metadata.patient_id,
                "laterality": metadata.laterality,
                "route": metadata.route,
                "deidentified": metadata.deidentified,
                "spacing_mm": list(metadata.spacing_mm) if metadata.spacing_mm else None,
                "measurement_spacing_mm": (
                    list(prepared.measurement_spacing_mm)
                    if prepared.measurement_spacing_mm
                    else None
                ),
                "shape": list(prepared.volume.shape),
                "valid": report.ok,
                "errors": report.errors,
                "warnings": sorted(set([*report.warnings, *prepared.warnings])),
                "transforms_applied": prepared.applied,
            }
        )

    manifest = {
        "config": str(args.config),
        "source": "synthetic_phantom" if use_synthetic else str(root),
        "n_studies": len(records),
        "n_valid": sum(1 for r in records if r["valid"]),
        "n_without_spacing": sum(1 for r in records if r["spacing_mm"] is None),
        "studies": records,
    }
    path = output_dir / "prepared_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"Prepared {manifest['n_studies']} studies "
        f"({manifest['n_valid']} valid, {manifest['n_without_spacing']} without spacing) "
        f"-> {path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
