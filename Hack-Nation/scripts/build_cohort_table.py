#!/usr/bin/env python
"""Pivot ingested PMOS tabular events into the wide cohort table the models read.

    python scripts/build_cohort_table.py --config configs/data/pmos_tabular.yaml

Why this exists
---------------
``prepare_pmos_tabular.py`` validates the source, maps it onto registry variable
codes, normalizes units, and writes an audit trail — but it writes *manifests
only*. The canonical events it builds are discarded. Meanwhile
``train_static_baselines.py`` reads a **wide** CSV keyed by ``patient_id`` whose
columns are canonical codes (see ``tests/fixtures/synthetic_tabular.py``).

Nothing connected the two, so on real data the training scripts silently fell
back to the synthetic fixture cohort. This script is that missing bridge.

Why a plain pivot rather than ``latest_per_code``
------------------------------------------------
``event_store.queries.latest_per_code`` resolves repeated measures by
observation time and **ignores undated events**. This cohort is cross-sectional:
one row per patient, no timestamps, so every event is undated and that helper
would return nothing at all. One value per (patient, code) is an invariant here,
and it is asserted rather than assumed — a duplicate means the source had
repeated measures and this script is the wrong tool for it.

Missingness is written as an empty cell (real NaN on read back), never as 0.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ingestion.tabular_pmos.loader import PmosTabularAdapter  # noqa: E402
from scripts._cli import (  # noqa: E402
    DATA_ROOT_ENV,
    add_standard_arguments,
    make_parser,
)
from scripts._experiment_io import resolve_data_path  # noqa: E402

DEFAULT_CONFIG = REPO_ROOT / "configs" / "data" / "pmos_tabular.yaml"


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser, following the shared CLI contract."""
    parser = make_parser(description=__doc__)
    add_standard_arguments(parser, config_default=DEFAULT_CONFIG, seed=False, experiment_id=False)
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail on validation errors instead of recording them as warnings.",
    )
    return parser


def cohort_frame(events: list[Any]) -> pd.DataFrame:
    """Pivot canonical events into one row per patient, one column per code."""
    records = [
        {
            "patient_id": e.patient_id,
            "code": e.canonical_variable_code,
            "value": e.value,
        }
        for e in events
        if e.missingness_status == "observed" and e.value is not None
    ]
    if not records:
        raise ValueError("No observed events to pivot; refusing to write an empty cohort.")

    long = pd.DataFrame.from_records(records)

    duplicated = long.duplicated(subset=["patient_id", "code"], keep=False)
    if bool(duplicated.any()):
        offenders = long.loc[duplicated, ["patient_id", "code"]].drop_duplicates()
        raise ValueError(
            "Expected exactly one value per (patient_id, canonical code) for this "
            f"cross-sectional cohort, found {len(offenders)} duplicated pairs, e.g.\n"
            f"{offenders.head(5).to_string(index=False)}\n"
            "Repeated measures need a documented resolution rule, not a silent pick."
        )

    wide = long.pivot(index="patient_id", columns="code", values="value").reset_index()
    wide.columns.name = None
    return wide


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    args = build_parser().parse_args(argv)

    config = yaml.safe_load(Path(args.config).read_text()) or {}
    data_cfg = config.get("data", {}) or {}
    source = resolve_data_path(config, args.data_root)

    if source is None or not source.exists() or source.is_dir():
        print(
            f"ERROR: PMOS tabular dataset not found at: {source or '(nothing configured)'}\n"
            f"Set {DATA_ROOT_ENV}, pass --data-root, or set `data.path` in {args.config}.",
            file=sys.stderr,
        )
        return 1

    adapter = PmosTabularAdapter(
        dataset_version=str(config.get("dataset_version", "unversioned")),
        id_column=str(data_cfg.get("id_column", "Patient File No.")),
        use=str((config.get("allowed_uses") or ["binary_baseline"])[0]),
    )
    events = adapter.run(source, strict=bool(args.strict))
    wide = cohort_frame(events)

    destination = Path(args.output_dir) if args.output_dir else source.with_name("cohort_wide.csv")
    if destination.is_dir():
        destination = destination / "cohort_wide.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    wide.to_csv(destination, index=False)

    label = str(data_cfg.get("label_column", ""))
    print(f"Wrote {destination}")
    print(f"  {len(wide)} patients x {wide.shape[1]} columns (from {len(events)} events)")
    observed = wide.notna().mean().sort_values()
    print(f"  sparsest columns: {dict(observed.head(3).round(3))}")
    if "pmos_binary" in wide.columns:
        print(f"  pmos_binary counts: {wide['pmos_binary'].value_counts().to_dict()}")
    elif label:
        print(f"  NOTE: source label '{label}' did not survive as `pmos_binary`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
