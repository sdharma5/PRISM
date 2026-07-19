#!/usr/bin/env python
"""Ingest the public PCOS tabular cohort into canonical events.

    python scripts/prepare_pcos_tabular.py --config configs/data/pcos_tabular.yaml

Reads the source CSV through :class:`PcosTabularAdapter`, which validates it,
records checksums, maps source columns onto registry variable codes, normalizes
units, and drops what it cannot map — recording every drop with a reason rather
than silently discarding it.

Like ``prepare_mcphases.py`` this has NO synthetic fallback. The training scripts
synthesize a cohort when no dataset is configured, which is what makes a fresh
clone runnable; a *preparation* step that synthesized would instead write a
manifest and a checksum set describing data that does not exist.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ingestion.tabular_pcos.loader import PcosTabularAdapter  # noqa: E402
from scripts._cli import (  # noqa: E402
    DATA_ROOT_ENV,
    add_standard_arguments,
    make_parser,
    resolve_output_dir,
)
from scripts._experiment_io import resolve_data_path  # noqa: E402

DEFAULT_CONFIG = REPO_ROOT / "configs" / "data" / "pcos_tabular.yaml"


def load_config(path: Path) -> dict[str, Any]:
    """Read a YAML config into a dict."""
    with Path(path).open() as fh:
        return yaml.safe_load(fh) or {}


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser. Exposed so the CLI contract test can inspect it."""
    parser = make_parser(description=__doc__)
    add_standard_arguments(parser, config_default=DEFAULT_CONFIG, seed=False, experiment_id=False)
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail on validation errors instead of recording them as warnings.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    args = build_parser().parse_args(argv)

    config = load_config(args.config)
    data_cfg = config.get("data", {}) or {}
    source = resolve_data_path(config, args.data_root)

    if source is None or not source.exists() or source.is_dir():
        # Name the exact file, every override, and where to get the data.
        looked_for = source if source is not None else "(nothing configured)"
        print(
            f"ERROR: PCOS tabular dataset not found at: {looked_for}\n"
            "\n"
            "This dataset is publicly available but is NOT committed to this repository. "
            "There is no synthetic substitute for this step.\n"
            "\n"
            "To fix this, download the cohort CSV (see DATASET_REGISTRY.md and "
            "registry/datasets.yaml for the source and its terms), store it outside the "
            "repository tree, then do ONE of:\n"
            f"  1. export {DATA_ROOT_ENV}=/path/to/data   (nothing auto-loads .env — "
            "see .env.example)\n"
            "  2. pass --data-root /path/to/data\n"
            f"  3. set `data.path` in {args.config}\n"
            "\n"
            "`data.path` is resolved relative to the data root when it is not absolute.",
            file=sys.stderr,
        )
        return 1

    output_dir = resolve_output_dir(
        config, args.output_dir, experiment_id="pcos_tabular", config_keys=("output.dir",)
    )

    adapter = PcosTabularAdapter(
        dataset_version=str(config.get("dataset_version", "unversioned")),
        id_column=str(data_cfg.get("id_column", "patient_id")),
        use=str((config.get("allowed_uses") or ["binary_baseline"])[0]),
    )
    events = adapter.run(source, strict=bool(args.strict))

    written = adapter.write_manifest_artifacts(
        output_dir,
        raw_manifest={
            "dataset_id": adapter.dataset_id,
            "source_path": str(source),
            "n_source_records": adapter.n_source_records,
        },
        processing_config={"config_path": str(args.config), "data": data_cfg},
        events=events,
    )

    if not args.quiet:
        print(f"Ingested {source}")
        print(f"Emitted {len(events)} events from {adapter.n_source_records} source records.")
        if adapter.validation_errors:
            for problem in adapter.validation_errors[:10]:
                print(f"  VALIDATION: {problem}")
        for warning in adapter.warnings[:10]:
            print(f"  WARNING: {warning}")
        for name, path in sorted(written.items()):
            print(f"  {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
