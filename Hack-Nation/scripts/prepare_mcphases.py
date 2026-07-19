#!/usr/bin/env python
"""Build participant-day tables from the raw mcPHASES dataset.

    python scripts/prepare_mcphases.py --config configs/data/mcphases.yaml --data-root /path/to/data

mcPHASES requires credentialed PhysioNet access and is never committed to this
repository, so unlike the training scripts this one has NO synthetic fallback.
Preparing data is a claim that real data exists; quietly substituting phantoms
here would write a manifest describing a dataset nobody has. When the dataset is
absent the script fails with the path it looked for and the environment variable
that would change it.

Writes ``participant_days.csv`` plus the full ingestion audit bundle (raw
manifest, checksums, variable mapping, validation report, dropped records,
processing config, processed manifest).
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

from ingestion.mcphases.loader import (  # noqa: E402
    STREAM_FILE_CANDIDATES,
    McPhasesAdapter,
    McPhasesDataNotFoundError,
    resolve_stream_files,
    to_frame,
)
from scripts._cli import (  # noqa: E402
    DATA_ROOT_ENV,
    add_standard_arguments,
    make_parser,
    resolve_output_dir,
)
from scripts._experiment_io import resolve_data_root  # noqa: E402

DEFAULT_CONFIG = REPO_ROOT / "configs" / "data" / "mcphases.yaml"


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
    root = resolve_data_root(config, args.data_root)

    if root is None or not root.exists():
        # Fail loudly and name every lever the user has. A prepare script that
        # succeeds on missing data produces an empty table that looks like a
        # result, and the emptiness is not noticed until modelling.
        looked_for = root if root is not None else "(nothing configured)"
        print(
            f"ERROR: mcPHASES dataset not found at: {looked_for}\n"
            "\n"
            "mcPHASES requires CREDENTIALED PhysioNet access. It is not public, it is "
            "never committed to this repository, and there is no synthetic substitute "
            "for this step.\n"
            "\n"
            "To fix this, obtain the dataset under its own access terms, store it "
            "outside the repository tree, then do ONE of:\n"
            f"  1. export {DATA_ROOT_ENV}=/path/to/data   (nothing auto-loads .env — "
            "see .env.example)\n"
            "  2. pass --data-root /path/to/data/mcphases\n"
            f"  3. set `data.root` in {args.config}\n"
            "\n"
            f"The directory must contain one of {', '.join(STREAM_FILE_CANDIDATES)} "
            "or any *.csv stream table.",
            file=sys.stderr,
        )
        return 1

    output_dir = resolve_output_dir(
        config, args.output_dir, experiment_id="mcphases", config_keys=("output.dir",)
    )

    data_cfg = config.get("data", {}) or {}
    adapter = McPhasesAdapter(
        dataset_version=str(config.get("dataset_version", "unversioned")),
        use=str((config.get("allowed_uses") or ["temporal_state_model"])[0]),
    )

    try:
        days = adapter.run(_stream_table(root), strict=bool(args.strict))
    except McPhasesDataNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    frame = to_frame(days)
    table_path = output_dir / "participant_days.csv"
    frame.to_csv(table_path, index=False)

    written = adapter.write_manifest_artifacts(
        output_dir,
        raw_manifest={
            "dataset_id": adapter.dataset_id,
            "root": str(root),
            "n_source_records": adapter.n_source_records,
        },
        processing_config={"config_path": str(args.config), "data": data_cfg},
    )

    n_participants = frame["participant_id"].nunique() if not frame.empty else 0
    if not args.quiet:
        print(f"Loaded mcPHASES from {root}")
        print(f"Built {len(frame)} participant-days across {n_participants} participants.")
        for warning in adapter.warnings[:10]:
            print(f"  WARNING: {warning}")
        print(f"Wrote {table_path}")
        for name, path in sorted(written.items()):
            print(f"  {name}: {path}")
    return 0


def _stream_table(root: Path) -> Path:
    """Pick the single stream table to ingest from an mcPHASES root.

    Delegates resolution to :func:`resolve_stream_files` rather than repeating
    it. The two implementations had already drifted apart: this one selected
    ``sorted(root.glob("*.csv"))[0]``, so a root holding several stream tables
    was silently ingested as whichever sorted first.
    """
    files = resolve_stream_files(root)
    if len(files) > 1:
        raise McPhasesDataNotFoundError(
            f"mcPHASES root '{root}' holds {len(files)} stream tables:\n"
            + "\n".join(f"  {p.name}" for p in files)
            + "\n\nThis script ingests exactly one. Point --data-root at a single file, "
            "or consolidate them first."
        )
    return files[0]


if __name__ == "__main__":
    raise SystemExit(main())
