"""Fit and persist the target-specific temporal state encoder.

Trains on the frozen participant split, evaluates on the untouched held-out
participants, and writes a loadable artifact. The per-target model selection is
inherited from the frozen benchmark rather than re-chosen here -- picking the
winner against the test split would make the reported numbers in-sample.

Usage::

    python scripts/train_temporal_state_encoder.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingestion.mcphases.splits import (  # noqa: E402
    assert_no_participant_leakage,
    fit_normalization_stats,
    load_participant_days,
)
from models.temporal.state_encoder import TemporalStateEncoder  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DAYS = Path("../datasets/mcphases/raw/participant_days.jsonl")
DEFAULT_SPLIT = Path("artifacts/splits/mcphases_participant_split_v1.json")
DEFAULT_OUTPUT = Path("artifacts/encoders/temporal_state_v1")


def resolve(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=Path, default=DEFAULT_DAYS)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--lookback", type=int, default=14)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    days_path = resolve(args.days)
    split_path = resolve(args.split)
    if not days_path.exists() or not split_path.exists():
        print(f"ERROR: need {days_path} and {split_path}", file=sys.stderr)
        return 1

    rows = load_participant_days(days_path)
    manifest = json.loads(split_path.read_text())
    assert_no_participant_leakage(manifest)

    print(
        f"train {len(manifest['train_ids'])} / val {len(manifest['validation_ids'])} "
        f"/ test {len(manifest['test_ids'])} participants"
    )

    stats = fit_normalization_stats(rows, manifest["train_ids"])
    encoder = TemporalStateEncoder(lookback=args.lookback).fit(
        rows, manifest["train_ids"], normalization_stats=stats
    )

    print("\nper-target model selection (from the frozen benchmark):")
    for hormone, method in encoder.target_models.items():
        print(f"  {hormone:6s} -> {method}")
    print("  phase  -> logistic_phase")

    test_metrics = encoder.evaluate(rows, manifest["test_ids"])
    print("\nheld-out participants:")
    for hormone, values in test_metrics["hormones"].items():
        print(
            f"  {hormone:6s} [{values['method']:13s}] MAE {values['mae']:8.3f}  "
            f"n_observed {values['n_observed']}"
        )
    if test_metrics["cycle_phase"]:
        print(
            f"  phase  [logistic     ] macro F1 {test_metrics['cycle_phase']['macro_f1']:.4f}  "
            f"balanced acc {test_metrics['cycle_phase']['balanced_accuracy']:.4f}"
        )

    assert encoder.artifact is not None
    encoder.artifact.metrics = test_metrics

    output_dir = resolve(args.output_dir)
    saved = encoder.save(output_dir)
    (output_dir / "config.yaml").write_text(
        "# Resolved configuration for the persisted temporal state encoder.\n"
        f"lookback_days: {args.lookback}\n"
        f"split_manifest: {split_path.name}\n"
        "task_type: causal_forecasting\n"
        "target_models:\n"
        + "".join(f"  {k}: {v}\n" for k, v in encoder.target_models.items())
        + "  cycle_phase: logistic_phase\n"
        "excluded_targets:\n"
        "  headaches: >-\n"
        "    Mixed ordinal labels and bare numerals ('2','3','High') with no\n"
        "    defensible source mapping. Excluded rather than blindly factorized.\n"
    )
    print(f"\nsaved -> {saved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
