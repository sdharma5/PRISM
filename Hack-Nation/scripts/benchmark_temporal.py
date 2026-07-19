"""Run the temporal baseline ladder on the frozen participant split.

This establishes what a sequence model must beat. It is deliberately run BEFORE
any GRU is trained: on 42 participants the honest question is whether dynamics
modelling helps at all, and that question is unanswerable once a neural result
already exists to anchor on.

Every metric is computed on **observed days only**, per channel. PdG is present
on roughly a third of days, so pooling it with imputed values would report the
imputation's error rather than the model's.

Usage::

    python scripts/benchmark_temporal.py
    python scripts/benchmark_temporal.py --lookback 21 \
        --output-dir artifacts/experiments/exp_temporal_baselines
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scipy import stats as scipy_stats  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)

from ingestion.mcphases.splits import (  # noqa: E402
    assert_no_participant_leakage,
    load_participant_days,
)
from models.temporal.baselines import (  # noqa: E402
    GlobalMeanBaseline,
    LocfBaseline,
    LogisticPhaseBaseline,
    MajorityPhaseBaseline,
    ParticipantHistoryMeanBaseline,
    RidgeWindowBaseline,
    build_sequences,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_DAYS = Path("../datasets/mcphases/raw/participant_days.jsonl")
DEFAULT_SPLIT = Path("artifacts/splits/mcphases_participant_split_v1.json")
DEFAULT_OUTPUT = Path("artifacts/experiments/exp_temporal_baselines")

#: Hormone targets. Wearable channels are inputs, not targets.
HORMONES = ("lh", "e3g", "pdg")
CHANNELS = (
    "lh",
    "e3g",
    "pdg",
    "resting_heart_rate",
    "wrist_temperature",
    "hrv_rmssd",
    "mean_glucose",
)


def resolve(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def hormone_metrics(
    truth: np.ndarray, prediction: np.ndarray, observed: np.ndarray
) -> dict[str, float]:
    """MAE / RMSE / Spearman on observed days only."""
    if observed.sum() < 2:
        return {"mae": float("nan"), "rmse": float("nan"), "spearman": float("nan"), "n": 0}

    y = truth[observed]
    p = prediction[observed]
    finite = np.isfinite(y) & np.isfinite(p)
    y, p = y[finite], p[finite]
    if y.size < 2:
        return {
            "mae": float("nan"),
            "rmse": float("nan"),
            "spearman": float("nan"),
            "n": int(y.size),
        }

    spearman = float("nan")
    if np.std(p) > 1e-9 and np.std(y) > 1e-9:
        spearman = float(scipy_stats.spearmanr(y, p).statistic)

    return {
        "mae": float(np.mean(np.abs(y - p))),
        "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
        "spearman": spearman,
        "n": int(y.size),
    }


def per_participant_mae(
    truth: np.ndarray,
    prediction: np.ndarray,
    observed: np.ndarray,
    participants: np.ndarray,
) -> dict[str, float]:
    """MAE per held-out participant.

    Pooled day-level means hide the case where one participant with many days
    dominates and another is predicted badly throughout.
    """
    results: dict[str, float] = {}
    for pid in np.unique(participants):
        rows = (participants == pid) & observed
        if rows.sum() == 0:
            continue
        y, p = truth[rows], prediction[rows]
        finite = np.isfinite(y) & np.isfinite(p)
        if finite.sum():
            results[str(pid)] = float(np.mean(np.abs(y[finite] - p[finite])))
    return results


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
    if not days_path.exists():
        print(f"ERROR: no participant-day table at {days_path}", file=sys.stderr)
        return 1
    if not split_path.exists():
        print(f"ERROR: no split manifest at {split_path}", file=sys.stderr)
        return 1

    rows = load_participant_days(days_path)
    manifest = json.loads(split_path.read_text())
    assert_no_participant_leakage(manifest)

    print(
        f"cohort: {manifest['n_participants']} participants, {manifest['n_days']} days\n"
        f"split:  train {len(manifest['train_ids'])} / val {len(manifest['validation_ids'])} "
        f"/ test {len(manifest['test_ids'])} participants\n"
        f"lookback: {args.lookback} days\n"
    )

    train = build_sequences(rows, manifest["train_ids"], channels=CHANNELS, lookback=args.lookback)
    test = build_sequences(rows, manifest["test_ids"], channels=CHANNELS, lookback=args.lookback)

    print(f"train days {len(train['values'])} | test days {len(test['values'])}")
    for index, name in enumerate(CHANNELS):
        if name in HORMONES:
            print(
                f"  {name:20s} observed: train {int(train['observed'][:, index].sum()):5d}  "
                f"test {int(test['observed'][:, index].sum()):5d}"
            )
    print()

    # -- hormone ladder ----------------------------------------------------
    hormone_models = [
        GlobalMeanBaseline(),
        LocfBaseline(),
        ParticipantHistoryMeanBaseline(),
        RidgeWindowBaseline(alpha=10.0),
    ]

    results: dict[str, Any] = {"hormones": {}, "cycle_phase": {}}

    print("=== hormone reconstruction (held-out participants, observed days only) ===")
    header = f"{'model':28s}" + "".join(f"{h:>22s}" for h in HORMONES)
    print(header)
    print("-" * len(header))

    for model in hormone_models:
        model.fit(train)
        prediction = model.predict(test)
        entry: dict[str, Any] = {}
        line = f"{model.name:28s}"
        for hormone in HORMONES:
            index = CHANNELS.index(hormone)
            metrics = hormone_metrics(
                test["values"][:, index], prediction[:, index], test["observed"][:, index]
            )
            metrics["per_participant_mae"] = per_participant_mae(
                test["values"][:, index],
                prediction[:, index],
                test["observed"][:, index],
                test["participant"],
            )
            entry[hormone] = metrics
            line += f"  MAE {metrics['mae']:7.3f} r{metrics['spearman']:+.2f}"
        results["hormones"][model.name] = entry
        print(line)

    # -- cycle-phase ladder ------------------------------------------------
    print("\n=== cycle phase (held-out participants) ===")
    labelled_test = test["phase"] != "unknown"
    print(f"labelled test days: {int(labelled_test.sum())} of {len(test['phase'])}")

    for model in (MajorityPhaseBaseline(), LogisticPhaseBaseline()):
        model.fit(train)
        prediction = model.predict(test)
        y_true = test["phase"][labelled_test]
        y_pred = prediction[labelled_test]

        classes = sorted(set(y_true))
        entry = {
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "classes": classes,
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=classes).tolist(),
            "n": int(labelled_test.sum()),
        }
        results["cycle_phase"][model.name] = entry
        print(
            f"  {model.name:28s} macro F1 {entry['macro_f1']:.4f}  "
            f"balanced acc {entry['balanced_accuracy']:.4f}"
        )

    # -- persist -----------------------------------------------------------
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "split_manifest": str(split_path),
        "lookback_days": args.lookback,
        "n_train_participants": len(manifest["train_ids"]),
        "n_test_participants": len(manifest["test_ids"]),
        "channels": list(CHANNELS),
        "results": results,
        "caveats": [
            "Held-out participants, not held-out days. Metrics describe generalisation "
            "to a NEW person.",
            "8 test participants: these are point estimates with wide uncertainty. "
            "Per-participant MAE is reported alongside the pooled value.",
            "Hormone metrics are computed on observed days only. PdG is observed on "
            "roughly a third of days and its n is correspondingly smaller.",
            "No sequence model is included here. This is the bar a GRU must clear.",
            "Symptom targets are omitted: the `headaches` column mixes ordinal labels "
            "with bare numerals and has no defensible mapping yet.",
        ],
    }
    (output_dir / "baseline_metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {output_dir / 'baseline_metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
