"""Train and PERSIST the static clinical encoder.

Distinct from ``train_static_baselines.py``, which cross-validates a family of
models to answer "how well would this generalise?" and then discards them. This
script answers a different question -- "what artifact scores a new patient?" --
and its whole point is the file it leaves behind.

Protocol:

* A patient-level held-out test split is reserved FIRST and never touched during
  fitting, so the reported metrics describe unseen patients.
* Cross-validation on the training portion gives a stability estimate.
* The final artifact is refit on the full training portion and saved with its
  preprocessing and domain reference statistics.

The held-out metrics are the honest ones. They are computed on one clinic's
cross-sectional cohort and describe how well the model reproduces that clinic's
recorded PCOS label -- not how well it diagnoses PCOS.

Usage::

    python scripts/train_static_encoder.py
    python scripts/train_static_encoder.py --test-size 0.25 --seed 7
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split  # noqa: E402

from evaluation.calibration import (  # noqa: E402
    ALLOWED_CALIBRATION_FIT_SOURCE,
    PlattCalibrator,
    simplified_calibration_report,
)
from models.tabular.encoder import StaticClinicalEncoder  # noqa: E402

DEFAULT_COHORT = Path("../datasets/pcos_tabular/cohort_wide.csv")
DEFAULT_OUTPUT = Path("artifacts/encoders/static_clinical")


def evaluate(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    """Discrimination and calibration together.

    AUROC alone is not enough: a model can rank patients well and still emit
    probabilities that are systematically wrong, and the adapter reports the
    probability, not the ranking. Brier score catches that.
    """
    return {
        "auroc": float(roc_auc_score(y_true, probabilities)),
        "auprc": float(average_precision_score(y_true, probabilities)),
        "brier": float(brier_score_loss(y_true, probabilities)),
        "positive_rate": float(np.mean(y_true)),
        "n": int(len(y_true)),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--target", default="pcos_binary")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    cohort_path = (
        args.cohort if args.cohort.is_absolute() else Path(__file__).parent.parent / args.cohort
    )
    if not cohort_path.exists():
        print(f"ERROR: cohort not found at {cohort_path.resolve()}", file=sys.stderr)
        return 1

    frame = pd.read_csv(cohort_path)
    if args.target not in frame.columns:
        print(f"ERROR: no '{args.target}' column in {cohort_path}", file=sys.stderr)
        return 1

    frame = frame[frame[args.target].notna()].reset_index(drop=True)
    y = frame[args.target].astype(float).to_numpy()

    print(f"Cohort: {len(frame)} patients, {int(y.sum())} positive ({y.mean():.1%})")

    # Held-out patients, reserved before anything is fit.
    train_frame, test_frame = train_test_split(
        frame, test_size=args.test_size, stratify=y, random_state=args.seed
    )
    train_frame = train_frame.reset_index(drop=True)
    test_frame = test_frame.reset_index(drop=True)
    y_train = train_frame[args.target].astype(float).to_numpy()
    y_test = test_frame[args.target].astype(float).to_numpy()
    print(f"Split: {len(train_frame)} train / {len(test_frame)} held-out\n")

    # Cross-validated stability on the training portion only. The per-fold
    # validation predictions are KEPT: they are the only predictions on training
    # patients that were not made by a model that had seen them, which makes them
    # the only honest data a calibrator may be fitted on.
    folds = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    fold_scores: list[float] = []
    oof = np.full(len(train_frame), np.nan, dtype=float)
    for index, (fit_idx, val_idx) in enumerate(folds.split(train_frame, y_train), start=1):
        encoder = StaticClinicalEncoder(random_state=args.seed).fit(
            train_frame.iloc[fit_idx], target_column=args.target
        )
        probabilities = encoder.predict_proba(train_frame.iloc[val_idx])
        oof[val_idx] = probabilities
        auroc = float(roc_auc_score(y_train[val_idx], probabilities))
        fold_scores.append(auroc)
        print(f"  fold {index}: AUROC {auroc:.4f}")

    print(f"\nCV AUROC: {np.mean(fold_scores):.4f} +/- {np.std(fold_scores):.4f}")

    # Fit the calibrator on out-of-fold TRAINING predictions, then freeze it.
    # The held-out patients are never involved in this fit; they only ever have
    # the finished calibrator applied to them, exactly once, below.
    calibrator: PlattCalibrator | None = PlattCalibrator()
    try:
        assert calibrator is not None
        calibrator.fit(y_train, oof, source=ALLOWED_CALIBRATION_FIT_SOURCE)
        print(
            f"Platt calibrator fitted on {calibrator.n_fit_} out-of-fold training "
            f"predictions (coef {calibrator.coef_:.3f}, intercept {calibrator.intercept_:+.3f})"
        )
    except ValueError as exc:
        calibrator = None
        print(f"No calibrator fitted: {exc}")

    # Final artifact: refit on the full training portion, evaluated on held-out.
    encoder = StaticClinicalEncoder(random_state=args.seed).fit(
        train_frame, target_column=args.target, source_dataset="kottarathil-2020"
    )
    raw_test = encoder.predict_proba(test_frame)
    held_out = evaluate(y_test, raw_test)
    print(
        f"HELD-OUT (raw): AUROC {held_out['auroc']:.4f}  AUPRC {held_out['auprc']:.4f}  "
        f"Brier {held_out['brier']:.4f}  (n={held_out['n']})"
    )

    calibration = simplified_calibration_report(y_test, raw_test, calibrator=calibrator)
    if calibration["calibrated"] is not None:
        print(
            f"HELD-OUT (calibrated): Brier {calibration['calibrated']['brier']:.4f} "
            f"(AUROC and AUPRC are unchanged by a monotone recalibration)"
        )

    assert encoder.artifact is not None
    encoder.artifact.metrics = {
        **{f"heldout_{k}": v for k, v in held_out.items()},
        "cv_auroc_mean": float(np.mean(fold_scores)),
        "cv_auroc_std": float(np.std(fold_scores)),
        **(
            {"heldout_calibrated_brier": calibration["calibrated"]["brier"]}
            if calibration["calibrated"] is not None
            else {}
        ),
    }

    output = (
        args.output if args.output.is_absolute() else Path(__file__).parent.parent / args.output
    )
    saved = encoder.save(output)
    if calibrator is not None:
        # Persisted separately from the model so that applying it is an explicit,
        # auditable step rather than something the encoder does invisibly.
        (output / "calibrator.json").write_text(json.dumps(calibrator.to_dict(), indent=2) + "\n")
    (output / "metrics.json").write_text(
        json.dumps(
            {
                "cohort": str(cohort_path),
                "n_train": len(train_frame),
                "held_out": held_out,
                "cv_auroc_mean": float(np.mean(fold_scores)),
                "cv_auroc_std": float(np.std(fold_scores)),
                "cv_fold_auroc": fold_scores,
                "calibration": calibration,
                "calibration_protocol": (
                    "Platt scaling fitted ONLY on out-of-fold predictions for the "
                    f"{len(train_frame)} training patients, then applied once, unchanged, to "
                    f"the {len(test_frame)} held-out patients. Nothing is fitted on held-out "
                    "labels. Both raw_model_score and calibrated_model_score are retained."
                ),
                "caveat": (
                    "Metrics describe reproduction of one clinic's recorded PCOS label on a "
                    "cross-sectional cohort. They are not diagnostic accuracy, and they do "
                    "not transfer to the multimodal report, whose cross-modal combination "
                    "is rule-based and carries no validated accuracy."
                ),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nSaved encoder -> {saved}")
    print(f"Saved metrics -> {output / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
