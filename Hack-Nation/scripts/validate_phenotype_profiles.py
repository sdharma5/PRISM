"""Cohort-level validation of phenotype profiles, plus calibration of the learned probability.

Two evaluations that must not be confused, and are therefore reported in
separate sections:

**Phenotype affinities have no ground truth.** The centroids are declared from
the literature and no subtype label exists, so there is nothing to be accurate
*against*. What can be measured is whether the assignment is an artifact of
choices we made: resampling noise, a dropped domain, the softmax temperature,
the indeterminacy cut-points. Agreement statistics (ARI / NMI) are computed
between the assignment under perturbation and the unperturbed assignment --
self-consistency, explicitly not accuracy.

**The learned PMOS probability does have ground truth** and is genuinely
calibratable. It gets Brier score, expected calibration error and a reliability
table on held-out patients.

Usage::

    python scripts/validate_phenotype_profiles.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.metrics import (  # noqa: E402
    adjusted_rand_score,
    brier_score_loss,
    normalized_mutual_info_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split  # noqa: E402

from evaluation.calibration import (  # noqa: E402
    ALLOWED_CALIBRATION_FIT_SOURCE,
    PlattCalibrator,
    simplified_calibration_report,
)
from features.phenotype_domains import PhenotypeDomainScorer  # noqa: E402
from models.adapters.pmos.prototype_similarity import (  # noqa: E402
    DOMAIN_PROTOTYPES,
    MIXED_MIN_ASSESSABLE_DOMAINS,
    PROFILE_DEFINING_DOMAINS,
    PrototypeSimilarityModel,
    androgenic_evidence_source,
    summarize,
)
from models.adapters.pmos.stability import PhenotypeStabilityEngine  # noqa: E402
from models.phenotype.indeterminate import INDETERMINATE  # noqa: E402
from models.tabular.encoder import StaticClinicalEncoder  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_COHORT = Path("../datasets/pmos_tabular/cohort_wide.csv")
DEFAULT_OUTPUT = Path("artifacts/experiments/exp_phenotype_validation")
DOMAINS = (
    "reproductive",
    "metabolic",
    "clinical_androgenic_evidence",
    "biochemical_androgenic_evidence",
    "ovarian",
    "lh_amh_pattern",
    "symptom_burden",
)


def resolve(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def score_domains(
    scorer: PhenotypeDomainScorer, frame: pd.DataFrame
) -> list[dict[str, float | None]]:
    """Domain composites per patient, preserving None for abstained domains."""
    scored = scorer.score_frame(frame)
    rows: list[dict[str, float | None]] = []
    for index in range(len(frame)):
        row: dict[str, float | None] = {}
        for domain, results in scored.items():
            value = getattr(results[index], "score", None)
            row[domain] = float(value) if isinstance(value, int | float) else None
        rows.append(row)
    return rows


def assign(model: PrototypeSimilarityModel, rows: list[dict[str, float | None]]) -> list[str]:
    return [str(model.predict(row).dominant) for row in rows]


def bootstrap_stability(
    rows: list[dict[str, float | None]],
    model: PrototypeSimilarityModel,
    *,
    n_bootstrap: int,
    noise: float,
    seed: int,
) -> dict[str, Any]:
    """Self-consistency of cohort assignments under resampling noise.

    ARI and NMI are computed against the UNPERTURBED assignment, not against any
    label. High agreement means the partition is reproducible; it says nothing
    about whether the partition is clinically meaningful.
    """
    reference = assign(model, rows)
    rng = np.random.default_rng(seed)

    ari_scores, nmi_scores, per_patient = [], [], np.zeros(len(rows))
    for _ in range(n_bootstrap):
        jittered = [
            {
                k: (v + float(rng.normal(0.0, noise)) if v is not None else None)
                for k, v in row.items()
            }
            for row in rows
        ]
        labels = assign(model, jittered)
        ari_scores.append(adjusted_rand_score(reference, labels))
        nmi_scores.append(normalized_mutual_info_score(reference, labels))
        per_patient += np.array(
            [a == b for a, b in zip(reference, labels, strict=True)], dtype=float
        )

    per_patient /= max(n_bootstrap, 1)
    return {
        "n_bootstrap": n_bootstrap,
        "noise_scale": noise,
        "adjusted_rand_index_mean": float(np.mean(ari_scores)),
        "adjusted_rand_index_std": float(np.std(ari_scores)),
        "normalized_mutual_info_mean": float(np.mean(nmi_scores)),
        "per_patient_agreement_mean": float(per_patient.mean()),
        "fraction_patients_below_0.6_agreement": float((per_patient < 0.6).mean()),
        "note": (
            "Agreement with the unperturbed assignment — self-consistency, NOT accuracy. "
            "No subtype ground truth exists in this repository."
        ),
    }


def centroid_stability(
    rows: list[dict[str, float | None]], *, n_bootstrap: int, seed: int
) -> dict[str, Any]:
    """How much each prototype's assigned-member centroid moves under resampling."""
    rng = np.random.default_rng(seed)
    model = PrototypeSimilarityModel()
    reference = assign(model, rows)

    drift: dict[str, list[float]] = {name: [] for name in DOMAIN_PROTOTYPES}
    for _ in range(n_bootstrap):
        index = rng.integers(0, len(rows), size=len(rows))
        resampled = [rows[i] for i in index]
        labels = assign(model, resampled)
        for name in DOMAIN_PROTOTYPES:
            members = [r for r, lab in zip(resampled, labels, strict=True) if lab == name]
            base = [r for r, lab in zip(rows, reference, strict=True) if lab == name]
            if not members or not base:
                continue
            shift = [
                abs(
                    np.mean([m[d] for m in members if m.get(d) is not None] or [0.0])
                    - np.mean([b[d] for b in base if b.get(d) is not None] or [0.0])
                )
                for d in DOMAINS
            ]
            drift[name].append(float(np.mean(shift)))

    return {
        name: {
            "mean_centroid_drift_z": float(np.mean(values)) if values else None,
            "n_resamples_with_members": len(values),
        }
        for name, values in drift.items()
    }


def missing_domain_stratification(
    rows: list[dict[str, float | None]], model: PrototypeSimilarityModel
) -> dict[str, Any]:
    """Assignment behaviour stratified by how many domains were observed.

    The deployment-relevant question: does the profile degrade gracefully as
    evidence thins, or does it keep asserting a confident answer?
    """
    strata: dict[int, list[dict[str, float | None]]] = {}
    for row in rows:
        n_observed = sum(1 for v in row.values() if v is not None)
        strata.setdefault(n_observed, []).append(row)

    out: dict[str, Any] = {}
    for n_observed in sorted(strata):
        subset = strata[n_observed]
        results = [model.predict(row) for row in subset]
        indeterminate = sum(1 for r in results if r.dominant == "indeterminate")
        out[str(n_observed)] = {
            "n_patients": len(subset),
            "indeterminate_fraction": round(indeterminate / len(subset), 4),
            "mean_entropy": round(float(np.mean([r.entropy for r in results])), 4),
            "mean_top_affinity": round(
                float(
                    np.mean(
                        [max(r.affinities.values()) if r.affinities else 0.0 for r in results]
                    )
                ),
                4,
            ),
        }
    return out


def temperature_sensitivity(
    rows: list[dict[str, float | None]],
    *,
    temperatures: tuple[float, ...] = (0.10, 0.25, 0.50, 1.00),
    baseline: float = 0.25,
) -> dict[str, Any]:
    """Cohort behaviour as the softmax temperature is swept.

    The temperature is a presentation choice, not a fitted parameter, so the
    honest question is what it actually controls. A softmax is monotone, so it
    cannot reorder the similarities: among patients who match a named prototype
    the dominant profile is invariant by construction. What the temperature does
    move is confidence -- the entropy and the top affinity -- and therefore the
    indeterminate boundary, since indeterminate mass is a function of affinity.
    A flip rate near zero here is the expected result and is evidence the
    temperature is not silently reassigning patients.
    """
    reference = assign(PrototypeSimilarityModel(temperature=baseline), rows)

    out: dict[str, Any] = {}
    for temperature in temperatures:
        model = PrototypeSimilarityModel(temperature=temperature)
        results = [model.predict(row) for row in rows]
        labels = [str(r.dominant) for r in results]
        flips = sum(1 for a, b in zip(reference, labels, strict=True) if a != b)
        out[f"{temperature:.2f}"] = {
            "dominant_flip_rate_vs_baseline": round(flips / max(len(rows), 1), 4),
            "indeterminate_fraction": round(
                sum(1 for label in labels if label == "indeterminate") / max(len(rows), 1), 4
            ),
            "mean_entropy": round(float(np.mean([r.entropy for r in results])), 4),
            "mean_top_affinity": round(
                float(
                    np.mean(
                        [max(r.affinities.values()) if r.affinities else 0.0 for r in results]
                    )
                ),
                4,
            ),
        }
    return {
        "baseline_temperature": baseline,
        "by_temperature": out,
        "note": (
            "Softmax is monotone, so temperature cannot reorder similarities. It sets "
            "sharpness, and through sharpness the indeterminate boundary. Affinities "
            "remain uncalibrated at every temperature."
        ),
    }


def threshold_sensitivity(
    rows: list[dict[str, float | None]],
    *,
    margins: tuple[float, ...] = (0.05, 0.10, 0.20),
    floors: tuple[float, ...] = (0.20, 0.30, 0.40),
    baseline_margin: float = 0.10,
    baseline_floor: float = 0.30,
) -> dict[str, Any]:
    """Cohort behaviour as the indeterminacy cut-points are swept.

    Unlike the temperature these genuinely gate the assignment: ``near_tie_margin``
    sends near-ties to indeterminate and ``min_similarity`` sends weak matches
    there. Both were chosen by judgement, so the size of the swing they produce
    is a direct measure of how arbitrary the reported profile is.
    """
    reference = assign(
        PrototypeSimilarityModel(near_tie_margin=baseline_margin, min_similarity=baseline_floor),
        rows,
    )

    grid: dict[str, Any] = {}
    for margin in margins:
        for floor in floors:
            model = PrototypeSimilarityModel(near_tie_margin=margin, min_similarity=floor)
            labels = assign(model, rows)
            flips = sum(1 for a, b in zip(reference, labels, strict=True) if a != b)
            grid[f"margin={margin:.2f},floor={floor:.2f}"] = {
                "dominant_flip_rate_vs_baseline": round(flips / max(len(rows), 1), 4),
                "indeterminate_fraction": round(
                    sum(1 for label in labels if label == "indeterminate") / max(len(rows), 1), 4
                ),
            }

    swing = [entry["indeterminate_fraction"] for entry in grid.values()]
    return {
        "baseline": {"near_tie_margin": baseline_margin, "min_similarity": baseline_floor},
        "grid": grid,
        "indeterminate_fraction_range": [min(swing), max(swing)] if swing else None,
        "note": (
            "These cut-points are judgement calls. The range above is how much of the "
            "cohort moves in or out of `indeterminate` purely by moving them."
        ),
    }


def minimum_observed_domain_rule(
    rows: list[dict[str, float | None]],
    *,
    floors: tuple[int, ...] = (2, 3, 4, 5),
    baseline_floor: int = 3,
) -> dict[str, Any]:
    """The explicit minimum-observed-domain rule, and what it actually excludes.

    A domain that abstained for coverage is ``None``, never 0.0. The rule refuses
    to place a patient at all below ``min_observed_domains`` rather than locating
    them from one or two axes. Reporting the observation rate per domain matters
    because a domain that is *never* observed silently caps every patient's
    evidence at five of six.
    """
    counts = [sum(1 for v in row.values() if v is not None) for row in rows]
    n = max(len(rows), 1)

    observation_rate = {
        domain: round(sum(1 for row in rows if row.get(domain) is not None) / n, 4)
        for domain in DOMAINS
    }
    never_observed = sorted(d for d, rate in observation_rate.items() if rate == 0.0)

    by_floor: dict[str, Any] = {}
    for floor in floors:
        model = PrototypeSimilarityModel(min_observed_domains=floor)
        labels = assign(model, rows)
        by_floor[str(floor)] = {
            "n_below_floor": int(sum(1 for c in counts if c < floor)),
            "fraction_below_floor": round(sum(1 for c in counts if c < floor) / n, 4),
            "indeterminate_fraction": round(
                sum(1 for label in labels if label == "indeterminate") / n, 4
            ),
        }

    return {
        "baseline_min_observed_domains": baseline_floor,
        "n_domains_defined": len(DOMAINS),
        "observed_domain_count_distribution": {
            str(k): int(sum(1 for c in counts if c == k)) for k in sorted(set(counts))
        },
        "per_domain_observation_rate": observation_rate,
        "never_observed_domains": never_observed,
        "by_floor": by_floor,
        "note": (
            "Below the floor the model returns indeterminate with all mass on "
            "indeterminate rather than locating a patient from too few axes. Domains "
            "listed in never_observed_domains are unavailable in this cohort entirely, "
            "so no patient can reach the full domain count."
        ),
    }


def defining_domain_support(
    rows: list[dict[str, float | None]], model: PrototypeSimilarityModel
) -> dict[str, Any]:
    """Was each assigned profile's *defining* domain actually measured?

    Similarity is computed only over observed domains, which is the right call --
    zero-filling an unmeasured domain would let "not measured" read as "average"
    and distort a contrastive centroid. On its own, though, that lets a profile
    win on its secondary weights while the domain giving it its name was never
    observed, which is a defensible similarity statement and a misleading label.

    The eligibility rule in :mod:`prototype_similarity` now removes such profiles
    before scoring, so this function is no longer a caveat-generator but an
    assertion: every fraction below must be 1.0. A value under 1.0 means the gate
    has regressed and a label is being issued without its defining evidence.
    """
    out: dict[str, Any] = {}
    for name in model.prototypes:
        defining = PROFILE_DEFINING_DOMAINS.get(name)
        assigned = [row for row in rows if str(model.predict(row).dominant) == name]
        label = list(defining) if defining else ["<any: mixed requires breadth, not one axis>"]
        if not assigned:
            out[name] = {"defining_domains": label, "n_assigned": 0}
            continue
        if defining:
            supported = sum(
                1 for row in assigned if any(row.get(d) is not None for d in defining)
            )
        else:
            supported = sum(
                1
                for row in assigned
                if sum(1 for v in row.values() if v is not None) >= MIXED_MIN_ASSESSABLE_DOMAINS
            )
        out[name] = {
            "defining_domains": label,
            "n_assigned": len(assigned),
            "fraction_with_defining_domain_observed": round(supported / len(assigned), 4),
        }

    violations = [
        name
        for name, entry in out.items()
        if entry.get("n_assigned") and entry["fraction_with_defining_domain_observed"] < 1.0
    ]
    return {
        "by_profile": out,
        "violations": violations,
        "note": (
            "Every fraction must be 1.0: a profile is only eligible for a patient when "
            "at least one of its defining domains was assessed. A non-empty `violations` "
            "list means the eligibility gate has regressed and a label is being issued "
            "on secondary weights alone."
        ),
    }


def calibration_report(
    y_true: np.ndarray, probabilities: np.ndarray, n_bins: int = 10
) -> dict[str, Any]:
    """Calibration of the LEARNED PMOS probability. This one has ground truth."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    index = np.clip(np.digitize(probabilities, bins) - 1, 0, n_bins - 1)

    table, ece = [], 0.0
    for b in range(n_bins):
        rows = index == b
        if not rows.any():
            continue
        confidence = float(probabilities[rows].mean())
        accuracy = float(y_true[rows].mean())
        weight = float(rows.mean())
        ece += weight * abs(accuracy - confidence)
        table.append(
            {
                "bin": f"[{bins[b]:.1f},{bins[b + 1]:.1f})",
                "n": int(rows.sum()),
                "mean_predicted": round(confidence, 4),
                "observed_rate": round(accuracy, 4),
                "gap": round(accuracy - confidence, 4),
            }
        )

    return {
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "expected_calibration_error": round(float(ece), 4),
        "auroc": float(roc_auc_score(y_true, probabilities)),
        "reliability_table": table,
        "note": (
            "The learned PMOS probability IS a probability and is calibrated here "
            "against held-out outcomes. Phenotype affinities are NOT and are not "
            "calibrated anywhere."
        ),
    }


def out_of_fold_predictions(
    train_frame: pd.DataFrame, y_train: np.ndarray, *, seed: int, folds: int = 5
) -> np.ndarray:
    """One prediction per training patient, from a model that never saw them.

    The only predictions on training patients a calibrator may be fitted on. An
    in-sample prediction would teach the calibrator to correct an over-fit the
    deployed model does not have on new patients.
    """
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    oof = np.full(len(train_frame), np.nan, dtype=float)
    for fit_idx, val_idx in splitter.split(train_frame, y_train):
        encoder = StaticClinicalEncoder(random_state=seed).fit(train_frame.iloc[fit_idx])
        oof[val_idx] = encoder.predict_proba(train_frame.iloc[val_idx])
    return oof


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-bootstrap", type=int, default=100)
    parser.add_argument("--noise", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--n-stability-patients",
        type=int,
        default=0,
        help="Patients to run per-patient stability on; 0 runs the whole cohort.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cohort_path = resolve(args.cohort)
    if not cohort_path.exists():
        print(f"ERROR: no cohort at {cohort_path}", file=sys.stderr)
        return 1

    frame = pd.read_csv(cohort_path)
    frame = frame[frame["pmos_binary"].notna()].reset_index(drop=True)
    y = frame["pmos_binary"].astype(float).to_numpy()
    print(f"cohort: {len(frame)} patients ({y.mean():.1%} positive)\n")

    train_frame, test_frame = train_test_split(
        frame, test_size=0.2, stratify=y, random_state=args.seed
    )
    train_frame = train_frame.reset_index(drop=True)
    test_frame = test_frame.reset_index(drop=True)

    scorer = PhenotypeDomainScorer().fit(train_frame)
    rows = score_domains(scorer, frame)
    model = PrototypeSimilarityModel()

    print("=== phenotype profile validation (self-consistency, NOT accuracy) ===")
    bootstrap = bootstrap_stability(
        rows, model, n_bootstrap=args.n_bootstrap, noise=args.noise, seed=args.seed
    )
    print(
        f"  ARI {bootstrap['adjusted_rand_index_mean']:.4f} "
        f"(+/-{bootstrap['adjusted_rand_index_std']:.4f})   "
        f"NMI {bootstrap['normalized_mutual_info_mean']:.4f}"
    )
    print(f"  per-patient agreement {bootstrap['per_patient_agreement_mean']:.4f}")
    print(
        f"  patients below 0.6 agreement: "
        f"{bootstrap['fraction_patients_below_0.6_agreement']:.1%}"
    )

    print("\n=== assignment by number of observed domains ===")
    strata = missing_domain_stratification(rows, model)
    for n_observed, entry in strata.items():
        print(
            f"  {n_observed} domains  n={entry['n_patients']:4d}  "
            f"indeterminate {entry['indeterminate_fraction']:.1%}  "
            f"entropy {entry['mean_entropy']:.3f}"
        )

    print("\n=== minimum-observed-domain rule ===")
    min_domains = minimum_observed_domain_rule(rows)
    if min_domains["never_observed_domains"]:
        never = ", ".join(min_domains["never_observed_domains"])
        print(f"  never observed in this cohort: {never}")
    for floor, entry in min_domains["by_floor"].items():
        print(
            f"  floor {floor}  below {entry['n_below_floor']:4d} "
            f"({entry['fraction_below_floor']:.1%})  "
            f"indeterminate {entry['indeterminate_fraction']:.1%}"
        )

    print("\n=== defining-domain support for assigned profiles ===")
    defining = defining_domain_support(rows, model)
    for name, entry in defining["by_profile"].items():
        if not entry["n_assigned"]:
            print(f"  {name:22s} n=0")
            continue
        print(
            f"  {name:22s} n={entry['n_assigned']:4d}  "
            f"defining={'|'.join(entry['defining_domains']):48s} "
            f"observed for {entry['fraction_with_defining_domain_observed']:.1%}"
        )
    if defining["violations"]:
        print(f"  !! ELIGIBILITY GATE VIOLATED for: {', '.join(defining['violations'])}")
    else:
        print("  every assigned profile had its defining domain assessed")

    print("\n=== temperature sensitivity ===")
    temperature = temperature_sensitivity(rows)
    for value, entry in temperature["by_temperature"].items():
        print(
            f"  T={value}  flip {entry['dominant_flip_rate_vs_baseline']:.1%}  "
            f"indeterminate {entry['indeterminate_fraction']:.1%}  "
            f"entropy {entry['mean_entropy']:.3f}  "
            f"top affinity {entry['mean_top_affinity']:.3f}"
        )

    print("\n=== threshold sensitivity ===")
    thresholds = threshold_sensitivity(rows)
    for key, entry in thresholds["grid"].items():
        print(
            f"  {key:28s} flip {entry['dominant_flip_rate_vs_baseline']:.1%}  "
            f"indeterminate {entry['indeterminate_fraction']:.1%}"
        )
    low, high = thresholds["indeterminate_fraction_range"]
    print(f"  indeterminate fraction spans {low:.1%} to {high:.1%} across the grid")

    print("\n=== prototype centroid stability ===")
    centroids = centroid_stability(rows, n_bootstrap=min(args.n_bootstrap, 50), seed=args.seed)
    for name, entry in centroids.items():
        drift = entry["mean_centroid_drift_z"]
        print(f"  {name:22s} drift {'n/a' if drift is None else format(drift, '.4f')} z")

    print("\n=== learned PMOS probability calibration (held-out) ===")
    encoder = StaticClinicalEncoder(random_state=args.seed).fit(train_frame)
    probabilities = encoder.predict_proba(test_frame)
    y_test = test_frame["pmos_binary"].astype(float).to_numpy()

    # The calibrator is fitted on out-of-fold predictions for the TRAINING
    # patients only, then applied once, frozen, to the held-out set. Fitting it
    # on `probabilities` would make the held-out Brier score a training number.
    y_train = train_frame["pmos_binary"].astype(float).to_numpy()
    oof = out_of_fold_predictions(train_frame, y_train, seed=args.seed)
    calibrator: PlattCalibrator | None = PlattCalibrator()
    try:
        assert calibrator is not None
        calibrator.fit(y_train, oof, source=ALLOWED_CALIBRATION_FIT_SOURCE)
    except ValueError as exc:
        calibrator = None
        print(f"  no calibrator fitted: {exc}")

    calibration = simplified_calibration_report(y_test, probabilities, calibrator=calibrator)
    raw = calibration["raw"]
    print(
        f"  raw:        AUROC {raw['auroc']:.4f}  AUPRC {raw['auprc']:.4f}  "
        f"Brier {raw['brier']:.4f}  (n={int(raw['n'])})"
    )
    if calibration["calibrated"] is not None:
        print(
            f"  calibrated: Brier {calibration['calibrated']['brier']:.4f}  "
            f"(Platt on {calibrator.n_fit_} out-of-fold TRAINING predictions)"
        )
    for entry in (calibration["calibrated"] or raw)["reliability_bins"]:
        flag = "" if entry["interpretable"] else "   [too few patients to read]"
        print(
            f"    bin {entry['bin']} n={entry['n']:3d}  predicted {entry['mean_predicted']:.3f}  "
            f"observed {entry['observed_rate']:.3f} "
            f"[{entry['observed_ci_lower']:.2f},{entry['observed_ci_upper']:.2f}]{flag}"
        )

    # -- androgenic evidence availability across the whole cohort ------------
    evidence_counts: dict[str, int] = {
        "symptoms_only": 0,
        "biochemical_only": 0,
        "both": 0,
        "unavailable": 0,
    }
    for patient in rows:
        evidence_counts[androgenic_evidence_source(patient)] += 1
    n_clinical = sum(1 for r in rows if r.get("clinical_androgenic_evidence") is not None)
    n_biochemical = sum(1 for r in rows if r.get("biochemical_androgenic_evidence") is not None)
    androgenic_evidence = {
        "n_with_clinical_androgenic_evidence": n_clinical,
        "n_with_biochemical_androgenic_evidence": n_biochemical,
        "by_source": evidence_counts,
    }
    print("\n=== androgenic evidence availability ===")
    print(f"  clinical (symptoms)    assessable for {n_clinical}/{len(rows)} patients")
    print(f"  biochemical (assay)    assessable for {n_biochemical}/{len(rows)} patients")
    for source, count in evidence_counts.items():
        print(f"    {source:16s} {count:4d}")

    # Per-patient stability, exercising the engine end-to-end. Sampled at random
    # rather than taking the first N, which would inherit any ordering in the
    # cohort file, and stratified by observed-domain count so that thin-evidence
    # patients cannot be averaged away by well-measured ones.
    engine = PhenotypeStabilityEngine(n_bootstrap=50, seed=args.seed)
    if args.n_stability_patients and args.n_stability_patients < len(rows):
        rng = np.random.default_rng(args.seed)
        index = sorted(
            rng.choice(len(rows), size=args.n_stability_patients, replace=False).tolist()
        )
    else:
        index = list(range(len(rows)))
    sample_rows = [rows[i] for i in index]
    sample = [
        engine.evaluate(row, model, available_modalities=["static_clinical"])
        for row in sample_rows
    ]
    unstable = sum(1 for r in sample if not r.is_stable)
    print(f"\nper-patient stability on {len(sample)} patients: {unstable}/{len(sample)} unstable")

    by_observed: dict[str, Any] = {}
    for row, report in zip(sample_rows, sample, strict=True):
        key = str(sum(1 for v in row.values() if v is not None))
        by_observed.setdefault(key, []).append(report)
    stability_strata = {
        key: {
            "n_patients": len(reports),
            "fraction_unstable": round(
                sum(1 for r in reports if not r.is_stable) / len(reports), 4
            ),
            "mean_stability_score": round(
                float(np.mean([r.stability_score for r in reports])), 4
            ),
            "mean_bootstrap_agreement": round(
                float(np.mean([r.bootstrap_agreement for r in reports])), 4
            ),
        }
        for key, reports in sorted(by_observed.items())
    }
    for key, entry in stability_strata.items():
        print(
            f"  {key} domains  n={entry['n_patients']:4d}  "
            f"unstable {entry['fraction_unstable']:.1%}  "
            f"stability {entry['mean_stability_score']:.3f}"
        )

    # -- final assignments, after BOTH gates ---------------------------------
    # This is the number that gets quoted, so it is computed the same way the
    # adapter computes it: eligibility first, then the stability verdict.
    final_counts: dict[str, int] = {}
    indeterminate_reasons: dict[str, int] = {}
    for patient, report in zip(sample_rows, sample, strict=True):
        stable = bool(report.is_stable and not report.abstain_from_profile)
        summary = summarize(model.predict(patient), is_stable=stable)
        label = summary["dominant_profile"] or INDETERMINATE
        final_counts[label] = final_counts.get(label, 0) + 1
        if summary["dominant_profile"] is None and summary["indeterminate_reasons"]:
            key = summary["indeterminate_reasons"][0].split(":")[0].split("(")[0].strip()
            indeterminate_reasons[key] = indeterminate_reasons.get(key, 0) + 1

    n_scored = max(len(sample_rows), 1)
    n_indeterminate = final_counts.get(INDETERMINATE, 0)
    assignments = {
        "n_patients": len(sample_rows),
        "counts": dict(sorted(final_counts.items(), key=lambda kv: -kv[1])),
        "n_indeterminate": n_indeterminate,
        "fraction_indeterminate": round(n_indeterminate / n_scored, 4),
        "indeterminate_reason_counts": dict(
            sorted(indeterminate_reasons.items(), key=lambda kv: -kv[1])
        ),
        "n_androgenic_leaning_without_androgenic_evidence": sum(
            1
            for patient, report in zip(sample_rows, sample, strict=True)
            if summarize(
                model.predict(patient),
                is_stable=bool(report.is_stable and not report.abstain_from_profile),
            )["dominant_profile"]
            == "androgenic_leaning"
            and androgenic_evidence_source(patient) == "unavailable"
        ),
    }
    print("\n=== final phenotype assignments (eligibility + stability gated) ===")
    for label, count in assignments["counts"].items():
        print(f"  {label:22s} {count:4d}  ({count / n_scored:.1%})")
    print(
        f"  indeterminate: {n_indeterminate}/{n_scored} "
        f"({assignments['fraction_indeterminate']:.1%})"
    )
    print(
        "  androgenic_leaning with unavailable androgenic evidence: "
        f"{assignments['n_androgenic_leaning_without_androgenic_evidence']}"
    )

    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "phenotype_validation.json").write_text(
        json.dumps(
            {
                "n_patients": len(frame),
                "bootstrap_stability": bootstrap,
                "missing_domain_stratification": strata,
                "minimum_observed_domain_rule": min_domains,
                "defining_domain_support": defining,
                "temperature_sensitivity": temperature,
                "threshold_sensitivity": thresholds,
                "centroid_stability": centroids,
                "learned_probability_calibration": calibration,
                "androgenic_evidence": androgenic_evidence,
                "final_assignments": assignments,
                "per_patient_stability": {
                    "n": len(sample),
                    "n_unstable": unstable,
                    "sampling": "full_cohort" if len(sample) == len(rows) else "random_seeded",
                    "mean_stability_score": float(np.mean([r.stability_score for r in sample])),
                    "mean_bootstrap_agreement": float(
                        np.mean([r.bootstrap_agreement for r in sample])
                    ),
                    "by_observed_domain_count": stability_strata,
                },
                "caveats": [
                    "Phenotype affinity scores are NOT calibrated probabilities and have no "
                    "ground truth. ARI/NMI here measure self-consistency under perturbation.",
                    "Prototype centroids are declared from the literature, not fitted, so "
                    "centroid drift measures membership churn rather than centroid learning.",
                    "Only the learned PMOS probability is calibrated against outcomes.",
                    "Single cohort, one clinic, cross-sectional. No external validation.",
                    "This cohort contains NO androgen assay, so "
                    "`biochemical_androgenic_evidence` is unavailable for every patient and "
                    "`androgenic_leaning` here always rests on clinical signs alone (acne, "
                    "androgenic alopecia, facial hair growth). Biochemical hyperandrogenism "
                    "must never be inferred from that label. See androgenic_evidence.",
                    "Skin darkening is scored in the metabolic domain: acanthosis nigricans "
                    "is a sign of insulin resistance, not of androgen excess.",
                    "A profile is only offered to a patient whose defining domain was "
                    "assessed; ineligible profiles are removed and the remaining "
                    "similarities renormalized, never zero-filled. See "
                    "defining_domain_support.violations, which must be empty.",
                    "A dominant profile is published only when the stability engine calls "
                    "the assignment stable. Unstable patients are returned as indeterminate "
                    "rather than given a provisional label.",
                ],
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nwrote {output_dir / 'phenotype_validation.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
