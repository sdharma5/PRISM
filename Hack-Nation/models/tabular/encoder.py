"""The static clinical encoder: train, persist, and encode a new patient.

This is the only branch in PRISM whose predictive head is fit against a PMOS
label, and it is legitimate precisely because the tabular cohort's symptoms,
history, labs and derived measurements all belong to the *same* person. It is
therefore the only component entitled to issue a whole-patient PMOS probability.

Why this module exists at all: ``scripts/train_static_baselines.py`` already
fits and evaluates models honestly, but with ``save_checkpoints`` defaulting to
False it discards every fitted estimator at the end of the run. Cross-validated
metrics tell you how well a model *would* generalise; they do not leave behind
anything that can score a new patient. This class closes that gap by persisting
the estimator together with the two things it is useless without:

* the **preprocessing statistics** (imputer + scaler), which must come from the
  training rows -- transforming a new patient with their own statistics is the
  classic silent leakage bug, and with one patient it is not even defined;
* the **domain reference statistics**, so the exported composites are z-scored
  against the training cohort rather than against the patient themselves.

The saved artifact is self-describing: it carries its feature order, cohort
version and metrics, so a checkpoint can never be silently applied to a frame
whose columns have moved.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features.phenotype_domains import PhenotypeDomainScorer
from features.static_features import derive_features
from registry.loader import load_variable_registry
from schemas.modality_token import ModalityToken

__all__ = ["LEGACY_FEATURE_ALIASES", "StaticClinicalEncoder", "StaticEncoderArtifact"]

_ARTIFACT_NAME = "static_clinical_encoder.joblib"
_METADATA_NAME = "static_clinical_encoder.json"

#: Columns that are identifiers or targets, never model inputs.
_NON_FEATURE = {"patient_id", "pmos_binary"}

#: ``source_dataset -> {artifact feature slot: canonical code that fills it}``.
#:
#: A frozen artifact keeps the feature names it was trained under. When a name
#: turns out to describe its column wrongly, the artifact can't be renamed
#: without retraining -- but feeding it the variable it's *named* after is worse
#: than feeding it nothing.
#:
#: kottarathil-2020's ``Cycle length(days)`` was mapped to ``cycle_length``, which
#: the registry defines as menses-to-menses (~28). The fitted scaler says
#: otherwise -- mean 4.94, scale 1.42 -- so it's bleeding duration. Routing a real
#: cycle length in puts the patient ~33 SD out of distribution and inverts the
#: score (5 -> 0.985, 52 -> 0.003). The slot keeps its name and is fed from
#: ``menses_duration``; true ``cycle_length`` stays absent and is imputed.
LEGACY_FEATURE_ALIASES: dict[str, dict[str, str]] = {
    "kottarathil-2020": {"cycle_length": "menses_duration"},
}


@lru_cache(maxsize=1)
def _registry_variable_codes() -> frozenset[str]:
    """Registry codes, for gating token passthrough. Cached -- consulted once per
    supplied variable per request."""
    return frozenset(load_variable_registry().variables)


@dataclass
class StaticEncoderArtifact:
    """Everything needed to score a new patient, kept together on purpose."""

    pipeline: Pipeline
    feature_names: list[str]
    domain_scorer: PhenotypeDomainScorer
    model_version: str = "static-clinical-0.1.0"
    source_dataset: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    n_train: int = 0


class StaticClinicalEncoder:
    """Fit, persist and apply the static clinical branch."""

    modality = "static_clinical"

    def __init__(
        self,
        *,
        model_version: str = "static-clinical-0.1.0",
        random_state: int = 0,
    ) -> None:
        self.model_version = model_version
        self.random_state = random_state
        self.artifact: StaticEncoderArtifact | None = None

    # -- training ----------------------------------------------------------

    @staticmethod
    def feature_columns(frame: pd.DataFrame) -> list[str]:
        """Numeric model inputs, in a stable order.

        Sorted rather than left in frame order: column order must not depend on
        how the cohort CSV happened to be written, or a checkpoint silently
        mismatches a re-exported cohort.
        """
        numeric = frame.select_dtypes(include=[np.number]).columns
        return sorted(column for column in numeric if column not in _NON_FEATURE)

    def fit(
        self,
        frame: pd.DataFrame,
        *,
        target_column: str = "pmos_binary",
        source_dataset: str | None = None,
    ) -> StaticClinicalEncoder:
        """Fit the head and the preprocessing on training rows only.

        Args:
            frame: Training cohort. Must already exclude any held-out rows.
            target_column: Binary PMOS label column.
            source_dataset: Dataset identifier stamped on every token.

        Returns:
            self, fitted.

        Raises:
            ValueError: If the target is missing or single-class.
        """
        if target_column not in frame.columns:
            raise ValueError(f"Cohort has no '{target_column}' column.")

        features = self.feature_columns(frame)
        if not features:
            raise ValueError("Cohort has no numeric feature columns.")

        y = frame[target_column].astype(float).to_numpy()
        if len(np.unique(y[np.isfinite(y)])) < 2:
            raise ValueError(
                f"'{target_column}' has a single class; a classifier cannot be fit on it."
            )
        X = frame[features]

        pipeline = Pipeline(
            [
                # Median rather than mean: several cohort variables are skewed
                # (AMH, LH/FSH), where a mean imputation shifts the centre.
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        # The cohort is 364 negative to 177 positive; without
                        # balancing, the majority class dominates the fit.
                        class_weight="balanced",
                        random_state=self.random_state,
                    ),
                ),
            ]
        )
        pipeline.fit(X, y)

        scorer = PhenotypeDomainScorer().fit(frame)

        self.artifact = StaticEncoderArtifact(
            pipeline=pipeline,
            feature_names=features,
            domain_scorer=scorer,
            model_version=self.model_version,
            source_dataset=source_dataset,
            n_train=int(len(frame)),
        )
        return self

    # -- prediction --------------------------------------------------------

    def _require(self) -> StaticEncoderArtifact:
        if self.artifact is None:
            raise RuntimeError("StaticClinicalEncoder is not fitted. Call fit() or load() first.")
        return self.artifact

    def _row_from_values(self, values: dict[str, Any]) -> pd.DataFrame:
        """Build a one-row frame in the artifact's exact feature order.

        Unknown keys are ignored and absent features become NaN, which the fitted
        imputer fills with the *training* median. That is the only defensible
        choice for a single patient: there is no cohort to impute from at
        inference time.

        Feature slots are resolved through :data:`LEGACY_FEATURE_ALIASES`, so a
        slot whose training column was later found to mean something else reads
        from the canonical code it actually holds rather than the one it is
        named after. See that constant for why this is not cosmetic.
        """
        artifact = self._require()
        aliases = LEGACY_FEATURE_ALIASES.get(artifact.source_dataset or "", {})
        row: dict[str, float] = {}
        for name in artifact.feature_names:
            source_code = aliases.get(name, name)
            value = values.get(source_code)
            row[name] = float(value) if isinstance(value, int | float | bool) else np.nan
        return pd.DataFrame([row], columns=artifact.feature_names)

    def predict_proba_from_features(self, values: dict[str, Any]) -> float:
        """P(PMOS) for one patient's canonical clinical variables."""
        artifact = self._require()
        frame = self._row_from_values(values)
        return float(artifact.pipeline.predict_proba(frame)[0, 1])

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        """P(PMOS) for a frame of patients.

        Resolves :data:`LEGACY_FEATURE_ALIASES` exactly as the single-patient
        path does. Without this, a cohort re-ingested under corrected canonical
        codes would reindex an aliased slot to all-NaN and score every row off
        the imputed median -- silently, since a missing column is indistinguish-
        able from an absent measurement after reindexing.
        """
        artifact = self._require()
        aliases = LEGACY_FEATURE_ALIASES.get(artifact.source_dataset or "", {})
        # Drop before renaming: a frame may carry both the true variable under
        # the slot's name and the one the slot actually holds, and the former is
        # exactly what must not reach it.
        source = frame.drop(columns=[s for s in aliases if s in frame.columns]).rename(
            columns={
                canonical: slot for slot, canonical in aliases.items() if canonical in frame.columns
            }
        )
        aligned = source.reindex(columns=artifact.feature_names)
        return artifact.pipeline.predict_proba(aligned)[:, 1]

    # -- token export ------------------------------------------------------

    def export_token(self, payload: Any, *, patient_id: str) -> ModalityToken:
        """Encode one patient into a :class:`ModalityToken`.

        Args:
            payload: A one-row DataFrame, a mapping of canonical variables, or a
                list of clinical events exposing ``canonical_variable_code`` and
                ``value``.
            patient_id: Patient identifier.

        Returns:
            A static clinical token carrying the domain composites the
            coordinator consumes plus the raw variables the adapter needs.
        """
        artifact = self._require()
        values = _derive(_coerce_to_values(payload))
        frame = self._row_from_values(values)

        probability = float(artifact.pipeline.predict_proba(frame)[0, 1])

        # Domain composites are scored against the TRAINING cohort statistics.
        scoring_frame = pd.DataFrame([values])
        structured: dict[str, Any] = {}
        missing: list[str] = []
        # Feature-slot misses only. `missing` also collects domain-composite
        # misses, and the fraction below is per-feature, so the two must not be
        # pooled -- a pooled count divided by the feature count can exceed 1.
        missing_features: list[str] = []
        warnings: list[str] = []
        try:
            scored = artifact.domain_scorer.score_frame(scoring_frame)
            for domain, results in scored.items():
                score = getattr(results[0], "score", None)
                # None when the domain abstained for insufficient coverage. Do NOT
                # coerce to 0.0: a z-score of 0.0 means "exactly average", which is
                # a measurement, while None means "not measured". Collapsing them
                # would tell the phenotype model this patient is average on a
                # domain that was never assessed.
                structured[f"{domain}_score"] = (
                    float(score) if isinstance(score, int | float) else None
                )
                # Coverage travels with the score -- otherwise a domain built
                # from most of its variables and one built from a single
                # observation are the same bare z-score. The `_coverage` suffix
                # sits outside the `{domain}_score` namespace the adapter reads.
                coverage = getattr(results[0], "coverage", None)
                if isinstance(coverage, int | float):
                    structured[f"{domain}_coverage"] = float(coverage)
                if score is None:
                    missing.append(f"{domain}_score")
        except Exception as exc:  # noqa: BLE001 - a token without composites is still useful
            warnings.append(f"Domain composites unavailable: {exc}")

        # Carry the raw canonical variables through so the adapter can apply
        # guideline thresholds to actual measurements rather than to composites.
        # Alias resolution mirrors _row_from_values: a slot whose training column
        # was later found to mean something else reads from the canonical code it
        # actually holds and is stored under that code, not the slot name.
        aliases = LEGACY_FEATURE_ALIASES.get(artifact.source_dataset or "", {})
        for name in artifact.feature_names:
            source_code = aliases.get(name, name)
            val = values.get(source_code)
            if isinstance(val, int | float | bool):
                structured[source_code] = val
            else:
                missing.append(name)
                missing_features.append(name)
        # Also carry aliased slot names when the caller provides them as the true
        # canonical variable. For kottarathil-2020 the "cycle_length" slot held
        # menses_duration; a patient form may additionally supply the real
        # menses-to-menses interval under "cycle_length". Pass it through so the
        # ovulatory-dysfunction axis rules can threshold it; setdefault prevents
        # overwriting a value already stored above.
        for slot in aliases:
            if slot not in structured:
                val = values.get(slot)
                if isinstance(val, int | float | bool):
                    structured[slot] = val

        # Carry through every other canonical variable the caller supplied.
        #
        # The loop above walks only the fitted columns, but the guideline axes
        # threshold variables the model never learned -- hirsutism, acne,
        # cycle_irregularity, total_testosterone -- so those need to reach the
        # adapter too.
        #
        # Registry-gated so a misspelled code is dropped rather than travelling
        # on as evidence, and `_NON_FEATURE` excluded so the training label can't
        # ride along in a token.
        for code, val in values.items():
            if code in structured or code in _NON_FEATURE:
                continue
            if isinstance(val, int | float | bool) and code in _registry_variable_codes():
                structured[code] = val

        structured["pmos_evidence_probability"] = probability

        n_features = max(len(artifact.feature_names), 1)
        observed_fraction = 1.0 - (len(missing_features) / n_features)
        # On the token so consumers don't re-derive it from key names, which
        # can't distinguish a feature slot from a passed-through variable.
        structured["observed_feature_fraction"] = round(observed_fraction, 4)
        if observed_fraction < 0.5:
            warnings.append(
                f"Only {observed_fraction:.0%} of the model's features were observed; the "
                "remainder were imputed with training medians and the probability is "
                "correspondingly less informative."
            )

        return ModalityToken(
            patient_id=patient_id,
            modality="static_clinical",
            embedding=[],
            structured_features=structured,
            quality_score=round(observed_fraction, 4),
            # Confidence tracks distance from the decision boundary: a 0.51 is a
            # coin flip regardless of how complete the input was.
            confidence_score=round(float(abs(probability - 0.5) * 2.0), 4),
            model_version=artifact.model_version,
            source_dataset=artifact.source_dataset,
            missing_fields=missing,
            warnings=warnings,
        )

    # -- persistence -------------------------------------------------------

    def save(self, directory: str | Path) -> Path:
        """Persist the estimator, preprocessing and domain statistics together."""
        import joblib  # noqa: PLC0415 - only needed when actually saving

        artifact = self._require()
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        joblib.dump(artifact, path / _ARTIFACT_NAME)
        (path / _METADATA_NAME).write_text(
            json.dumps(
                {
                    "model_version": artifact.model_version,
                    "source_dataset": artifact.source_dataset,
                    "n_train": artifact.n_train,
                    "n_features": len(artifact.feature_names),
                    "feature_names": artifact.feature_names,
                    "metrics": artifact.metrics,
                },
                indent=2,
            )
            + "\n"
        )
        return path / _ARTIFACT_NAME

    @classmethod
    def load(cls, directory: str | Path) -> StaticClinicalEncoder:
        """Restore an encoder saved by :meth:`save`."""
        import joblib  # noqa: PLC0415

        path = Path(directory) / _ARTIFACT_NAME
        if not path.exists():
            raise FileNotFoundError(
                f"No static clinical encoder at {path}. Train one with "
                "scripts/train_static_encoder.py first."
            )
        artifact: StaticEncoderArtifact = joblib.load(path)
        encoder = cls(model_version=artifact.model_version)
        encoder.artifact = artifact
        return encoder


def _derive(values: dict[str, Any]) -> dict[str, Any]:
    """Fill derivable features (bmi, ratios) from what the patient supplied.

    Uses the same ``derive_features`` as training. Computing BMI differently at
    inference -- or not at all, letting the imputer substitute a median -- is
    train/serve skew, and a patient who gave weight and height has supplied it.

    ``derive_features`` never overwrites a measured value and won't build a
    derived value from an imputed input.
    """
    numeric = {
        code: value
        for code, value in values.items()
        if isinstance(value, int | float) and not isinstance(value, bool)
    }
    if not numeric:
        return values
    try:
        derived, _ = derive_features(pd.DataFrame([numeric]))
    except Exception:  # noqa: BLE001 - derivation is a bonus, never a failure
        return values

    row = derived.iloc[0].to_dict()
    out = dict(values)
    for code, value in row.items():
        if (code not in out or out.get(code) is None) and (
            isinstance(value, int | float) and pd.notna(value)
        ):
            out[code] = float(value)
    return out


def _coerce_to_values(payload: Any) -> dict[str, Any]:
    """Accept a frame, a mapping, or a list of events and return a value map."""
    if isinstance(payload, pd.DataFrame):
        if len(payload) != 1:
            raise ValueError(f"Expected exactly one patient row, got {len(payload)}.")
        return payload.iloc[0].to_dict()
    if isinstance(payload, dict):
        return dict(payload)
    if isinstance(payload, list):
        values: dict[str, Any] = {}
        for event in payload:
            code = getattr(event, "canonical_variable_code", None)
            if code is not None:
                values[code] = getattr(event, "value", None)
        return values
    raise TypeError(
        f"Unsupported payload type {type(payload).__name__}; expected DataFrame, dict, "
        "or list of clinical events."
    )
