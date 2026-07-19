"""Target-specific temporal encoder: the current benchmark winners, persisted.

The benchmark (``artifacts/experiments/exp_temporal_baselines``) produced a
result worth taking seriously rather than designing around: **no single model
wins every target.** On held-out participants,

* LH   -- LOCF (MAE 3.265) beats ridge-on-window (3.668)
* PdG  -- LOCF (3.255) beats ridge (3.306)
* E3G  -- ridge (50.64) narrowly beats LOCF (52.09)
* phase-- logistic on the window (macro F1 0.565) beats majority (0.115)

So this encoder selects a model **per target** instead of forcing one
architecture to serve all four. That is not inelegance; it is the honest reading
of the evidence, and it means a future GRU has to beat a specific, named
incumbent on each target rather than an artificially weak single baseline.

The negative result is deliberately preserved in the exported token: every
hormone field carries the ``method`` that produced it, so a consumer can see
that LH came from persistence and E3G from a fitted linear model.

**Uncertainty comes from empirical residuals**, not a learned head. With 26
training participants a neural uncertainty estimate would be fitting noise;
quantiles of the training residual distribution are transparent, cheap, and
about as trustworthy as anything else available at this sample size.

**Causal only.** Every prediction for day *t* uses days < *t*. This is a
*forecasting* encoder, labelled ``task_type='causal_forecasting'`` on the token,
and it is not comparable to a bidirectional reconstruction model that may see
future context.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from models.temporal.baselines import (
    LocfBaseline,
    LogisticPhaseBaseline,
    RidgeWindowBaseline,
    build_sequences,
)
from schemas.modality_token import ModalityToken

__all__ = ["TemporalStateEncoder", "TemporalStateArtifact"]

_ARTIFACT_NAME = "temporal_state_encoder.joblib"

#: Model chosen per target by the frozen benchmark. Changing an entry requires
#: re-running scripts/benchmark_temporal.py and showing the new winner on the
#: same split -- never a preference for the more sophisticated option.
DEFAULT_TARGET_MODELS: dict[str, str] = {
    "lh": "locf",
    "e3g": "ridge_window",
    "pdg": "locf",
}

HORMONES: tuple[str, ...] = ("lh", "e3g", "pdg")
CHANNELS: tuple[str, ...] = (
    "lh",
    "e3g",
    "pdg",
    "resting_heart_rate",
    "wrist_temperature",
    "hrv_rmssd",
    "mean_glucose",
)

#: Minimum usable days before the encoder will speak at all. Below this the
#: window is mostly padding and a "current state" claim is not supportable.
MIN_DAYS_FOR_STATE = 14


@dataclass
class TemporalStateArtifact:
    """Everything needed to reproduce inference in a fresh process."""

    locf: LocfBaseline
    ridge: RidgeWindowBaseline
    phase: LogisticPhaseBaseline
    target_models: dict[str, str]
    channels: tuple[str, ...]
    lookback: int
    #: Residual quantiles per hormone: {"lh": {"lo": float, "hi": float}}.
    residual_intervals: dict[str, dict[str, float]] = field(default_factory=dict)
    normalization_stats: dict[str, dict[str, float]] = field(default_factory=dict)
    model_version: str = "temporal-state-v1"
    benchmark_version: str = "temporal_baseline_v1"
    split_version: str = "mcphases_participant_split_v1"
    source_dataset: str = "mcPHASES"
    metrics: dict[str, Any] = field(default_factory=dict)


class TemporalStateEncoder:
    """Multi-head temporal encoder with independently selected per-target models."""

    modality = "longitudinal_hormonal_state"

    def __init__(
        self,
        *,
        lookback: int = 14,
        target_models: dict[str, str] | None = None,
        interval_coverage: float = 0.8,
        model_version: str = "temporal-state-v1",
    ) -> None:
        """
        Args:
            lookback: Days of causal history a windowed model may use.
            target_models: Per-hormone model choice; defaults to the benchmark winners.
            interval_coverage: Central coverage of the reported uncertainty interval.
            model_version: Stamped on every exported token.
        """
        self.lookback = lookback
        self.target_models = dict(target_models or DEFAULT_TARGET_MODELS)
        self.interval_coverage = interval_coverage
        self.model_version = model_version
        self.artifact: TemporalStateArtifact | None = None

    # -- lifecycle ---------------------------------------------------------

    def fit(
        self,
        rows: list[dict[str, Any]],
        train_ids: list[str],
        *,
        normalization_stats: dict[str, dict[str, float]] | None = None,
    ) -> TemporalStateEncoder:
        """Fit every head on training participants only.

        Args:
            rows: Participant-day records.
            train_ids: Training participant identifiers.
            normalization_stats: Train-only stats, stored for provenance.
        """
        train = build_sequences(rows, train_ids, channels=CHANNELS, lookback=self.lookback)

        locf = LocfBaseline().fit(train)
        ridge = RidgeWindowBaseline(alpha=10.0).fit(train)
        phase = LogisticPhaseBaseline().fit(train)

        # Residual intervals from TRAINING predictions. Using held-out residuals
        # would tune the interval on the evaluation set; using training residuals
        # is mildly optimistic but honest and reported as such.
        intervals: dict[str, dict[str, float]] = {}
        predictions = {"locf": locf.predict(train), "ridge_window": ridge.predict(train)}
        tail = (1.0 - self.interval_coverage) / 2.0

        for hormone in HORMONES:
            index = CHANNELS.index(hormone)
            method = self.target_models.get(hormone, "locf")
            observed = train["observed"][:, index]
            if observed.sum() < 10:
                continue
            residual = train["values"][observed, index] - predictions[method][observed, index]
            residual = residual[np.isfinite(residual)]
            if residual.size < 10:
                continue
            intervals[hormone] = {
                "lo": float(np.quantile(residual, tail)),
                "hi": float(np.quantile(residual, 1.0 - tail)),
                "n": int(residual.size),
            }

        self.artifact = TemporalStateArtifact(
            locf=locf,
            ridge=ridge,
            phase=phase,
            target_models=dict(self.target_models),
            channels=CHANNELS,
            lookback=self.lookback,
            residual_intervals=intervals,
            normalization_stats=normalization_stats or {},
            model_version=self.model_version,
        )
        return self

    def _require(self) -> TemporalStateArtifact:
        if self.artifact is None:
            raise RuntimeError("TemporalStateEncoder is not fitted. Call fit() or load().")
        return self.artifact

    def save(self, directory: str | Path) -> Path:
        """Persist every head, rule and statistic together."""
        import joblib  # noqa: PLC0415

        artifact = self._require()
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, destination / _ARTIFACT_NAME)

        (destination / "target_models.json").write_text(
            json.dumps(
                {
                    "target_models": artifact.target_models,
                    "cycle_phase": "logistic_phase",
                    "selection_basis": (
                        "Winners of the frozen benchmark on held-out participants. LOCF "
                        "beats ridge for LH and PdG; ridge narrowly beats LOCF for E3G."
                    ),
                    "classes": artifact.phase.classes_,
                },
                indent=2,
            )
            + "\n"
        )
        (destination / "feature_manifest.json").write_text(
            json.dumps(
                {
                    "channels": list(artifact.channels),
                    "lookback_days": artifact.lookback,
                    "window_layout": "[values(lookback x C), observed_mask(lookback x C)]",
                    "task_type": "causal_forecasting",
                    "missingness_rule": (
                        "Unobserved entries enter as 0 accompanied by their mask channel; "
                        "they are never imputed silently."
                    ),
                },
                indent=2,
            )
            + "\n"
        )
        (destination / "normalization_stats.json").write_text(
            json.dumps(artifact.normalization_stats, indent=2) + "\n"
        )
        (destination / "benchmark_metrics.json").write_text(
            json.dumps(
                {
                    "benchmark_version": artifact.benchmark_version,
                    "split_version": artifact.split_version,
                    "residual_intervals": artifact.residual_intervals,
                    "metrics": artifact.metrics,
                },
                indent=2,
            )
            + "\n"
        )
        return destination / _ARTIFACT_NAME

    @classmethod
    def load(cls, directory: str | Path) -> TemporalStateEncoder:
        """Restore an encoder saved by :meth:`save`."""
        import joblib  # noqa: PLC0415

        path = Path(directory) / _ARTIFACT_NAME
        if not path.exists():
            raise FileNotFoundError(
                f"No temporal encoder at {path}. Train one with "
                "scripts/train_temporal_state_encoder.py."
            )
        artifact: TemporalStateArtifact = joblib.load(path)
        encoder = cls(
            lookback=artifact.lookback,
            target_models=artifact.target_models,
            model_version=artifact.model_version,
        )
        encoder.artifact = artifact
        return encoder

    # -- prediction --------------------------------------------------------

    def predict(self, rows: list[dict[str, Any]], participant_ids: list[str]) -> dict[str, Any]:
        """Predict hormones and cycle phase for every day of the given participants."""
        artifact = self._require()
        data = build_sequences(
            rows, participant_ids, channels=artifact.channels, lookback=artifact.lookback
        )
        if len(data["values"]) == 0:
            return {"data": data, "hormones": {}, "phase": np.array([]), "phase_proba": None}

        predictions = {
            "locf": artifact.locf.predict(data),
            "ridge_window": artifact.ridge.predict(data),
        }
        hormones = {
            hormone: predictions[artifact.target_models.get(hormone, "locf")][
                :, artifact.channels.index(hormone)
            ]
            for hormone in HORMONES
        }
        return {
            "data": data,
            "hormones": hormones,
            "phase": artifact.phase.predict(data),
            "phase_proba": artifact.phase.predict_proba(data),
            "phase_classes": artifact.phase.classes_,
        }

    def evaluate(self, rows: list[dict[str, Any]], participant_ids: list[str]) -> dict[str, Any]:
        """MAE per hormone on observed days only, plus phase accuracy."""
        from sklearn.metrics import balanced_accuracy_score, f1_score  # noqa: PLC0415

        result = self.predict(rows, participant_ids)
        data = result["data"]
        artifact = self._require()

        metrics: dict[str, Any] = {"hormones": {}, "cycle_phase": {}}
        for hormone in HORMONES:
            index = artifact.channels.index(hormone)
            observed = data["observed"][:, index]
            truth = data["values"][observed, index]
            predicted = result["hormones"][hormone][observed]
            finite = np.isfinite(truth) & np.isfinite(predicted)
            metrics["hormones"][hormone] = {
                "method": artifact.target_models.get(hormone, "locf"),
                "mae": float(np.mean(np.abs(truth[finite] - predicted[finite])))
                if finite.any()
                else float("nan"),
                "n_observed": int(finite.sum()),
            }

        labelled = data["phase"] != "unknown"
        if labelled.any():
            metrics["cycle_phase"] = {
                "macro_f1": float(
                    f1_score(
                        data["phase"][labelled],
                        result["phase"][labelled],
                        average="macro",
                        zero_division=0,
                    )
                ),
                "balanced_accuracy": float(
                    balanced_accuracy_score(data["phase"][labelled], result["phase"][labelled])
                ),
                "n": int(labelled.sum()),
            }
        return metrics

    # -- token export ------------------------------------------------------

    def export_token(self, payload: Any, *, patient_id: str) -> ModalityToken:
        """Encode one participant's recent history into a :class:`ModalityToken`.

        Args:
            payload: Participant-day records (a list, or an object with
                ``participant_days``).
            patient_id: Patient identifier.

        Returns:
            A current-state token. It never carries a PCOS probability or
            subtype -- the temporal branch reports physiological state, and any
            diagnostic reading of it happens in the adapter under explicit rules.
        """
        artifact = self._require()
        rows = _coerce_rows(payload, patient_id)

        warnings: list[str] = ["Current-state estimate; not a PCOS diagnosis"]
        missing: list[str] = []

        if not rows:
            return self._abstain(patient_id, "No longitudinal data supplied.", warnings)

        participant_ids = sorted({str(row["participant_id"]) for row in rows})
        n_days = len(rows)
        if n_days < MIN_DAYS_FOR_STATE:
            return self._abstain(
                patient_id,
                f"Only {n_days} days of usable data were available; at least "
                f"{MIN_DAYS_FOR_STATE} are required to characterise current state.",
                warnings,
            )

        result = self.predict(rows, participant_ids)
        data = result["data"]
        last = len(data["values"]) - 1

        structured: dict[str, Any] = {
            "task_type": "causal_forecasting",
            "input_window_days": artifact.lookback,
            "observed_days": n_days,
        }

        # Coverage: fraction of channel-days genuinely observed in the window.
        window_start = max(last - artifact.lookback + 1, 0)
        coverage = float(data["observed"][window_start : last + 1].mean())
        structured["input_coverage"] = round(coverage, 4)

        for hormone in HORMONES:
            index = artifact.channels.index(hormone)
            value = float(result["hormones"][hormone][last])
            method = artifact.target_models.get(hormone, "locf")
            interval = artifact.residual_intervals.get(hormone)
            structured[f"predicted_{hormone}"] = value
            structured[f"predicted_{hormone}_method"] = method
            if interval:
                # Flat scalars, not a list: ModalityToken.structured_features is
                # dict[str, float|int|str|bool|None] by design, so every modality's
                # features stay directly comparable. Nesting here would break the
                # shared envelope for every consumer.
                structured[f"predicted_{hormone}_interval_low"] = round(value + interval["lo"], 4)
                structured[f"predicted_{hormone}_interval_high"] = round(value + interval["hi"], 4)
            if not data["observed"][:, index].any():
                missing.append(hormone)

        probabilities: dict[str, float] = {}
        if result["phase_proba"] is not None:
            probabilities = {
                str(cls): float(result["phase_proba"][last][position])
                for position, cls in enumerate(result["phase_classes"])
            }
            structured["predicted_cycle_phase"] = str(result["phase"][last])
            # One flat key per class, for the same envelope reason as above.
            for cls, probability in probabilities.items():
                structured[f"cycle_phase_probability_{cls}"] = round(probability, 4)
            entropy = -sum(p * np.log(max(p, 1e-12)) for p in probabilities.values())
            # Consumed by inference/domain_mapper.py as current_state confidence.
            structured["cycle_phase_entropy"] = round(float(entropy), 4)
        else:
            missing.append("cycle_phase")

        # Cycle irregularity feeds the REPRODUCTIVE domain. Derived from how much
        # the predicted phase sequence churns: a short window that keeps changing
        # phase is not evidence of a regular cycle. Deliberately conservative --
        # a 14-day window cannot characterise long-term regularity, and the
        # coordinator's disagreement rule exists to say so.
        if len(result["phase"]) > 1:
            recent = result["phase"][window_start : last + 1]
            transitions = float(np.mean(recent[1:] != recent[:-1])) if len(recent) > 1 else 0.0
            structured["cycle_irregularity"] = round(min(transitions * 2.0, 1.0), 4)
            warnings.append(
                "cycle_irregularity is computed over a short observation window and "
                "cannot characterise long-term cycle regularity."
            )

        for channel in artifact.channels:
            index = artifact.channels.index(channel)
            if not data["observed"][:, index].any():
                missing.append(channel)

        return ModalityToken(
            patient_id=patient_id,
            modality="longitudinal_hormonal_state",
            structured_features=structured,
            quality_score=round(min(max(coverage, 0.0), 1.0), 4),
            # Confidence from phase certainty: a near-uniform posterior over four
            # phases means the current state is not actually known.
            confidence_score=round(float(max(probabilities.values())) if probabilities else 0.0, 4),
            model_version=artifact.model_version,
            source_dataset=artifact.source_dataset,
            missing_fields=sorted(set(missing)),
            warnings=warnings,
        )

    def _abstain(self, patient_id: str, reason: str, warnings: list[str]) -> ModalityToken:
        """Emit a token that carries no state estimate at all."""
        return ModalityToken(
            patient_id=patient_id,
            modality="longitudinal_hormonal_state",
            structured_features={
                "temporal_state_available": False,
                "reason": reason,
                "task_type": "causal_forecasting",
            },
            quality_score=0.0,
            confidence_score=0.0,
            model_version=self.model_version,
            source_dataset="mcPHASES",
            missing_fields=["predicted_cycle_phase", "predicted_lh", "predicted_e3g"],
            warnings=[*warnings, reason],
        )


def _coerce_rows(payload: Any, patient_id: str) -> list[dict[str, Any]]:
    """Accept participant-day dicts, or an object exposing ``participant_days``."""
    if payload is None:
        return []
    if hasattr(payload, "participant_days"):
        payload = payload.participant_days
    if not isinstance(payload, list):
        raise TypeError(
            f"Unsupported temporal payload {type(payload).__name__}; expected a list of "
            "participant-day records or an object with .participant_days."
        )

    rows: list[dict[str, Any]] = []
    for item in payload:
        row = item if isinstance(item, dict) else getattr(item, "__dict__", None)
        if row is None:
            continue
        rows.append({**row, "participant_id": row.get("participant_id", patient_id)})
    return rows
