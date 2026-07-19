"""The dynamic hormonal-state model: participant-days in, current state out.

What this model claims, precisely: given the previous 14-30 days of a
participant's data, here is an estimate of **where that person is right now** —
their likely cycle phase, their likely hormone levels, and the symptoms they are
likely to report tomorrow.

What it does **not** claim, and what no consumer of its output may infer: a
subtype, a phenotype, a diagnosis, or any trait. State is a property of *today*
and changes tomorrow; a trait would be a property of the person. Conflating the
two is the specific error this module is written to make impossible: the exported
:class:`ModalityToken` uses modality ``longitudinal_hormonal_state``, the output
carries a fixed ``interpretation`` string, and both carry explicit warnings.

``input_coverage`` is reported on every estimate. A state estimate built from a
window where 90% of values were missing is a different object from one built on
dense data, and hiding that behind a confident point prediction would be the
most misleading thing this module could do.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from models.temporal.gru import (
    DEFAULT_LOOKBACK_DAYS,
    FeatureSpec,
    build_dataset,
    build_sequence_encoder,
    fit_feature_spec,
)
from models.temporal.heads import (
    CYCLE_CLASSES,
    HORMONE_TARGETS,
    CycleStateHead,
    HormoneReconstructionHead,
    MaskedReconstructionHead,
    SymptomHead,
    TemporalHeads,
)
from models.temporal.losses import LossWeights, make_artificial_mask, state_loss
from models.temporal.tcn import build_tcn
from schemas.modality_token import ModalityToken
from schemas.model_output import ModelCardMetadata
from schemas.temporal import CyclePhase, ParticipantDay, TemporalStateOutput

#: Repeated verbatim on every token and output. State, never trait.
STATE_NOT_SUBTYPE_WARNING = (
    "This is a current hormonal-state estimate for one day. It is not a subtype, "
    "phenotype, diagnosis, or clinical assessment, and it does not describe a "
    "stable trait of this person."
)

#: Coverage below which the estimate is flagged as thinly supported.
LOW_COVERAGE_THRESHOLD = 0.25


@dataclass
class TemporalTrainingReport:
    """What one fit produced, for the experiment record."""

    n_windows: int
    n_participants: int
    lookback_days: int
    encoder: str
    embedding_dim: int
    losses: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def grouped_participant_split(
    groups: list[str], *, test_fraction: float = 0.3, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Split window indices by PARTICIPANT, never by day.

    Splitting days randomly would put day 40 of a participant in train and day 41
    in test. Those two rows share a lookback window, a cycle and a body, so the
    test score would measure interpolation within a known person and would be
    wildly optimistic about a new person. Every split in this module is therefore
    grouped.

    This is the implementation. ``training.splits`` exposes only *manifest*
    builders (k-fold, LOPO, holdout) and no plain grouped train/test helper, so
    there is nothing to delegate to; an earlier version tried to import one and
    silently fell through to this code on every call.

    Args:
        groups: Participant id per window.
        test_fraction: Fraction of *participants* (not windows) held out.
        seed: RNG seed.

    Returns:
        ``(train_indices, test_indices)``.
    """
    unique = sorted(set(groups))
    rng = np.random.default_rng(seed)
    shuffled = list(unique)
    rng.shuffle(shuffled)
    n_test = max(1, int(round(len(shuffled) * test_fraction)))
    test_ids = set(shuffled[:n_test])
    array = np.asarray(groups)
    test_index = np.where(np.isin(array, list(test_ids)))[0]
    train_index = np.where(~np.isin(array, list(test_ids)))[0]
    return train_index, test_index


def _hormone_targets(days: list[ParticipantDay]) -> tuple[np.ndarray, np.ndarray]:
    """Hormone target matrix and its observation mask. Missing stays NaN."""
    values = np.full((len(days), len(HORMONE_TARGETS)), np.nan)
    mask = np.zeros_like(values, dtype=bool)
    for row, day in enumerate(days):
        for col, name in enumerate(HORMONE_TARGETS):
            raw = day.values.get(name)
            if day.is_observed.get(name) and raw is not None:
                values[row, col] = float(raw)
                mask[row, col] = True
    return values, mask


def _symptom_targets(
    days: list[ParticipantDay], all_days_by_participant: dict[str, list[ParticipantDay]]
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """NEXT-day symptom targets, with a mask marking where a next day exists."""
    names = sorted({key for day in days for key in day.daily_symptoms})
    targets = {name: np.zeros(len(days)) for name in names}
    masks = {name: np.zeros(len(days), dtype=bool) for name in names}
    lookup = {
        (participant, day.study_day): day
        for participant, series in all_days_by_participant.items()
        for day in series
    }
    for row, day in enumerate(days):
        nxt = lookup.get((day.participant_id, day.study_day + 1))
        if nxt is None:
            continue  # No next day: masked out, never defaulted to False.
        for name in names:
            if name in nxt.daily_symptoms:
                targets[name][row] = float(nxt.daily_symptoms[name])
                masks[name][row] = True
    return targets, masks


def _window_coverage(x: np.ndarray, spec: FeatureSpec) -> np.ndarray:
    """Mean observation rate across the window, per sample.

    The ``is_observed`` columns sit at stride 3, offset 1, in the feature layout.
    """
    n_channels = len(spec.channels)
    if n_channels == 0:
        return np.zeros(x.shape[0])
    indicator_columns = [3 * i + 1 for i in range(n_channels)]
    return np.asarray(x[:, :, indicator_columns].mean(axis=(1, 2)), dtype=float)


class TemporalStateModel:
    """Encoder plus heads, producing :class:`TemporalStateOutput` per day.

    Conforms to the ``BasePrismModel`` interface (``fit`` / ``predict`` /
    ``export_model_card_metadata``); the base class is imported defensively so
    this module does not depend on the order modules are added to the repo.
    """

    model_name = "longitudinal_hormonal_state_model"
    model_version = "0.1.0"
    modality = "longitudinal_hormonal_state"

    def __init__(
        self,
        *,
        lookback_days: int = DEFAULT_LOOKBACK_DAYS,
        hidden_size: int = 32,
        encoder_kind: str = "gru",
        backend: str = "auto",
        use_decay: bool = True,
        decay_gamma: float = 0.35,
        loss_weights: LossWeights | None = None,
        channel_groups: dict[str, str] | None = None,
        seed: int = 0,
    ) -> None:
        """
        Args:
            lookback_days: Window length, clamped to [14, 30].
            hidden_size: State embedding dimension.
            encoder_kind: ``"gru"`` or ``"tcn"``.
            backend: ``"auto"``, ``"torch"`` or ``"numpy"``.
            use_decay: Enable GRU-D style decay of stale carried values.
            decay_gamma: Decay rate per day of staleness.
            loss_weights: Weights of the four L_state terms.
            channel_groups: Channel -> modality group, for the indicator block.
            seed: RNG seed.
        """
        self.lookback_days = lookback_days
        self.hidden_size = hidden_size
        self.encoder_kind = encoder_kind
        self.backend = backend
        self.use_decay = use_decay
        self.decay_gamma = decay_gamma
        self.loss_weights = loss_weights or LossWeights()
        self.channel_groups = dict(channel_groups or {})
        self.seed = seed

        self.spec: FeatureSpec | None = None
        self.encoder: Any | None = None
        self.heads = TemporalHeads()
        self.report: TemporalTrainingReport | None = None

    # -- fitting -----------------------------------------------------------

    def _build_encoder(self, input_size: int) -> Any:
        if self.encoder_kind == "tcn":
            return build_tcn(
                input_size, hidden_size=self.hidden_size, backend=self.backend, seed=self.seed
            )
        return build_sequence_encoder(
            input_size, hidden_size=self.hidden_size, backend=self.backend, seed=self.seed
        )

    def embed(
        self, days: list[ParticipantDay]
    ) -> tuple[np.ndarray, list[ParticipantDay], list[str], np.ndarray]:
        """Build windows and encode them into state embeddings.

        Returns:
            ``(embeddings, target_days, participant_ids, coverage)``.
        """
        if self.spec is None:
            raise RuntimeError("Call fit() before embed().")
        x, targets, groups = build_dataset(days, self.spec, decay_gamma=self.decay_gamma)
        if x.shape[0] == 0:
            return np.zeros((0, self.hidden_size)), [], [], np.zeros(0)
        if self.encoder is None:
            self.encoder = self._build_encoder(x.shape[2])
        return self.encoder(x), targets, groups, _window_coverage(x, self.spec)

    def fit(self, days: list[ParticipantDay]) -> TemporalStateModel:
        """Fit the feature spec, the encoder and all four heads.

        Args:
            days: Participant-days from the training participants only. It is
                the caller's job to have split by participant; see
                :func:`grouped_participant_split`.
        """
        self.spec = fit_feature_spec(
            days,
            channel_groups=self.channel_groups,
            lookback_days=self.lookback_days,
            use_decay=self.use_decay,
        )
        embeddings, targets, groups, _ = self.embed(days)
        warnings: list[str] = []
        if embeddings.shape[0] == 0:
            raise ValueError(
                f"No windows could be built: every participant has fewer than "
                f"{self.spec.lookback_days} days."
            )

        hormone_values, hormone_mask = _hormone_targets(targets)
        self.heads.hormone = HormoneReconstructionHead().fit(
            embeddings, hormone_values, hormone_mask
        )

        labelled = [i for i, day in enumerate(targets) if day.cycle_phase in CYCLE_CLASSES]
        if labelled:
            self.heads.cycle = CycleStateHead().fit(
                embeddings[labelled], [targets[i].cycle_phase for i in labelled]
            )
        else:
            warnings.append("No cycle-phase labels available; the cycle head is uninformative.")

        by_participant: dict[str, list[ParticipantDay]] = {}
        for day in days:
            by_participant.setdefault(day.participant_id, []).append(day)
        symptom_targets, symptom_masks = _symptom_targets(targets, by_participant)
        if symptom_targets:
            self.heads.symptom = SymptomHead().fit(embeddings, symptom_targets, symptom_masks)
        else:
            warnings.append("No symptom reports available; the symptom head is uninformative.")

        channel_values, channel_mask = self._channel_targets(targets)
        artificial = make_artificial_mask(channel_mask, mask_fraction=0.2, seed=self.seed)
        self.heads.masked = MaskedReconstructionHead().fit(
            embeddings, channel_values, artificial.astype(bool)
        )

        self.report = TemporalTrainingReport(
            n_windows=int(embeddings.shape[0]),
            n_participants=len(set(groups)),
            lookback_days=self.spec.lookback_days,
            encoder=type(self.encoder).__name__,
            embedding_dim=int(embeddings.shape[1]),
            losses=self.training_losses(embeddings, targets, channel_values, artificial),
            warnings=warnings,
        )
        return self

    def _channel_targets(self, days: list[ParticipantDay]) -> tuple[np.ndarray, np.ndarray]:
        """Standardised values and observation mask for every channel."""
        assert self.spec is not None
        channels = self.spec.channels
        values = np.zeros((len(days), len(channels)))
        mask = np.zeros_like(values, dtype=bool)
        for row, day in enumerate(days):
            for col, channel in enumerate(channels):
                raw = day.values.get(channel)
                if day.is_observed.get(channel) and raw is not None:
                    mean = self.spec.channel_means.get(channel, 0.0)
                    scale = self.spec.channel_scales.get(channel, 1.0)
                    values[row, col] = (float(raw) - mean) / scale
                    mask[row, col] = True
        return values, mask

    def training_losses(
        self,
        embeddings: np.ndarray,
        targets: list[ParticipantDay],
        channel_values: np.ndarray,
        artificial_mask: np.ndarray,
    ) -> dict[str, float]:
        """Evaluate L_state and its components on the fitted training windows."""
        hormone_values, hormone_mask = _hormone_targets(targets)
        predictions = self.heads.hormone.predict(embeddings)
        hormone_pred = np.column_stack([predictions[name] for name in HORMONE_TARGETS])

        labelled = [i for i, day in enumerate(targets) if day.cycle_phase in CYCLE_CLASSES]
        cycle_probs = self.heads.cycle.predict_proba(embeddings[labelled]) if labelled else None
        cycle_target = (
            np.asarray([CYCLE_CLASSES.index(targets[i].cycle_phase) for i in labelled])
            if labelled
            else None
        )

        symptom_probs = None
        symptom_target = None
        predicted_symptoms = self.heads.symptom.predict_proba(embeddings)
        if predicted_symptoms:
            names = sorted(predicted_symptoms)
            symptom_probs = np.column_stack([predicted_symptoms[n] for n in names])
            symptom_target = np.column_stack(
                [[float(day.daily_symptoms.get(n, False)) for day in targets] for n in names]
            )

        return state_loss(
            hormone_pred=hormone_pred,
            hormone_target=np.nan_to_num(hormone_values),
            hormone_mask=hormone_mask.astype(float),
            cycle_probs=cycle_probs,
            cycle_target=cycle_target,
            symptom_probs=symptom_probs,
            symptom_target=symptom_target,
            masked_pred=self.heads.masked.predict(embeddings),
            masked_target=channel_values,
            masked_mask=artificial_mask,
            weights=self.loss_weights,
        )

    # -- prediction --------------------------------------------------------

    def predict(self, days: list[ParticipantDay]) -> list[TemporalStateOutput]:
        """Produce a current-state estimate for each window that can be built."""
        embeddings, targets, _, coverage = self.embed(days)
        return [
            self._make_output(embeddings[i : i + 1], targets[i], float(coverage[i]))
            for i in range(embeddings.shape[0])
        ]

    def predict_latest(self, days: list[ParticipantDay]) -> TemporalStateOutput | None:
        """State estimate for the most recent day only, per participant window."""
        outputs = self.predict(days)
        return outputs[-1] if outputs else None

    def _make_output(
        self, embedding: np.ndarray, day: ParticipantDay, coverage: float
    ) -> TemporalStateOutput:
        """Assemble one :class:`TemporalStateOutput` with uncertainty."""
        hormones, residuals = self.heads.hormone.predict_with_uncertainty(embedding)
        cycle_probs = self.heads.cycle.predict_proba(embedding)[0]
        entropy = float(self.heads.cycle.predict_entropy(embedding)[0])
        symptoms = self.heads.symptom.predict_proba(embedding)

        warnings = [STATE_NOT_SUBTYPE_WARNING]
        if coverage < LOW_COVERAGE_THRESHOLD:
            warnings.append(
                f"Input coverage is only {coverage:.0%} of channel-days over the "
                f"{self.lookback_days}-day window; this estimate is thinly supported."
            )
        if entropy > 0.85:
            warnings.append(
                "Cycle-phase distribution is close to uniform; the phase is effectively unknown."
            )

        uncertainty: dict[str, float] = {
            f"{k}_residual_sd": float(v) for k, v in residuals.items() if np.isfinite(v)
        }
        uncertainty["cycle_phase_entropy"] = entropy
        uncertainty["input_coverage"] = float(coverage)

        return TemporalStateOutput(
            patient_id=day.participant_id,
            as_of_date=day.calendar_date or f"study_day_{day.study_day}",
            state_embedding=[float(v) for v in np.asarray(embedding).ravel()],
            hormone_predictions={k: float(v[0]) for k, v in hormones.items()},
            cycle_phase_probabilities={
                phase: float(cycle_probs[i]) for i, phase in enumerate(CYCLE_CLASSES)
            },
            symptom_probabilities={k: float(v[0]) for k, v in symptoms.items()},
            uncertainty=uncertainty,
            input_coverage=float(np.clip(coverage, 0.0, 1.0)),
            lookback_days=self.lookback_days,
            model_version=self.model_version,
            warnings=warnings,
        )

    # -- token export ------------------------------------------------------

    def to_token(
        self, output: TemporalStateOutput, *, source_dataset: str | None = None
    ) -> ModalityToken:
        """Export a state estimate as a ``longitudinal_hormonal_state`` token.

        The modality name, the structured features and the warnings all say
        *state*. Nothing in this token may be read as a subtype or a diagnosis.
        """
        predicted: CyclePhase = output.predicted_phase()
        structured: dict[str, Any] = {
            "predicted_cycle_phase": predicted,
            "cycle_phase_entropy": float(output.uncertainty.get("cycle_phase_entropy", 1.0)),
            "input_coverage": float(output.input_coverage),
            "lookback_days": int(output.lookback_days),
            "interpretation": output.interpretation,
        }
        for phase, probability in output.cycle_phase_probabilities.items():
            structured[f"p_{phase}"] = float(probability)
        for hormone, value in output.hormone_predictions.items():
            structured[f"predicted_{hormone}"] = float(value)
        for symptom, probability in output.symptom_probabilities.items():
            structured[f"p_symptom_{symptom}"] = float(probability)

        confidence = float(
            np.clip(1.0 - output.uncertainty.get("cycle_phase_entropy", 1.0), 0.0, 1.0)
        )
        return ModalityToken(
            patient_id=output.patient_id,
            modality="longitudinal_hormonal_state",
            embedding=list(output.state_embedding),
            structured_features=structured,
            quality_score=float(np.clip(output.input_coverage, 0.0, 1.0)),
            confidence_score=confidence,
            observed_at=output.as_of_date,
            model_version=self.model_version,
            source_dataset=source_dataset,
            missing_fields=[
                name for name in HORMONE_TARGETS if name not in output.hormone_predictions
            ],
            warnings=[
                STATE_NOT_SUBTYPE_WARNING,
                *output.warnings,
            ],
        )

    def export_model_card_metadata(self) -> ModelCardMetadata:
        """Model card stating the state/trait boundary explicitly."""
        return ModelCardMetadata(
            model_name=self.model_name,
            model_version=self.model_version,
            intended_use=(
                "Research-only estimation of a participant's CURRENT hormonal state "
                f"(cycle phase, hormone levels, next-day symptom probabilities) from the "
                f"previous {self.lookback_days} participant-days."
            ),
            out_of_scope_uses=[
                "Assigning a subtype, phenotype, or diagnosis of any kind.",
                "Inferring a stable trait of a person from a state estimate.",
                "Contraceptive or fertility decision-making.",
                "Any clinical decision-making.",
                "Application to participants from a different dataset than the one fitted.",
            ],
            limitations=[
                "Trained and evaluated on synthetic longitudinal data in this repository.",
                "Missingness is non-ignorable; estimates on sparse windows are weak, which "
                "input_coverage reports rather than hides.",
                "The peri-ovulatory class is short and the hardest to call correctly.",
                "The torch-free fallback encoder uses fixed random recurrent weights.",
            ],
            ethical_considerations=[
                "State estimates are easily over-read as diagnoses; every output and token "
                "carries an explicit state-not-trait disclaimer.",
                "Splits are always grouped by participant, so reported performance refers to "
                "unseen people rather than unseen days of known people.",
            ],
        )


__all__ = [
    "LOW_COVERAGE_THRESHOLD",
    "STATE_NOT_SUBTYPE_WARNING",
    "TemporalStateModel",
    "TemporalTrainingReport",
    "grouped_participant_split",
]
