"""Encode confirmed speech events into a :class:`ModalityToken`.

Why only confirmed events: the token is a modelling input, and PRISM's whole
provenance argument collapses if an unreviewed extraction can reach a model
through a side door. The encoder therefore filters on
``HormonalHealthEvent.is_model_ready`` rather than trusting its caller.

Why quality and confidence are separate numbers: a pristine recording can still
yield a hedged, low-confidence extraction ("maybe twice a year?"), and a noisy
recording can yield an unambiguous one. Collapsing them would hide both cases.
"""

from __future__ import annotations

from statistics import fmean

from models.speech.event_embedding import EMBEDDING_VERSION, encode_codes
from schemas.event import HormonalHealthEvent
from schemas.modality_token import ModalityToken

ENCODER_VERSION = "speech_symptom_encoder/1.0.0"

#: Codes whose structured feature is reported as a plain boolean presence flag.
_BOOLEAN_SUFFIX = "_current"


def _feature_name(event: HormonalHealthEvent) -> str:
    code = event.canonical_variable_code
    if event.historical:
        return f"{code}_historical"
    return code


def encode_speech_events(
    events: list[HormonalHealthEvent],
    *,
    patient_id: str,
    audio_quality: float = 1.0,
    source_dataset: str | None = None,
    observed_at: str | None = None,
    n_unsupported: int = 0,
) -> ModalityToken:
    """Build the ``speech_symptoms`` token for one patient.

    Args:
        events: Speech-derived events; non-model-ready ones are excluded and
            reported in ``warnings``.
        patient_id: Patient the token describes.
        audio_quality: 0-1 quality score from ``ingestion.speech.audio``.
        source_dataset: Dataset id recorded on the token.
        observed_at: ISO timestamp of the recording, if known.
        n_unsupported: Extractions dropped for lacking evidence, for the record.

    Returns:
        A :class:`ModalityToken` with ``modality='speech_symptoms'``.
    """
    warnings: list[str] = []
    speech_events = [e for e in events if e.modality in {"patient_voice", "clinician_voice"}]
    excluded = [e for e in speech_events if not e.is_model_ready]
    usable = [e for e in speech_events if e.is_model_ready]

    if excluded:
        warnings.append(
            f"{len(excluded)} unconfirmed or non-observed speech event(s) excluded from the token."
        )
    if n_unsupported:
        warnings.append(f"{n_unsupported} extraction(s) dropped for missing evidence spans.")
    if not usable:
        warnings.append("no confirmed speech events; token carries no structured features.")

    structured: dict[str, bool | float | int | str | None] = {}
    present_codes: list[str] = []
    negated_codes: list[str] = []
    historical_codes: list[str] = []

    for event in usable:
        code = event.canonical_variable_code
        if event.negated:
            negated_codes.append(code)
        elif event.historical:
            historical_codes.append(code)
        else:
            present_codes.append(code)

        name = _feature_name(event)
        value = event.value
        if isinstance(value, bool):
            structured[name] = value and not event.negated
        elif isinstance(value, (int, float)):
            structured[name] = float(value)
        else:
            key = f"{name}{_BOOLEAN_SUFFIX}" if code == "medication_current" else name
            structured[f"{key}:{value}"] = not event.negated

    confidences = [e.extraction_confidence for e in usable]
    confidence = float(fmean(confidences)) if confidences else 0.0
    # Quality is the joint ceiling: bad audio caps a confident extraction, and a
    # hedged extraction caps a clean recording.
    quality = float(min(1.0, max(0.0, audio_quality))) * confidence

    return ModalityToken(
        patient_id=patient_id,
        modality="speech_symptoms",
        embedding=encode_codes(present_codes, negated_codes, historical_codes),
        structured_features=structured,
        quality_score=round(quality, 4),
        confidence_score=round(confidence, 4),
        observed_at=observed_at,
        model_version=f"{ENCODER_VERSION}+{EMBEDDING_VERSION}",
        source_dataset=source_dataset,
        provenance_ids=[str(e.event_id) for e in usable],
        missing_fields=sorted({e.canonical_variable_code for e in excluded}),
        warnings=warnings,
    )
