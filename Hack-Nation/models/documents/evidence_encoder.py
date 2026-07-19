"""Encode document-extracted evidence into a :class:`ModalityToken`.

Why the token carries canonical values only: the source value is the reviewer's
reference point, but a model that saw 2.7 (nmol/L) for one patient and 78
(ng/dL) for another would learn a unit, not a biology. The source values stay in
the events and in the extraction records; the token is the harmonized view.

Why unconfirmed document values can still enter the token when explicitly
allowed: unlike speech, a lab PDF is a primary record, and there are legitimate
research uses for unreviewed extractions. But it is never the default, the
caller must opt in, and the token says so in ``warnings`` so no downstream
report can claim these numbers were clinician-confirmed.
"""

from __future__ import annotations

from statistics import fmean

from schemas.event import HormonalHealthEvent
from schemas.modality_token import ModalityToken

ENCODER_VERSION = "document_evidence_encoder/1.0.0"

DOCUMENT_MODALITIES = {"laboratory", "clinical_document", "ultrasound_report"}

#: Fixed feature order, so the embedding is comparable across patients and runs.
FEATURE_ORDER: tuple[str, ...] = (
    "total_testosterone",
    "free_testosterone",
    "dheas",
    "shbg",
    "luteinizing_hormone",
    "follicle_stimulating_hormone",
    "anti_mullerian_hormone",
    "estradiol",
    "progesterone",
    "fasting_glucose",
    "fasting_insulin",
    "hdl_cholesterol",
    "ldl_cholesterol",
    "triglycerides",
    "follicle_count_left",
    "follicle_count_right",
    "ovary_volume_ml",
)


def encode_document_events(
    events: list[HormonalHealthEvent],
    *,
    patient_id: str,
    source_dataset: str | None = None,
    observed_at: str | None = None,
    n_unsupported: int = 0,
    allow_unconfirmed: bool = False,
) -> ModalityToken:
    """Build the ``clinical_document`` token for one patient.

    Args:
        events: Document-derived events.
        patient_id: Patient the token describes.
        source_dataset: Dataset id recorded on the token.
        observed_at: ISO timestamp of the document, if known.
        n_unsupported: Values dropped for missing page grounding, for the record.
        allow_unconfirmed: Include ``awaiting_*`` events. Off by default.

    Returns:
        A :class:`ModalityToken` with ``modality='clinical_document'``.
    """
    warnings: list[str] = []
    document_events = [e for e in events if e.modality in DOCUMENT_MODALITIES]

    if allow_unconfirmed:
        usable = [e for e in document_events if e.missingness_status == "observed"]
        unconfirmed = [e for e in usable if not e.is_model_ready]
        if unconfirmed:
            warnings.append(
                f"{len(unconfirmed)} unconfirmed document value(s) INCLUDED via "
                "allow_unconfirmed=True; these are machine reads, not clinician-confirmed."
            )
    else:
        usable = [e for e in document_events if e.is_model_ready]
        excluded = [e for e in document_events if not e.is_model_ready]
        if excluded:
            warnings.append(
                f"{len(excluded)} unconfirmed document event(s) excluded from the token."
            )

    if n_unsupported:
        warnings.append(f"{n_unsupported} value(s) dropped for missing page grounding.")
    if not usable:
        warnings.append("no usable document events; token carries no structured features.")

    structured: dict[str, bool | float | int | str | None] = {}
    units: dict[str, str | None] = {}
    for event in usable:
        code = event.canonical_variable_code
        value = event.value
        if isinstance(value, (bool, int, float)):
            structured[code] = value if isinstance(value, bool) else float(value)
        else:
            structured[code] = str(value)
        units[code] = event.unit

    for code, unit in units.items():
        if unit:
            structured[f"{code}_unit"] = unit

    def _numeric_channel(code: str) -> float:
        """Value channel for one feature; 0.0 stands in for "absent or non-numeric".

        The zero is only interpretable alongside ``presence_mask`` below, which
        is what distinguishes an absent analyte from a genuine zero reading.
        Booleans are excluded deliberately: ``True`` is not the number 1 here.
        """
        value = structured.get(code)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return 0.0
        return float(value)

    embedding = [_numeric_channel(code) for code in FEATURE_ORDER]
    presence_mask = [1.0 if code in structured else 0.0 for code in FEATURE_ORDER]

    confidences = [e.extraction_confidence for e in usable]
    confidence = float(fmean(confidences)) if confidences else 0.0
    coverage = sum(presence_mask) / len(FEATURE_ORDER)

    missing = [code for code in FEATURE_ORDER if code not in structured]

    return ModalityToken(
        patient_id=patient_id,
        modality="clinical_document",
        # Value channel followed by a presence mask: a missing analyte reads as
        # 0.0 in the value channel, and the mask is what distinguishes that from
        # a genuine zero.
        embedding=embedding + presence_mask,
        structured_features=structured,
        quality_score=round(min(1.0, coverage * 2.0) * confidence, 4),
        confidence_score=round(confidence, 4),
        observed_at=observed_at,
        model_version=ENCODER_VERSION,
        source_dataset=source_dataset,
        provenance_ids=[str(e.event_id) for e in usable],
        missing_fields=missing,
        warnings=warnings,
    )
