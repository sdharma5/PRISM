"""All five modality tokens must serialize under one shared contract.

This is the Milestone 9 definition of done. It is deliberately a *contract* test
rather than a fusion test: the tokens share an envelope so they are comparable,
not so they can be concatenated into a model. See ADR-002.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from schemas.modality_token import ModalityToken

#: The five tokens produced at the end of Step 9.
EXPECTED_MODALITIES = [
    "static_clinical",
    "speech_symptoms",
    "clinical_document",
    "ovarian_ultrasound",
    "longitudinal_hormonal_state",
]

ENVELOPE_FIELDS = {
    "patient_id",
    "modality",
    "embedding",
    "structured_features",
    "quality_score",
    "confidence_score",
    "observed_at",
    "model_version",
    "source_dataset",
    "provenance_ids",
    "missing_fields",
    "warnings",
}


def _token(modality: str) -> ModalityToken:
    return ModalityToken(
        patient_id="SYNTH-001",
        modality=modality,  # type: ignore[arg-type]
        embedding=[0.1, -0.2, 0.3],
        structured_features={"example_feature": 1.0},
        quality_score=0.8,
        confidence_score=0.7,
        source_dataset="synthetic",
    )


@pytest.mark.parametrize("modality", EXPECTED_MODALITIES)
def test_every_token_uses_the_same_envelope(modality: str) -> None:
    payload = _token(modality).model_dump(mode="json")
    assert set(payload) == ENVELOPE_FIELDS, (
        f"{modality} token envelope drifted. Every encoder must export the same "
        "fields so tokens stay comparable across modalities."
    )


@pytest.mark.parametrize("modality", EXPECTED_MODALITIES)
def test_token_round_trips_through_disk(modality: str, tmp_path: Path) -> None:
    original = _token(modality)
    path = original.write_json(tmp_path / f"{modality}_token.json")

    restored = ModalityToken.read_json(path)
    assert restored == original

    # The on-disk form must be plain JSON any consumer can read.
    raw = json.loads(path.read_text())
    assert raw["modality"] == modality
    assert raw["model_version"]


def test_all_five_tokens_serialize_together(tmp_path: Path) -> None:
    """Milestone 9: all five tokens serialize under one shared contract."""
    written = [
        _token(modality).write_json(tmp_path / f"{modality}_token.json")
        for modality in EXPECTED_MODALITIES
    ]
    assert len(written) == 5

    loaded = [ModalityToken.read_json(p) for p in written]
    assert {t.modality for t in loaded} == set(EXPECTED_MODALITIES)

    # Same schema, five different modalities, one patient id namespace per dataset.
    assert len({tuple(sorted(t.model_dump().keys())) for t in loaded}) == 1


def test_unknown_modality_is_rejected() -> None:
    """A sixth token type cannot be introduced without updating the contract."""
    with pytest.raises(ValueError):
        ModalityToken(
            patient_id="SYNTH-001",
            modality="genomic",  # type: ignore[arg-type]
            quality_score=0.5,
            confidence_score=0.5,
        )


def test_scores_are_bounded() -> None:
    """Quality and confidence are probabilities, not arbitrary floats."""
    for bad in (-0.1, 1.1):
        with pytest.raises(ValueError):
            ModalityToken(
                patient_id="SYNTH-001",
                modality="static_clinical",
                quality_score=bad,
                confidence_score=0.5,
            )
