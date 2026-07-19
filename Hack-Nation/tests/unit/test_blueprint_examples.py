"""The three extraction examples quoted verbatim from the PRISM blueprint.

These are the reference cases the speech-extraction contract was specified
against. They are asserted individually rather than only through aggregate
corpus metrics, because each one pins a distinction that an averaged F1 would
happily absorb:

* U073 — an approximate range plus a duration is still a *present* finding.
* U074 — "I had X, but not now" is *historical and resolved*, not *negated*.
* U075 — a relative's diagnosis is never the patient's.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from ingestion.speech.extraction import ExtractionResult, RuleBasedExtractor
from ingestion.speech.transcription import ScriptedTranscriptionAdapter

EXAMPLE_1 = "My periods have been between 45 and 70 days apart for about a year."
EXAMPLE_2 = "I had acne in high school, but I do not have it now."
EXAMPLE_3 = "My sister has PMOS."


@pytest.fixture(scope="module")
def extractor() -> RuleBasedExtractor:
    return RuleBasedExtractor()


@pytest.fixture(scope="module")
def transcriber() -> ScriptedTranscriptionAdapter:
    return ScriptedTranscriptionAdapter()


def extract(transcriber, extractor, text: str, speaker_role: str = "patient") -> ExtractionResult:
    transcript = transcriber.transcribe(
        [{"speaker_role": speaker_role, "text": text}], recording_id="REC-BLUEPRINT"
    )
    return extractor.extract(transcript, patient_id="P-BLUEPRINT")


def find(result: ExtractionResult, code: str):
    matches = [e for e in result.events if e.canonical_code == code]
    assert matches, (
        f"expected an event for '{code}', got {sorted(e.canonical_code for e in result.events)}"
    )
    return matches[0]


def test_example_1_cycle_irregularity_is_present(transcriber, extractor) -> None:
    result = extract(transcriber, extractor, EXAMPLE_1)
    event = find(result, "cycle_irregularity")

    assert event.negated is False
    assert event.uncertain is False
    assert event.historical is False
    assert event.temporality == "current"

    # The duration is stated approximately. PRISM records the approximation
    # rather than inventing a precise onset date.
    assert event.duration_days is not None
    assert 300 <= event.duration_days <= 400, (
        f"'for about a year' should normalize to roughly 365 days, got {event.duration_days}"
    )

    # Evidence linking is mandatory for every speech-derived event.
    assert event.evidence.text
    assert event.evidence.text.lower() in EXAMPLE_1.lower()


def test_example_2_resolved_acne_is_historical_not_negated(transcriber, extractor) -> None:
    """The trap: a naive negation detector sees "do not have it" and marks the
    symptom absent, erasing a real past finding. The patient affirmed the
    symptom occurred *and* reported that it resolved — two different claims.
    """
    result = extract(transcriber, extractor, EXAMPLE_2)
    event = find(result, "acne")

    assert event.historical is True
    assert event.temporality in {"historical", "historical_resolved", "resolved"}
    assert event.negated is False, (
        "A resolved historical symptom must not be labelled negated — the patient "
        "affirmed it happened."
    )


def test_example_3_family_history_never_becomes_patient_diagnosis(transcriber, extractor) -> None:
    result = extract(transcriber, extractor, EXAMPLE_3)

    event = find(result, "family_history_pmos")
    assert event.attribution == "family_member"
    assert event.relation == "sister"
    assert event.negated is False

    # The critical assertion: nothing in this utterance may produce a
    # patient-level PMOS assertion of any kind.
    codes = {e.canonical_code for e in result.events}
    assert "pmos_binary" not in codes, (
        "A relative's diagnosis was written to a patient-level PMOS code. That "
        "would contaminate every evaluation using pmos_binary as ground truth."
    )
    patient_level = {
        e.canonical_code
        for e in result.events
        if e.attribution == "patient" and "pmos" in e.canonical_code.lower()
    }
    assert not patient_level, f"patient-attributed PMOS codes leaked: {patient_level}"


def test_blueprint_examples_are_pinned_in_the_corpus() -> None:
    """The three utterances must stay in the committed evaluation corpus."""
    corpus_path = (
        Path(__file__).resolve().parents[1] / "fixtures" / "speech_eval" / "scripted_corpus.yaml"
    )
    texts = {u["text"] for u in yaml.safe_load(corpus_path.read_text())["utterances"]}

    for expected in (EXAMPLE_1, EXAMPLE_2, EXAMPLE_3):
        assert expected in texts, f"Blueprint example dropped from the corpus: {expected!r}"
