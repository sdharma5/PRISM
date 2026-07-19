"""Extraction semantics: negation, temporality, attribution, evidence spans.

These are the tests that guard the claims PRISM makes about speech. If any of
them regress, the pipeline is manufacturing findings.
"""

from __future__ import annotations

import pytest

from ingestion.speech.extraction import (
    EvidenceSpan,
    ExtractedSymptomEvent,
    ExtractionResult,
    LlmExtractor,
    RuleBasedExtractor,
)
from ingestion.speech.transcription import ScriptedTranscriptionAdapter
from ingestion.speech.validation import span_is_supported, validate_extractions


@pytest.fixture(scope="module")
def extractor() -> RuleBasedExtractor:
    return RuleBasedExtractor()


@pytest.fixture(scope="module")
def transcriber() -> ScriptedTranscriptionAdapter:
    return ScriptedTranscriptionAdapter()


def extract(transcriber, extractor, text: str, speaker_role: str = "patient") -> ExtractionResult:
    transcript = transcriber.transcribe(
        [{"speaker_role": speaker_role, "text": text}], recording_id="REC-TEST"
    )
    return extractor.extract(transcript, patient_id="P-TEST")


def codes(result: ExtractionResult) -> set[str]:
    return {e.canonical_code for e in result.events}


def find(result: ExtractionResult, code: str) -> ExtractedSymptomEvent:
    matches = [e for e in result.events if e.canonical_code == code]
    assert matches, f"expected an event for '{code}', got {sorted(codes(result))}"
    return matches[0]


# -- Negation ---------------------------------------------------------------


def test_negated_symptom_is_marked_negated(transcriber, extractor):
    result = extract(transcriber, extractor, "I don't have acne at all.")
    event = find(result, "acne")
    assert event.negated is True
    assert event.value is False


def test_negation_scope_stops_at_terminator(transcriber, extractor):
    """ "but" ends the negation scope: the cramps are asserted, not denied."""
    result = extract(transcriber, extractor, "I don't get mood swings, but I do get cramps.")
    assert find(result, "mood_change").negated is True
    assert find(result, "pelvic_pain").negated is False


def test_negation_inside_a_phrase_is_detected(transcriber, extractor):
    """ "periods are not regular" must still match the 'periods are regular' phrase."""
    result = extract(transcriber, extractor, "My periods are not regular.")
    assert find(result, "cycle_regularity").negated is True


def test_clinician_denial_negates(transcriber, extractor):
    result = extract(transcriber, extractor, "She denies pelvic pain.", speaker_role="clinician")
    assert find(result, "pelvic_pain").negated is True


# -- Temporality ------------------------------------------------------------


def test_historical_symptom_is_marked_historical(transcriber, extractor):
    result = extract(transcriber, extractor, "I used to get acne when I was a teenager.")
    event = find(result, "acne")
    assert event.historical is True
    assert event.temporality == "historical"


def test_in_high_school_is_historical(transcriber, extractor):
    result = extract(transcriber, extractor, "I had irregular periods in high school.")
    assert find(result, "cycle_irregularity").historical is True


def test_since_keeps_a_longstanding_symptom_current(transcriber, extractor):
    """ "since high school" is onset, not resolution: the acne is still present."""
    result = extract(
        transcriber, extractor, "I've had acne since high school and it's still there."
    )
    event = find(result, "acne")
    assert event.historical is False
    assert event.temporality == "current"


def test_onset_phrase_is_recorded_without_fabricating_a_date(transcriber, extractor):
    result = extract(transcriber, extractor, "I've had irregular periods since high school.")
    assert find(result, "cycle_irregularity").onset == "since high school"


# -- Uncertainty ------------------------------------------------------------


def test_hedge_marks_uncertainty(transcriber, extractor):
    result = extract(transcriber, extractor, "I think I have hair loss, but I'm not certain.")
    assert find(result, "androgenic_alopecia").uncertain is True


def test_not_sure_is_uncertainty_not_negation(transcriber, extractor):
    """Pseudo-negation: doubt about a finding is not a denial of it."""
    result = extract(transcriber, extractor, "I'm not sure I have facial hair, it might be normal.")
    event = find(result, "hair_growth_face")
    assert event.uncertain is True
    assert event.negated is False


def test_uncertainty_lowers_extraction_confidence(transcriber, extractor):
    certain = extract(transcriber, extractor, "I get acne along my jawline.")
    hedged = extract(transcriber, extractor, "Maybe acne, it's hard to say.")
    assert find(hedged, "acne").extraction_confidence < find(certain, "acne").extraction_confidence


# -- Family history: the critical invariant ---------------------------------


def test_relatives_diagnosis_becomes_family_history(transcriber, extractor):
    result = extract(transcriber, extractor, "My mom has PCOS but I don't have diabetes.")
    event = find(result, "family_history_pcos")
    assert event.attribution == "family_member"
    assert event.relation == "mother"


def test_family_history_never_becomes_a_patient_diagnosis(transcriber, extractor):
    """The single most damaging failure mode in this pipeline."""
    result = extract(transcriber, extractor, "My mom has PCOS but I don't have diabetes.")
    assert "pcos_binary" not in codes(result)
    for event in result.events:
        if event.canonical_code.startswith("family_history_"):
            assert event.attribution == "family_member"
        else:
            assert event.attribution != "family_member"


def test_patient_symptom_is_not_captured_by_a_nearby_relative(transcriber, extractor):
    result = extract(transcriber, extractor, "My mother has PCOS but I have acne myself.")
    assert find(result, "family_history_pcos").attribution == "family_member"
    assert find(result, "acne").attribution == "patient"


def test_family_history_code_rejects_patient_attribution():
    """The model itself refuses the misattribution, not just the extractor."""
    with pytest.raises(ValueError, match="family_member"):
        ExtractedSymptomEvent(
            extraction_id="X1",
            recording_id="R1",
            patient_id="P1",
            canonical_code="family_history_pcos",
            variable_name="Family history of PCOS",
            surface_form="pcos",
            attribution="patient",
            evidence=EvidenceSpan(text="pcos", char_start=0, char_end=4),
            extraction_confidence=0.9,
        )


def test_patient_variable_rejects_family_attribution():
    with pytest.raises(ValueError, match="must not be recorded"):
        ExtractedSymptomEvent(
            extraction_id="X2",
            recording_id="R1",
            patient_id="P1",
            canonical_code="acne",
            variable_name="Acne",
            surface_form="acne",
            attribution="family_member",
            evidence=EvidenceSpan(text="acne", char_start=0, char_end=4),
            extraction_confidence=0.9,
        )


def test_self_reported_pcos_is_unmapped_not_a_label(transcriber, extractor):
    """A self-reported diagnosis must not be written into the dataset label."""
    result = extract(transcriber, extractor, "I have PCOS.")
    assert "pcos_binary" not in codes(result)
    assert any("pcos" in m.surface_form.lower() for m in result.unmapped)


# -- Questions and speaker attribution --------------------------------------


def test_clinician_question_asserts_nothing(transcriber, extractor):
    result = extract(
        transcriber, extractor, "Have you noticed any excess hair growth?", speaker_role="clinician"
    )
    assert result.events == []
    assert result.suppressed_questions >= 1


def test_clinician_third_person_attributes_to_patient(transcriber, extractor):
    result = extract(
        transcriber,
        extractor,
        "She reports her periods are irregular and she denies acne.",
        speaker_role="clinician",
    )
    irregular = find(result, "cycle_irregularity")
    assert irregular.attribution == "patient"
    assert irregular.speaker_role == "clinician"
    assert find(result, "acne").negated is True


# -- Medications ------------------------------------------------------------


def test_medication_stop_is_not_a_current_medication(transcriber, extractor):
    result = extract(transcriber, extractor, "I stopped metformin about three months ago.")
    event = find(result, "medication_current")
    assert event.value == "metformin"
    assert event.medication_action == "stop"
    assert event.historical is True


def test_medication_start_is_current(transcriber, extractor):
    result = extract(transcriber, extractor, "I started metformin two months ago.")
    event = find(result, "medication_current")
    assert event.medication_action == "start"
    assert event.historical is False


# -- Numeric normalization --------------------------------------------------


def test_cycle_length_is_normalized(transcriber, extractor):
    result = extract(transcriber, extractor, "My cycles come every 45 days.")
    assert find(result, "cycle_length").value == 45.0


def test_menstrual_frequency_is_normalized(transcriber, extractor):
    result = extract(transcriber, extractor, "I only get about three periods a year.")
    assert find(result, "menstrual_frequency_per_year").value == 3.0


def test_frequency_requires_a_menstrual_context(transcriber, extractor):
    """ "three coffees a day" must never become a menstrual frequency."""
    result = extract(transcriber, extractor, "I drink three coffees a day.")
    assert "menstrual_frequency_per_year" not in codes(result)


def test_duration_is_normalized_to_days(transcriber, extractor):
    result = extract(transcriber, extractor, "I've had fatigue for about a year.")
    assert find(result, "fatigue").duration_days == pytest.approx(365.25)


# -- Evidence spans ---------------------------------------------------------


def test_every_event_carries_an_evidence_span(transcriber, extractor):
    transcript = transcriber.transcribe(
        [{"speaker_role": "patient", "text": "I have acne and my periods are irregular."}],
        recording_id="REC-SPAN",
    )
    result = extractor.extract(transcript, patient_id="P-TEST")
    assert result.events
    for event in result.events:
        span = event.evidence
        assert span.char_end > span.char_start
        assert transcript.text[span.char_start : span.char_end] == span.text
        assert span.start_seconds is not None
        assert span.end_seconds is not None
        assert span.end_seconds >= span.start_seconds


def test_spans_are_verified_against_the_transcript(transcriber, extractor):
    transcript = transcriber.transcribe(
        [{"speaker_role": "patient", "text": "I have acne."}], recording_id="REC-VERIFY"
    )
    result = extractor.extract(transcript, patient_id="P-TEST")
    event = result.events[0]
    assert span_is_supported(transcript, event) is True

    tampered = event.model_copy(
        update={"evidence": event.evidence.model_copy(update={"text": "hirsutism"})}
    )
    assert span_is_supported(transcript, tampered) is False

    report = validate_extractions(transcript, [tampered])
    assert tampered.extraction_id in report.unsupported_extraction_ids
    assert report.ok is False


def test_extraction_output_is_json_serializable(transcriber, extractor):
    result = extract(transcriber, extractor, "My mom has PCOS and I have acne.")
    payload = result.model_dump(mode="json")
    assert isinstance(payload["events"], list)
    import json

    json.dumps(payload)  # must not raise


# -- LLM stub ---------------------------------------------------------------


def test_llm_extractor_refuses_without_a_client(transcriber):
    """The stub must never silently attempt a network call."""
    transcript = transcriber.transcribe(
        [{"speaker_role": "patient", "text": "I have acne."}], recording_id="REC-LLM"
    )
    with pytest.raises(NotImplementedError, match="no configured client"):
        LlmExtractor().extract(transcript, patient_id="P-TEST")


# -- Numeric assertions carrying no lexical cue -----------------------------
#
# Blueprint example 1 exposed this whole class: a patient states irregularity as
# a cycle-length range and never says "irregular". These tests pin the numeric
# path, including the cases that must stay silent.


def test_wide_cycle_range_asserts_irregularity(transcriber, extractor):
    result = extract(transcriber, extractor, "My periods come 45 to 70 days apart.")
    event = find(result, "cycle_irregularity")
    assert event.negated is False
    assert event.uncertain is False
    assert event.temporality == "current"


@pytest.mark.parametrize(
    "text",
    [
        "My cycles are anywhere from 30 to 60 days.",
        "My periods have been between 45 and 70 days apart.",
        "My cycles come every 25 to 50 days.",
    ],
)
def test_range_phrasings_all_reach_irregularity(transcriber, extractor, text):
    assert "cycle_irregularity" in codes(extract(transcriber, extractor, text))


def test_normal_tight_range_asserts_nothing_about_regularity(transcriber, extractor):
    """A normal range must not be flagged irregular — nor cleared as regular."""
    result = extract(transcriber, extractor, "My cycles range from 28 to 32 days.")
    assert "cycle_irregularity" not in codes(result)
    assert "cycle_regularity" not in codes(result)
    assert find(result, "cycle_length").value == pytest.approx(30.0)


def test_a_single_long_cycle_is_not_irregularity(transcriber, extractor):
    """A reliably 45-day cycle is oligomenorrhoea, a different finding."""
    result = extract(transcriber, extractor, "My cycles come every 45 days.")
    assert "cycle_irregularity" not in codes(result)
    assert find(result, "cycle_length").value == pytest.approx(45.0)


def test_range_midpoint_is_marked_uncertain_and_keeps_its_endpoints(transcriber, extractor):
    """The midpoint is synthesized, so it must say so and never lose the range."""
    result = extract(transcriber, extractor, "My periods come 45 to 70 days apart.")
    event = find(result, "cycle_length")
    assert event.value == pytest.approx(57.5)
    assert event.uncertain is True
    assert event.value_range == [45.0, 70.0]


def test_irregularity_from_a_range_is_not_uncertain(transcriber, extractor):
    """The patient stated the variability as fact; only the midpoint is inferred."""
    result = extract(transcriber, extractor, "My cycles are anywhere from 30 to 60 days.")
    assert find(result, "cycle_irregularity").uncertain is False
    assert find(result, "cycle_length").uncertain is True


def test_duration_attaches_to_a_numeric_cycle_event(transcriber, extractor):
    result = extract(
        transcriber,
        extractor,
        "My periods have been between 45 and 70 days apart for about a year.",
    )
    assert find(result, "cycle_irregularity").duration_days == pytest.approx(365.25)


def test_thresholds_come_from_the_lexicon_not_from_code():
    """The clinical cutoffs must be configuration a clinician can review."""
    tuned = RuleBasedExtractor()
    assert tuned.normal_cycle_min_days == 24
    assert tuned.normal_cycle_max_days == 38
    assert tuned.max_variability_days == 9
    assert tuned._range_asserts_irregularity(45, 70) is True
    assert tuned._range_asserts_irregularity(28, 32) is False
    assert tuned._range_asserts_irregularity(26, 36) is True  # 10-day spread > 9
    assert tuned._range_asserts_irregularity(30, 38) is False  # 8-day spread, both ends in band
    assert tuned._range_asserts_irregularity(30, 42) is True  # upper end past 38
    assert tuned._range_asserts_irregularity(20, 28) is True  # lower end below 24
    assert tuned._range_asserts_irregularity(30, 34) is False
    # Order-insensitive: a speaker may state the range either way round.
    assert tuned._range_asserts_irregularity(70, 45) is True


def test_behavioural_skipping_is_irregularity(transcriber, extractor):
    assert "cycle_irregularity" in codes(
        extract(transcriber, extractor, "I skip months at a time.")
    )


def test_going_months_without_a_period_is_amenorrhea(transcriber, extractor):
    assert "amenorrhea" in codes(extract(transcriber, extractor, "I go months without a period."))


def test_numeric_weight_gain_without_the_word_weight(transcriber, extractor):
    result = extract(transcriber, extractor, "I've put on 20 pounds since January.")
    event = find(result, "weight_gain")
    assert event.value is True
    assert event.negated is False
    assert "20 pounds" in event.evidence.text


def test_numeric_weight_change_does_not_emit_an_absolute_weight(transcriber, extractor):
    """A delta is not a measurement; `weight` must not be fabricated from it."""
    result = extract(transcriber, extractor, "I've gained about 15 kg in the last year.")
    assert "weight_gain" in codes(result)
    assert "weight" not in codes(result)


def test_grooming_behaviour_implies_facial_hair(transcriber, extractor):
    assert "hair_growth_face" in codes(
        extract(transcriber, extractor, "I have to shave my chin every day.")
    )


def test_explicit_fg_score_is_extracted(transcriber, extractor):
    result = extract(transcriber, extractor, "Her Ferriman-Gallwey score was 12.", "clinician")
    assert find(result, "ferriman_gallwey_score").value == pytest.approx(12.0)


def test_fg_score_is_never_inferred_from_a_hair_description(transcriber, extractor):
    """A score is a performed examination, not something to synthesize."""
    result = extract(transcriber, extractor, "I have a lot of facial hair on my chin.")
    assert "ferriman_gallwey_score" not in codes(result)


def test_numeric_extraction_still_requires_a_menstrual_context(transcriber, extractor):
    result = extract(transcriber, extractor, "I take 30 to 60 minutes to fall asleep.")
    assert "cycle_length" not in codes(result)
    assert "cycle_irregularity" not in codes(result)


def test_lexical_and_numeric_assertions_are_not_double_counted(transcriber, extractor):
    """One breath, one finding — even when stated twice two different ways."""
    result = extract(
        transcriber, extractor, "My periods are all over the place, between 45 and 70 days apart."
    )
    irregularity = [e for e in result.events if e.canonical_code == "cycle_irregularity"]
    assert len(irregularity) == 1


def test_numeric_events_are_grounded_on_their_own_phrase(transcriber, extractor):
    """The span must point at the number, not at the whole sentence."""
    transcript = transcriber.transcribe(
        [
            {
                "speaker_role": "patient",
                "text": "My periods come 45 to 70 days apart and I have acne.",
            }
        ],
        recording_id="REC-NUM",
    )
    result = extractor.extract(transcript, patient_id="P-TEST")
    event = next(e for e in result.events if e.canonical_code == "cycle_length")
    assert (
        transcript.text[event.evidence.char_start : event.evidence.char_end] == event.evidence.text
    )
    assert "acne" not in event.evidence.text
    assert "45" in event.evidence.text
