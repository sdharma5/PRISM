"""Speech metric correctness, including the hand-rolled Levenshtein/WER."""

from __future__ import annotations

import pytest

from evaluation.speech import (
    corpus_word_error_rate,
    evaluate_corpus,
    levenshtein,
    match_events,
    medication_event_f1,
    negation_f1,
    normalize_for_wer,
    speaker_attribution_accuracy,
    symptom_extraction_prf,
    temporality_f1,
    uncertainty_f1,
    unsupported_event_rate,
    user_correction_rate,
    word_error_rate,
)


def event(code, **kwargs):
    base = {
        "canonical_code": code,
        "attribution": "patient",
        "negated": False,
        "historical": False,
        "uncertain": False,
        "speaker_role": "patient",
        "value": True,
        "medication_action": None,
    }
    base.update(kwargs)
    return base


# -- Levenshtein / WER ------------------------------------------------------


@pytest.mark.parametrize(
    ("a", "b", "distance"),
    [
        ([], [], 0),
        (["a"], [], 1),
        ([], ["a"], 1),
        (["a", "b"], ["a", "b"], 0),
        (["a", "b"], ["a", "c"], 1),  # substitution
        (["a", "b"], ["a"], 1),  # deletion
        (["a"], ["a", "b"], 1),  # insertion
        (["k", "i", "t", "t", "e", "n"], ["s", "i", "t", "t", "i", "n", "g"], 3),
    ],
)
def test_levenshtein(a, b, distance):
    assert levenshtein(a, b) == distance


def test_levenshtein_is_symmetric():
    a, b = ["one", "two", "three"], ["one", "three"]
    assert levenshtein(a, b) == levenshtein(b, a)


def test_wer_identical_is_zero():
    assert word_error_rate("my periods are irregular", "my periods are irregular") == 0.0


def test_wer_counts_one_substitution():
    assert word_error_rate("my periods are irregular", "my periods are regular") == pytest.approx(
        0.25
    )


def test_wer_ignores_case_and_punctuation():
    assert word_error_rate("I don't have acne.", "i dont have acne") == 0.0


def test_wer_empty_reference():
    assert word_error_rate("", "") == 0.0
    assert word_error_rate("", "something") == 1.0


def test_normalize_for_wer_strips_punctuation():
    assert normalize_for_wer("I've had acne, since 2019!") == [
        "ive",
        "had",
        "acne",
        "since",
        "2019",
    ]


def test_corpus_wer_is_length_weighted():
    pairs = [
        ("one two three four", "one two three four"),  # 4 words, 0 errors
        ("five", "six"),  # 1 word, 1 error
    ]
    assert corpus_word_error_rate(pairs) == pytest.approx(0.2)


def test_corpus_wer_of_nothing_is_zero():
    assert corpus_word_error_rate([]) == 0.0


# -- Matching ---------------------------------------------------------------


def test_matching_pairs_on_code_and_attribution():
    gold = [event("acne"), event("family_history_pmos", attribution="family_member")]
    pred = [event("family_history_pmos", attribution="family_member"), event("acne")]
    matched, spurious, missed = match_events(gold, pred)
    assert len(matched) == 2
    assert spurious == [] and missed == []


def test_family_attribution_mismatch_is_not_a_match():
    """acne(patient) and acne(family) are different assertions."""
    gold = [event("acne", attribution="patient")]
    pred = [event("acne", attribution="family_member")]
    matched, spurious, missed = match_events(gold, pred)
    assert matched == []
    assert len(spurious) == 1 and len(missed) == 1


def test_extraction_prf_perfect():
    gold = pred = [event("acne"), event("fatigue")]
    score = symptom_extraction_prf(gold, pred)
    assert score.precision == 1.0 and score.recall == 1.0 and score.f1 == 1.0
    assert score.support == 2


def test_extraction_prf_counts_misses_and_spurious():
    gold = [event("acne"), event("fatigue")]
    pred = [event("acne"), event("hirsutism")]
    score = symptom_extraction_prf(gold, pred)
    assert score.true_positives == 1
    assert score.false_positives == 1
    assert score.false_negatives == 1
    assert score.f1 == pytest.approx(0.5)


def test_empty_inputs_do_not_divide_by_zero():
    score = symptom_extraction_prf([], [])
    assert score.f1 == 0.0 and score.support == 0


# -- Attribute metrics ------------------------------------------------------


def test_negation_f1_rewards_correct_negation():
    gold = [event("acne", negated=True)]
    pred = [event("acne", negated=True)]
    assert negation_f1(gold, pred).f1 == 1.0


def test_negation_f1_punishes_a_missed_negation():
    """Reading "I don't have acne" as present acne is the failure this catches."""
    gold = [event("acne", negated=True)]
    pred = [event("acne", negated=False)]
    score = negation_f1(gold, pred)
    assert score.f1 == 0.0
    assert score.false_negatives == 1


def test_negation_f1_punishes_a_spurious_negation():
    gold = [event("acne", negated=False)]
    pred = [event("acne", negated=True)]
    assert negation_f1(gold, pred).false_positives == 1


def test_an_undetected_negated_event_is_a_negation_false_negative():
    gold = [event("acne", negated=True)]
    assert negation_f1(gold, []).false_negatives == 1


def test_temporality_f1():
    gold = [event("acne", historical=True), event("fatigue")]
    pred = [event("acne", historical=False), event("fatigue")]
    score = temporality_f1(gold, pred)
    assert score.false_negatives == 1
    assert score.f1 == 0.0


def test_uncertainty_f1():
    gold = [event("acne", uncertain=True)]
    pred = [event("acne", uncertain=True)]
    assert uncertainty_f1(gold, pred).f1 == 1.0


# -- Medications ------------------------------------------------------------


def test_medication_f1_matches_drug_and_action():
    gold = [event("medication_current", value="metformin", medication_action="stop")]
    pred = [event("medication_current", value="metformin", medication_action="stop")]
    assert medication_event_f1(gold, pred).f1 == 1.0


def test_start_and_stop_are_different_facts():
    gold = [event("medication_current", value="metformin", medication_action="stop")]
    pred = [event("medication_current", value="metformin", medication_action="start")]
    score = medication_event_f1(gold, pred)
    assert score.f1 == 0.0
    assert score.false_positives == 1 and score.false_negatives == 1


def test_medication_f1_ignores_non_medication_events():
    gold = [event("acne")]
    pred = [event("acne")]
    assert medication_event_f1(gold, pred).support == 0


# -- Speaker, unsupported, corrections --------------------------------------


def test_speaker_attribution_accuracy():
    gold = [event("acne", speaker_role="clinician"), event("fatigue")]
    pred = [event("acne", speaker_role="patient"), event("fatigue")]
    assert speaker_attribution_accuracy(gold, pred) == pytest.approx(0.5)


def test_speaker_accuracy_with_no_matches_is_zero():
    assert speaker_attribution_accuracy([], []) == 0.0


def test_unsupported_event_rate():
    assert unsupported_event_rate([event("acne")] * 9, 1) == pytest.approx(0.1)
    assert unsupported_event_rate([], 0) == 0.0


def test_user_correction_rate():
    assert user_correction_rate(4, 1) == pytest.approx(0.25)
    assert user_correction_rate(0, 0) == 0.0


# -- Report assembly --------------------------------------------------------


def test_evaluate_corpus_assembles_a_serializable_report():
    gold = [event("acne", negated=True), event("fatigue")]
    pred = [event("acne", negated=True), event("fatigue")]
    report = evaluate_corpus(
        gold,
        pred,
        corpus_id="test_corpus",
        n_utterances=2,
        transcript_pairs=[("I have acne", "I have acne")],
        n_unsupported=0,
        n_reviewed=2,
        n_corrected=0,
        categories={"negated": ([gold[0]], [pred[0]])},
    )
    assert report.symptom_extraction.f1 == 1.0
    assert report.negation.f1 == 1.0
    assert report.word_error_rate == 0.0
    assert "negated" in report.per_category

    import json

    json.dumps(report.model_dump(mode="json"))  # must not raise
