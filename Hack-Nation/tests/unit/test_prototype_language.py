"""The banned-phrase guard must raise, and generated descriptions must pass it."""

from __future__ import annotations

import numpy as np
import pytest

from models.adapters.pcos.prototype_rules import (
    PROTOTYPE_PROFILES,
    match_prototype,
    name_clusters,
)
from models.phenotype.prototype_mapping import (
    BANNED_PHRASES,
    HEDGE_VERBS,
    ProhibitedLanguageError,
    assert_hedged_language,
    characterize_clusters,
    describe_cluster,
)
from tests.fixtures.synthetic_clusters import make_synthetic_cluster_frame

# ----------------------------------------------------------------- the guard


@pytest.mark.parametrize("phrase", BANNED_PHRASES)
def test_every_banned_phrase_raises(phrase):
    with pytest.raises(ProhibitedLanguageError):
        assert_hedged_language(f"This group is a {phrase} of the condition.")


def test_guard_is_case_insensitive():
    with pytest.raises(ProhibitedLanguageError):
        assert_hedged_language("This is a CLINICALLY VALIDATED SUBTYPE.")


def test_guard_error_names_the_offending_phrase():
    with pytest.raises(ProhibitedLanguageError, match="confirmed subtype"):
        assert_hedged_language("A confirmed subtype was found.")


def test_guard_passes_hedged_text_and_returns_it():
    text = "This participant resembles a data-driven group found in this cohort."
    assert assert_hedged_language(text) == text


@pytest.mark.parametrize("verb", HEDGE_VERBS)
def test_hedge_verbs_are_themselves_allowed(verb):
    assert_hedged_language(f"This participant {verb} the metabolic-leaning research profile.")


def test_banned_list_covers_the_required_phrases():
    for required in ("clinically validated subtype", "diagnosis", "confirmed subtype"):
        assert required in BANNED_PHRASES


# ------------------------------------------------------- generated descriptions


def test_generated_description_is_hedged_and_names_direction():
    text = describe_cluster("profile_0", {"bmi": 1.4, "shbg": -1.1, "acne": 0.05}, n_members=22)
    assert "bmi" in text and "shbg" in text
    assert "exploratory pattern, not a validated clinical category" in text
    assert_hedged_language(text)


def test_description_for_a_featureless_cluster_says_so():
    text = describe_cluster("profile_1", {"bmi": 0.01, "shbg": -0.02}, n_members=10)
    assert "no feature deviating" in text


def test_characterize_clusters_emits_guarded_descriptions_for_all_clusters():
    frame, truth = make_synthetic_cluster_frame(n_per_group=25, seed=0)
    codes = {name: i for i, name in enumerate(dict.fromkeys(truth))}
    labels = np.array([codes[t] for t in truth])
    chars = characterize_clusters(frame.to_numpy(), labels, list(frame.columns))
    assert len(chars) == 3
    for char in chars.values():
        assert char.n_members == 25
        assert char.elevated or char.reduced
        assert_hedged_language(char.description)


def test_characterize_clusters_rejects_a_feature_name_mismatch():
    with pytest.raises(ValueError, match="feature names"):
        characterize_clusters(np.zeros((4, 3)), np.array([0, 0, 1, 1]), ["a", "b"])


# --------------------------------------------------- post hoc prototype naming


def test_prototype_matching_labels_a_metabolic_looking_cluster():
    enrichment = {"bmi": 1.8, "fasting_insulin": 1.6, "homa_ir": 1.5, "shbg": -1.0}
    match = match_prototype("profile_0", enrichment)
    assert match.profile_name == "metabolic_leaning"
    assert match.similarity > 0.6
    assert match.similarity > match.all_similarities["androgenic_leaning"]
    assert_hedged_language(match.label)


def test_prototype_matching_labels_an_lh_amh_looking_cluster():
    enrichment = {
        "luteinizing_hormone": 2.0,
        "anti_mullerian_hormone": 2.0,
        "lh_fsh_ratio": 1.5,
        "bmi": -0.5,
    }
    assert match_prototype("profile_1", enrichment).profile_name == "lh_amh_leaning"


def test_unmatched_cluster_keeps_its_neutral_name():
    match = match_prototype("profile_2", {"unrelated_variable": 3.0})
    assert match.profile_name is None
    assert match.label == "profile_2"
    assert match.warnings


def test_name_clusters_returns_one_match_per_cluster():
    matches = name_clusters(
        {
            "profile_0": {"bmi": 1.5, "homa_ir": 1.4},
            "profile_1": {"total_testosterone": 2.0, "shbg": -1.5},
        }
    )
    assert set(matches) == {"profile_0", "profile_1"}
    assert matches["profile_1"].profile_name == "androgenic_leaning"


def test_prototype_profiles_carry_a_rationale_and_literature_note():
    expected = {"metabolic_leaning", "lh_amh_leaning", "androgenic_leaning", "lean_reproductive"}
    assert set(PROTOTYPE_PROFILES) == expected
    for profile in PROTOTYPE_PROFILES.values():
        assert profile.rationale and profile.literature_note
        assert profile.enrichment
        assert_hedged_language(profile.rationale)
        assert_hedged_language(profile.literature_note)
