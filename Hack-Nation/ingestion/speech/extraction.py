"""Symptom extraction from transcripts.

Why the default extractor is rule-based: an extraction that a clinician cannot
audit is an extraction they cannot confirm, and nothing enters a PRISM snapshot
without confirmation. The rule-based extractor is deterministic, offline, and
every decision it makes (negated / historical / uncertain / whose symptom it is)
traces to a named cue at a known character offset.

The hardest failure mode in this domain is not a missed symptom — it is a
*misattributed* one. "My mother has PMOS" becoming the patient's diagnosis
corrupts every downstream label. So attribution is resolved explicitly, and a
concept with no ``family_code`` can never emit a family event, while a concept
with no patient-level ``code`` can never emit a patient event.

Not every clinical assertion carries a lexical cue. Patients routinely express
irregularity, weight change and hirsutism *numerically* — "between 45 and 70
days apart", "I've put on 20 pounds" — with none of the words a lexicon matches
on. A phrase-only extractor silently returns nothing for those, and an aggregate
F1 computed on a lexicon-shaped corpus will never reveal it. The numeric
extractors below exist specifically to cover that blind spot, and their clinical
cutoffs live in ``lexicon.yaml`` rather than in this file.

KNOWN COVERAGE LIMITS — deliberately not handled, recorded here so that silence
is a documented decision rather than an unnoticed gap:

* **Inferential frequency.** "I only need tampons a few times a year" implies
  oligomenorrhoea but never states a cycle fact. Extracting it would require
  the pipeline to reason about a proxy behaviour it cannot verify.
* **Anchored dates.** "since January", "since last summer" are recorded verbatim
  as ``onset`` and never resolved to a date: resolution needs the recording date
  plus a hemisphere/locale assumption, and a fabricated date is worse than an
  approximate phrase.
* **Cross-sentence coreference.** "My sister has PMOS. I think I have it too."
  The second sentence's "it" is not resolved, so it becomes an unmapped mention
  rather than a guessed diagnosis.
* **Absolute self-reported anthropometrics.** "I weigh 200 pounds" is left alone:
  it needs registry unit conversion plus a measurement-context check (measured
  when? clothed?), and a self-reported absolute is not equivalent to a measured
  ``weight`` observation. Weight *change* is handled, because the binary
  ``weight_gain`` variable is exactly what a change statement asserts.
* **Comparative and prosodic severity.** "worse than my sister's", or severity
  conveyed by tone, are not recoverable from a transcript.
* **Non-English and code-switched speech.** The lexicon is English-only.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from ingestion.speech.transcription import Transcript
from registry.loader import load_variable_registry

LEXICON_PATH = Path(__file__).resolve().parent / "lexicon.yaml"
EXTRACTOR_VERSION = "rule_based/1.0.0"

Attribution = Literal["patient", "family_member", "other", "unknown"]
Temporality = Literal["current", "historical", "unknown"]
SpeakerRole = Literal["patient", "clinician", "unknown"]
MedicationAction = Literal["start", "stop", "continue", "change"]
Severity = Literal["mild", "moderate", "severe"]

_NUMBER_WORDS: dict[str, float] = {
    "a": 1,
    "an": 1,
    "one": 1,
    "once": 1,
    "two": 2,
    "twice": 2,
    "couple": 2,
    "three": 3,
    "few": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}

_DURATION_DAYS: dict[str, float] = {
    "day": 1.0,
    "days": 1.0,
    "week": 7.0,
    "weeks": 7.0,
    "month": 30.4375,
    "months": 30.4375,
    "year": 365.25,
    "years": 365.25,
}

_THIRD_PERSON = ("she", "her", "hers", "patient")


class EvidenceSpan(BaseModel):
    """Where in the transcript — and in the audio — an assertion came from.

    Both coordinate systems are kept because they answer different questions:
    character offsets let a reviewer highlight the text, audio seconds let them
    press play and hear it.
    """

    text: str
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    start_seconds: float | None = None
    end_seconds: float | None = None
    segment_id: str | None = None

    @model_validator(mode="after")
    def _check(self) -> EvidenceSpan:
        if self.char_end <= self.char_start:
            raise ValueError("evidence span must cover at least one character.")
        return self


class ExtractedSymptomEvent(BaseModel):
    """One candidate clinical assertion recovered from speech.

    This is *not* a :class:`~schemas.event.HormonalHealthEvent`. It is a proposal
    awaiting human confirmation; the conversion happens in ``confirmation.py``
    and only for confirmed items. Keeping the two types distinct makes it
    structurally impossible for an unreviewed extraction to be mistaken for a
    confirmed observation.
    """

    extraction_id: str
    recording_id: str
    patient_id: str

    canonical_code: str
    variable_name: str
    value: Any = True
    unit: str | None = None

    surface_form: str
    category: str = "symptom"

    negated: bool = False
    historical: bool = False
    uncertain: bool = False
    temporality: Temporality = "current"

    attribution: Attribution = "patient"
    relation: str | None = None
    speaker_role: SpeakerRole = "unknown"

    onset: str | None = None
    duration_days: float | None = None
    frequency_per_year: float | None = None
    severity: Severity | None = None
    medication_action: MedicationAction | None = None

    #: Endpoints of a spoken range, when ``value`` was derived from one. Kept so
    #: that "between 45 and 70 days" is never collapsed to its midpoint without
    #: the original span surviving alongside it.
    value_range: list[float] | None = None

    evidence: EvidenceSpan
    extraction_confidence: float = Field(ge=0.0, le=1.0)
    extractor_version: str = EXTRACTOR_VERSION
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> ExtractedSymptomEvent:
        # A family-history code may only ever carry family attribution, and a
        # non-family code may never carry it. This is the invariant that keeps a
        # relative's diagnosis out of the patient's record.
        is_family_code = self.canonical_code.startswith("family_history_")
        if is_family_code and self.attribution != "family_member":
            raise ValueError(
                f"{self.canonical_code} requires attribution='family_member', "
                f"got '{self.attribution}'."
            )
        if not is_family_code and self.attribution == "family_member":
            raise ValueError(
                f"{self.canonical_code}: a family-attributed mention must not be recorded "
                "as a patient-level variable."
            )
        return self


class UnmappedMention(BaseModel):
    """A recognized concept with no canonical variable to put it in.

    Recorded rather than dropped: silent drops are how "the model never saw the
    patient's own PMOS report" becomes an invisible bug.
    """

    recording_id: str
    surface_form: str
    reason: str
    evidence: EvidenceSpan
    attribution: Attribution = "patient"


class ExtractionResult(BaseModel):
    """Everything one extraction pass produced, including what it refused."""

    recording_id: str
    patient_id: str
    events: list[ExtractedSymptomEvent] = Field(default_factory=list)
    unmapped: list[UnmappedMention] = Field(default_factory=list)
    suppressed_questions: int = 0
    extractor_version: str = EXTRACTOR_VERSION
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Lexicon loading
# ---------------------------------------------------------------------------


@lru_cache(maxsize=4)
def load_lexicon(path: Path | None = None) -> dict[str, Any]:
    """Load and cache the YAML lexicon."""
    with (path or LEXICON_PATH).open() as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError("lexicon.yaml must contain a mapping at the top level.")
    return data


def _normalize_token(raw: str) -> str:
    """Lowercase, drop apostrophes and edge punctuation: ``Don't,`` -> ``dont``."""
    lowered = raw.lower().replace("'", "").replace("’", "")
    return re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", lowered)


def _phrase_key(phrase: str) -> tuple[str, ...]:
    return tuple(t for t in (_normalize_token(w) for w in phrase.split()) if t)


@dataclass(frozen=True)
class Token:
    """One transcript token with its position in text, audio and discourse."""

    text: str
    raw: str
    char_start: int
    char_end: int
    sentence_id: int
    segment_id: str
    speaker_role: str


def tokenize_transcript(transcript: Transcript) -> list[Token]:
    """Tokenize a transcript, tracking sentence and speaker boundaries.

    Sentences break on ``.?!;`` and always break at a segment boundary, because
    two speakers never share a negation scope.
    """
    tokens: list[Token] = []
    sentence_id = 0
    for segment in transcript.segments:
        for match in re.finditer(r"\S+", segment.text):
            raw = match.group(0)
            normalized = _normalize_token(raw)
            if normalized:
                tokens.append(
                    Token(
                        text=normalized,
                        raw=raw,
                        char_start=segment.char_start + match.start(),
                        char_end=segment.char_start + match.end(),
                        sentence_id=sentence_id,
                        segment_id=segment.segment_id,
                        speaker_role=segment.speaker_role,
                    )
                )
            if raw and raw[-1] in ".?!;":
                sentence_id += 1
        sentence_id += 1
    return tokens


class _Matcher:
    """Longest-match n-gram phrase matcher over normalized tokens.

    ``skippable`` holds single tokens that may sit *inside* a lexicon phrase
    without breaking the match — in practice, negation words. "my periods are
    not regular" has to match the ``periods are regular`` phrase, because the
    alternative is enumerating a negated variant of every phrase in the lexicon
    and inevitably missing some. When a skip happens the match reports it, so
    the caller can negate the assertion rather than lose the negation.
    """

    def __init__(
        self, entries: dict[tuple[str, ...], Any], skippable: frozenset[str] = frozenset()
    ) -> None:
        self.entries = entries
        self.skippable = skippable
        self.max_len = max((len(k) for k in entries), default=0)

    def find_all(self, tokens: list[Token]) -> list[tuple[int, int, Any, bool]]:
        """Return ``(start, end_exclusive, payload, skipped_negation)`` matches."""
        results: list[tuple[int, int, Any, bool]] = []
        index = 0
        max_window = min(self.max_len + 1, len(tokens)) if self.max_len else 0
        while index < len(tokens):
            matched = False
            for length in range(min(max_window, len(tokens) - index), 0, -1):
                window = tokens[index : index + length]
                if len({t.sentence_id for t in window}) > 1:
                    continue
                key = tuple(t.text for t in window)
                payload = self.entries.get(key)
                if payload is not None:
                    results.append((index, index + length, payload, False))
                    index += length
                    matched = True
                    break
                hit = self._match_with_skip(window)
                if hit is not None:
                    results.append((index, index + length, hit, True))
                    index += length
                    matched = True
                    break
            if not matched:
                index += 1
        return results

    def _match_with_skip(self, window: list[Token]) -> Any | None:
        """Try the window again with one interior skippable token removed."""
        if len(window) < 3:
            return None
        for position in range(1, len(window) - 1):
            if window[position].text not in self.skippable:
                continue
            key = tuple(t.text for i, t in enumerate(window) if i != position)
            payload = self.entries.get(key)
            if payload is not None:
                return payload
        return None


def _cue_hit(
    tokens: list[Token], lo: int, hi: int, cues: list[tuple[str, ...]]
) -> tuple[int, tuple[str, ...]] | None:
    """Find the *latest* cue occurrence inside ``tokens[lo:hi]``."""
    best: tuple[int, tuple[str, ...]] | None = None
    for cue in cues:
        span = len(cue)
        for start in range(lo, hi - span + 1):
            matches = tuple(t.text for t in tokens[start : start + span]) == cue
            if matches and (best is None or start > best[0]):
                best = (start, cue)
    return best


class ExtractionAdapter(ABC):
    """Contract for any symptom extractor, rule-based or model-based."""

    name: str = "abstract"
    version: str = "0.0.0"

    @abstractmethod
    def extract(
        self, transcript: Transcript, *, patient_id: str, **kwargs: Any
    ) -> ExtractionResult:
        """Extract candidate assertions, each carrying an evidence span."""


class RuleBasedExtractor(ExtractionAdapter):
    """Deterministic, lexicon-driven extractor. The offline default."""

    name = "rule_based"
    version = "1.0.0"

    def __init__(self, lexicon_path: Path | None = None) -> None:
        self.lexicon = load_lexicon(lexicon_path)
        cues = self.lexicon["cues"]
        cfg = self.lexicon["config"]

        self.negation_cues = [_phrase_key(c) for c in cues["negation"]]
        self.pseudo_negation = [_phrase_key(c) for c in cues["pseudo_negation"]]
        self.terminators = {_phrase_key(c) for c in cues["terminators"]}
        self.uncertainty_cues = [_phrase_key(c) for c in cues["uncertainty"]]
        self.historical_cues = [_phrase_key(c) for c in cues["historical"]]
        self.still_present_cues = [_phrase_key(c) for c in cues["still_present"]]
        self.self_reference = {_phrase_key(c) for c in cues["self_reference"]}
        self.relations = {_phrase_key(k): v for k, v in cues["family_relations"].items()}
        self.severity_cues = {
            level: [_phrase_key(p) for p in phrases] for level, phrases in cues["severity"].items()
        }
        self.medication_actions = {
            action: [_phrase_key(p) for p in phrases]
            for action, phrases in cues["medication_action"].items()
        }

        # Clinical cutoffs for numeric cycle statements. Configuration, not
        # constants — see the rationale block in lexicon.yaml.
        thresholds = self.lexicon["cycle_thresholds"]
        self.normal_cycle_min_days = float(thresholds["normal_cycle_min_days"])
        self.normal_cycle_max_days = float(thresholds["normal_cycle_max_days"])
        self.max_variability_days = float(thresholds["max_normal_variability_days"])

        self.negation_scope = int(cfg["negation_scope_tokens"])
        self.uncertainty_scope = int(cfg["uncertainty_scope_tokens"])
        self.historical_scope = int(cfg["historical_scope_tokens"])
        self.family_scope = int(cfg["family_scope_tokens"])

        # Only single-token negation cues may be skipped inside a phrase.
        skippable = frozenset(cue[0] for cue in self.negation_cues if len(cue) == 1)

        concept_entries: dict[tuple[str, ...], dict[str, Any]] = {}
        for concept in self.lexicon["concepts"]:
            for phrase in concept["phrases"]:
                concept_entries[_phrase_key(phrase)] = concept
        self.concept_matcher = _Matcher(concept_entries, skippable)

        med_entries: dict[tuple[str, ...], dict[str, Any]] = {}
        for med in self.lexicon.get("medications", []):
            for phrase in med["phrases"]:
                med_entries[_phrase_key(phrase)] = med
        self.medication_matcher = _Matcher(med_entries, skippable)

        self._variables = load_variable_registry().variables

    # -- Public API ---------------------------------------------------------

    def extract(
        self, transcript: Transcript, *, patient_id: str, **kwargs: Any
    ) -> ExtractionResult:
        """Extract every supportable assertion from ``transcript``."""
        tokens = tokenize_transcript(transcript)
        result = ExtractionResult(
            recording_id=transcript.recording_id,
            patient_id=patient_id,
            extractor_version=f"{self.name}/{self.version}",
        )
        if not tokens:
            return result

        sentences = self._sentence_bounds(tokens)
        counter = 0

        for start, end, concept, inner_negation in self.concept_matcher.find_all(tokens):
            lo, hi = sentences[tokens[start].sentence_id]
            if self._is_question(tokens, lo, hi):
                result.suppressed_questions += 1
                continue

            ctx = self._context(tokens, start, end, lo, hi, inner_negation=inner_negation)
            span = self._span(transcript, tokens, start, end)

            code, unmapped_reason = self._resolve_code(concept, ctx["attribution"])
            if code is None:
                result.unmapped.append(
                    UnmappedMention(
                        recording_id=transcript.recording_id,
                        surface_form=span.text,
                        reason=unmapped_reason,
                        evidence=span,
                        attribution=ctx["attribution"],
                    )
                )
                continue

            if ctx["attribution"] == "other":
                result.unmapped.append(
                    UnmappedMention(
                        recording_id=transcript.recording_id,
                        surface_form=span.text,
                        reason="attribution_unresolved_third_party",
                        evidence=span,
                        attribution="other",
                    )
                )
                continue

            counter += 1
            value = concept.get("value", True)
            if ctx["negated"] and isinstance(value, bool):
                value = False

            result.events.append(
                ExtractedSymptomEvent(
                    extraction_id=f"{transcript.recording_id}-x{counter:03d}",
                    recording_id=transcript.recording_id,
                    patient_id=patient_id,
                    canonical_code=code,
                    variable_name=self._variable_name(code),
                    value=value,
                    unit=self._canonical_unit(code),
                    surface_form=span.text,
                    category=concept.get("category", "symptom"),
                    negated=ctx["negated"],
                    historical=ctx["historical"],
                    uncertain=ctx["uncertain"],
                    temporality="historical" if ctx["historical"] else "current",
                    attribution=ctx["attribution"],
                    relation=ctx["relation"],
                    speaker_role=self._speaker_role(tokens[start].speaker_role),
                    onset=self._onset(transcript, tokens, lo, hi),
                    duration_days=self._duration_days(tokens, lo, hi),
                    frequency_per_year=None,
                    severity=self._severity(tokens, lo, hi),
                    evidence=span,
                    extraction_confidence=self._confidence(ctx),
                    extractor_version=f"{self.name}/{self.version}",
                )
            )

        counter = self._extract_medications(transcript, tokens, sentences, result, counter)
        # Numeric extractors: assertions made with numbers rather than words.
        counter = self._extract_numeric_cycle_facts(transcript, tokens, sentences, result, counter)
        counter = self._extract_numeric_weight_change(
            transcript, tokens, sentences, result, counter
        )
        counter = self._extract_numeric_scores(transcript, tokens, sentences, result, counter)
        self._deduplicate(result)
        return result

    @staticmethod
    def _deduplicate(result: ExtractionResult) -> None:
        """Collapse a concept asserted both lexically and numerically.

        "My cycles are all over the place, between 45 and 70 days apart" asserts
        cycle_irregularity twice in one breath. Emitting two events would double
        count it in every metric and make a reviewer confirm the same fact
        twice. The higher-confidence event wins and absorbs any normalized
        detail (duration, onset, range) the loser had, so merging never loses
        information.

        Scoped to one segment: the same symptom raised again later in a
        conversation is a genuinely separate mention and is left alone.
        """
        best: dict[tuple[str | None, str, str, bool], ExtractedSymptomEvent] = {}
        order: list[ExtractedSymptomEvent] = []

        for event in result.events:
            key = (
                event.evidence.segment_id,
                event.canonical_code,
                event.attribution,
                event.negated,
            )
            incumbent = best.get(key)
            if incumbent is None:
                best[key] = event
                order.append(event)
                continue

            winner, loser = (
                (incumbent, event)
                if incumbent.extraction_confidence >= event.extraction_confidence
                else (event, incumbent)
            )
            for field in ("duration_days", "onset", "value_range", "severity"):
                if getattr(winner, field) is None and getattr(loser, field) is not None:
                    setattr(winner, field, getattr(loser, field))
            if winner is not incumbent:
                order[order.index(incumbent)] = winner
                best[key] = winner

        result.events = order

    # -- Concept resolution -------------------------------------------------

    def _resolve_code(
        self, concept: dict[str, Any], attribution: Attribution
    ) -> tuple[str | None, str]:
        """Pick the canonical code for a concept given who it is about."""
        if attribution == "family_member":
            family_code = concept.get("family_code")
            if family_code:
                return family_code, ""
            return None, (
                "no family-history variable exists for this concept; a relative's report "
                "must never be stored as the patient's."
            )
        code = concept.get("code")
        if code:
            return code, ""
        return None, str(concept.get("unmapped_reason", "no canonical patient variable")).strip()

    # -- Context (ConText-style) -------------------------------------------

    def _context(
        self,
        tokens: list[Token],
        start: int,
        end: int,
        lo: int,
        hi: int,
        *,
        inner_negation: bool = False,
    ) -> dict[str, Any]:
        """Resolve negation, uncertainty, temporality and attribution."""
        scope_lo = self._clause_start(tokens, lo, start)
        scope_hi = self._clause_end(tokens, end, hi)

        negated = inner_negation or self._is_negated(tokens, scope_lo, start, end, scope_hi)
        uncertain = (
            _cue_hit(
                tokens, max(scope_lo, start - self.uncertainty_scope), start, self.uncertainty_cues
            )
            is not None
            or _cue_hit(tokens, end, min(scope_hi, end + 3), self.uncertainty_cues) is not None
        )
        attribution, relation = self._attribution(tokens, start, lo, hi)
        # Family history is a standing fact about the family, not a resolved
        # episode: "family history of diabetes" is current, however it is phrased.
        historical = attribution != "family_member" and self._is_historical(tokens, lo, hi)

        return {
            "negated": negated,
            "uncertain": uncertain,
            "historical": historical,
            "attribution": attribution,
            "relation": relation,
        }

    def _is_negated(
        self, tokens: list[Token], scope_lo: int, start: int, end: int, scope_hi: int
    ) -> bool:
        window_lo = max(scope_lo, start - self.negation_scope)
        hit = _cue_hit(tokens, window_lo, start, self.negation_cues)
        if hit is None:
            return False
        cue_index = hit[0]
        # Pseudo-negation: "I'm not sure if..." expresses doubt, not absence.
        for pseudo in self.pseudo_negation:
            if tuple(t.text for t in tokens[cue_index : cue_index + len(pseudo)]) == pseudo:
                return False
        del end, scope_hi
        return True

    def _is_historical(self, tokens: list[Token], lo: int, hi: int) -> bool:
        if _cue_hit(tokens, lo, hi, self.historical_cues) is None:
            return False
        # "I've had acne since high school" is present-tense despite the date.
        return _cue_hit(tokens, lo, hi, self.still_present_cues) is None

    def _attribution(
        self, tokens: list[Token], start: int, lo: int, hi: int
    ) -> tuple[Attribution, str | None]:
        """Nearest preceding attribution marker inside the clause wins."""
        clause_lo = self._clause_start(tokens, lo, start)
        window_lo = max(clause_lo, start - self.family_scope)

        best_index = -1
        best: tuple[Attribution, str | None] = ("unknown", None)
        for index in range(window_lo, start):
            key = (tokens[index].text,)
            two = tuple(t.text for t in tokens[index : index + 2])
            three = tuple(t.text for t in tokens[index : index + 3])
            if three in self.relations:
                best_index, best = index, ("family_member", self.relations[three])
            elif two in self.relations:
                best_index, best = index, ("family_member", self.relations[two])
            elif key in self.relations:
                best_index, best = index, ("family_member", self.relations[key])
            elif key in self.self_reference:
                best_index, best = index, ("patient", None)
            elif tokens[index].text in _THIRD_PERSON:
                speaker = tokens[start].speaker_role
                # A clinician saying "she reports cramping" is describing the
                # patient. A patient saying "her cramping" is describing someone
                # else, and PRISM refuses to guess who.
                best_index, best = (
                    index,
                    (("patient", None) if speaker == "clinician" else ("other", None)),
                )
        if best_index >= 0:
            return best

        sentence_text = " ".join(t.text for t in tokens[lo:hi])
        if "runs in my family" in sentence_text or "family history of" in sentence_text:
            return ("family_member", "unspecified")
        return ("patient", None)

    def _clause_start(self, tokens: list[Token], lo: int, start: int) -> int:
        for index in range(start - 1, lo - 1, -1):
            if (tokens[index].text,) in self.terminators:
                return index + 1
        return lo

    def _clause_end(self, tokens: list[Token], end: int, hi: int) -> int:
        for index in range(end, hi):
            if (tokens[index].text,) in self.terminators:
                return index
        return hi

    def _is_question(self, tokens: list[Token], lo: int, hi: int) -> bool:
        """A question asserts nothing, so it must not create an event."""
        return hi > lo and tokens[hi - 1].raw.rstrip().endswith("?")

    # -- Normalizers --------------------------------------------------------

    def _severity(self, tokens: list[Token], lo: int, hi: int) -> Severity | None:
        level: Severity
        for level in ("severe", "moderate", "mild"):
            if _cue_hit(tokens, lo, hi, self.severity_cues[level]) is not None:
                return level
        return None

    def _duration_days(self, tokens: list[Token], lo: int, hi: int) -> float | None:
        """Normalize "for about a year" / "for three months" into days."""
        text = " ".join(t.text for t in tokens[lo:hi])
        match = re.search(
            r"\bfor (?:about |around |roughly |nearly |over )?"
            r"(\d+|" + "|".join(_NUMBER_WORDS) + r")\s+"
            r"(day|days|week|weeks|month|months|year|years)\b",
            text,
        )
        if not match:
            return None
        return round(self._as_number(match.group(1)) * _DURATION_DAYS[match.group(2)], 2)

    def _onset(self, transcript: Transcript, tokens: list[Token], lo: int, hi: int) -> str | None:
        """Return a coarse onset descriptor; approximate dates stay approximate."""
        del transcript
        text = " ".join(t.text for t in tokens[lo:hi])
        patterns = [
            r"since (high school|college|my teens|puberty|last year|last summer)",
            r"(?:when i was) (\d+)",
            r"(\d+|" + "|".join(_NUMBER_WORDS) + r") (?:years|months|weeks) ago",
            r"in (high school|college)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(0)
        return None

    @staticmethod
    def _as_number(token: str) -> float:
        if token.isdigit():
            return float(token)
        return float(_NUMBER_WORDS.get(token, 1))

    def _confidence(self, ctx: dict[str, Any]) -> float:
        confidence = 0.90
        if ctx["uncertain"]:
            confidence -= 0.25
        if ctx["attribution"] == "family_member":
            confidence -= 0.05
        if ctx["historical"]:
            confidence -= 0.05
        return round(max(0.05, min(0.99, confidence)), 3)

    # -- Medications --------------------------------------------------------

    def _extract_medications(
        self,
        transcript: Transcript,
        tokens: list[Token],
        sentences: dict[int, tuple[int, int]],
        result: ExtractionResult,
        counter: int,
    ) -> int:
        for start, end, med, inner_negation in self.medication_matcher.find_all(tokens):
            lo, hi = sentences[tokens[start].sentence_id]
            if self._is_question(tokens, lo, hi):
                result.suppressed_questions += 1
                continue
            ctx = self._context(tokens, start, end, lo, hi, inner_negation=inner_negation)
            if ctx["attribution"] != "patient":
                continue
            action = self._medication_action(tokens, lo, hi)
            span = self._span(transcript, tokens, start, end)
            counter += 1
            result.events.append(
                ExtractedSymptomEvent(
                    extraction_id=f"{transcript.recording_id}-x{counter:03d}",
                    recording_id=transcript.recording_id,
                    patient_id=result.patient_id,
                    canonical_code="medication_current",
                    variable_name=self._variable_name("medication_current"),
                    value=med["name"],
                    surface_form=span.text,
                    category="medication",
                    negated=ctx["negated"],
                    # A stopped medication is by definition not a current one.
                    historical=ctx["historical"] or action == "stop",
                    uncertain=ctx["uncertain"],
                    temporality=(
                        "historical" if (ctx["historical"] or action == "stop") else "current"
                    ),
                    attribution="patient",
                    speaker_role=self._speaker_role(tokens[start].speaker_role),
                    onset=self._onset(transcript, tokens, lo, hi),
                    duration_days=self._duration_days(tokens, lo, hi),
                    medication_action=action,
                    evidence=span,
                    extraction_confidence=self._confidence(ctx),
                    extractor_version=f"{self.name}/{self.version}",
                )
            )
        return counter

    def _medication_action(self, tokens: list[Token], lo: int, hi: int) -> MedicationAction | None:
        best: tuple[int, MedicationAction] | None = None
        for action, cues in self.medication_actions.items():
            hit = _cue_hit(tokens, lo, hi, cues)
            if hit is not None and (best is None or hit[0] > best[0]):
                best = (hit[0], action)
        return best[1] if best else None

    # -- Numeric cycle facts ------------------------------------------------

    def _range_asserts_irregularity(self, low: float, high: float) -> bool:
        """Decide whether a spoken cycle-length RANGE asserts irregularity.

        Thresholds come from ``lexicon.yaml`` (FIGO / Fraser normal cycle band
        and intercycle variability); see that file for the clinical rationale.

        Only ranges reach this method. A single point value never asserts
        irregularity however long it is, because a reliably 45-day cycle is
        oligomenorrhoea rather than irregularity and the registry has no
        variable for the former. See the note in ``lexicon.yaml``.
        """
        if high < low:
            low, high = high, low
        return (
            (high - low) > self.max_variability_days
            or high > self.normal_cycle_max_days
            or low < self.normal_cycle_min_days
        )

    def _extract_numeric_cycle_facts(
        self,
        transcript: Transcript,
        tokens: list[Token],
        sentences: dict[int, tuple[int, int]],
        result: ExtractionResult,
        counter: int,
    ) -> int:
        """Recover cycle length, variability and frequency from spoken numbers.

        Only fires in a sentence that already mentions periods/cycles, so that
        "three cups of coffee a day" can never become a menstrual frequency.
        """
        number_pattern = r"(\d+|" + "|".join(_NUMBER_WORDS) + r")"
        # "between 45 and 70 days apart", "45 to 70 days", "anywhere from 30 to
        # 60 days", "every 45 to 70 days", "range from 28 to 32 days".
        range_pattern = re.compile(
            r"(?:between|from|anywhere from|ranges? from|every)?\s*"
            + number_pattern
            + r"\s*(?:to|and|or|-)\s*"
            + number_pattern
            + r"\s*days"
        )
        single_pattern = re.compile(r"every\s*" + number_pattern + r"\s*days")
        frequency_pattern = re.compile(
            number_pattern + r"(?: times| periods| period| cycles| cycle)? (?:a|per) year"
        )

        for _sentence_id, (lo, hi) in sentences.items():
            if hi <= lo or self._is_question(tokens, lo, hi):
                continue
            text = " ".join(t.text for t in tokens[lo:hi])
            if not re.search(r"\b(period|periods|cycle|cycles|bleed|menstrual)\b", text):
                continue

            duration = self._duration_days(tokens, lo, hi)
            onset = self._onset(transcript, tokens, lo, hi)
            uncertain = _cue_hit(tokens, lo, hi, self.uncertainty_cues) is not None

            freq = frequency_pattern.search(text)
            if freq:
                counter += 1
                result.events.append(
                    self._numeric_event(
                        transcript,
                        tokens,
                        lo,
                        hi,
                        freq,
                        counter,
                        result.patient_id,
                        code="menstrual_frequency_per_year",
                        value=self._as_number(freq.group(1)),
                        uncertain=uncertain,
                        duration_days=duration,
                        onset=onset,
                    )
                )

            range_match = range_pattern.search(text)
            single_match = None if range_match else single_pattern.search(text)

            if range_match:
                low = self._as_number(range_match.group(1))
                high = self._as_number(range_match.group(2))
                if high < low:
                    low, high = high, low

                if self._range_asserts_irregularity(low, high):
                    # The patient stated the variability as a fact, so this is
                    # NOT uncertain even though the endpoints are approximate.
                    counter += 1
                    result.events.append(
                        self._numeric_event(
                            transcript,
                            tokens,
                            lo,
                            hi,
                            range_match,
                            counter,
                            result.patient_id,
                            code="cycle_irregularity",
                            value=True,
                            uncertain=uncertain,
                            duration_days=duration,
                            onset=onset,
                            confidence=0.85,
                        )
                    )
                # A normal-looking range is deliberately NOT turned into a
                # cycle_regularity='regular' assertion. Flagging an abnormal
                # number for review is a safe machine action; issuing a clean
                # bill of health the patient never gave is not. The asymmetry
                # is intentional.

                # Range -> point value rule: the midpoint, marked uncertain
                # because the pipeline synthesized it rather than heard it. The
                # endpoints survive in value_range and in the evidence span, so
                # nothing the patient said is lost.
                counter += 1
                result.events.append(
                    self._numeric_event(
                        transcript,
                        tokens,
                        lo,
                        hi,
                        range_match,
                        counter,
                        result.patient_id,
                        code="cycle_length",
                        value=round((low + high) / 2.0, 2),
                        uncertain=True,
                        duration_days=duration,
                        onset=onset,
                        value_range=[low, high],
                    )
                )
            elif single_match:
                counter += 1
                result.events.append(
                    self._numeric_event(
                        transcript,
                        tokens,
                        lo,
                        hi,
                        single_match,
                        counter,
                        result.patient_id,
                        code="cycle_length",
                        value=self._as_number(single_match.group(1)),
                        uncertain=uncertain,
                        duration_days=duration,
                        onset=onset,
                    )
                )
        return counter

    def _extract_numeric_weight_change(
        self,
        transcript: Transcript,
        tokens: list[Token],
        sentences: dict[int, tuple[int, int]],
        result: ExtractionResult,
        counter: int,
    ) -> int:
        """Recover weight *change* stated numerically ("put on 20 pounds").

        Emits the binary ``weight_gain``, which is precisely what such a
        statement asserts. It deliberately does NOT emit ``weight``: a delta is
        not an absolute measurement, and the registry has no weight-change
        variable to hold the magnitude. The amount stays in the evidence span so
        a reviewer can see it.
        """
        gain_pattern = re.compile(
            r"(?:gained|put on|gone up (?:by)?|up)\s+(?:about |around |roughly |over )?"
            r"(\d+|" + "|".join(_NUMBER_WORDS) + r")\s*"
            r"(pounds|pound|lbs|lb|kilos|kilograms|kg|stone)\b"
        )
        for _sentence_id, (lo, hi) in sentences.items():
            if hi <= lo or self._is_question(tokens, lo, hi):
                continue
            text = " ".join(t.text for t in tokens[lo:hi])
            match = gain_pattern.search(text)
            if not match:
                continue
            ctx_tokens = self._token_range_for_match(tokens, lo, hi, match)
            ctx = self._context(tokens, ctx_tokens[0], ctx_tokens[1], lo, hi)
            if ctx["attribution"] != "patient":
                continue
            counter += 1
            result.events.append(
                self._numeric_event(
                    transcript,
                    tokens,
                    lo,
                    hi,
                    match,
                    counter,
                    result.patient_id,
                    code="weight_gain",
                    value=not ctx["negated"],
                    uncertain=ctx["uncertain"],
                    negated=ctx["negated"],
                    historical=ctx["historical"],
                    duration_days=self._duration_days(tokens, lo, hi),
                    onset=self._onset(transcript, tokens, lo, hi),
                    confidence=0.85,
                )
            )
        return counter

    def _extract_numeric_scores(
        self,
        transcript: Transcript,
        tokens: list[Token],
        sentences: dict[int, tuple[int, int]],
        result: ExtractionResult,
        counter: int,
    ) -> int:
        """Recover an explicitly stated Ferriman-Gallwey score.

        Only an explicit score is taken. The pipeline never infers an FG score
        from a hair description — that is a scored clinical examination, and
        manufacturing one from "I have a lot of facial hair" would fabricate a
        measurement that nobody performed.
        """
        # Tokens keep internal hyphens ("ferriman-gallwey"), so both the
        # hyphenated and spaced surface forms have to be accepted here.
        score_pattern = re.compile(
            r"(?:ferriman[- ]?gallwey|ferriman|fg)\s*(?:score)?\s*(?:was|is|of|:)?\s*(\d{1,2})\b"
        )
        for _sentence_id, (lo, hi) in sentences.items():
            if hi <= lo or self._is_question(tokens, lo, hi):
                continue
            text = " ".join(t.text for t in tokens[lo:hi])
            match = score_pattern.search(text)
            if not match:
                continue
            counter += 1
            result.events.append(
                self._numeric_event(
                    transcript,
                    tokens,
                    lo,
                    hi,
                    match,
                    counter,
                    result.patient_id,
                    code="ferriman_gallwey_score",
                    value=float(match.group(1)),
                    uncertain=_cue_hit(tokens, lo, hi, self.uncertainty_cues) is not None,
                    confidence=0.9,
                )
            )
        return counter

    @staticmethod
    def _token_range_for_match(
        tokens: list[Token], lo: int, hi: int, match: re.Match[str]
    ) -> tuple[int, int]:
        """Map a regex match on the joined sentence back onto token indices.

        Needed so a numeric event is grounded on the phrase that produced it
        ("between 45 and 70 days") rather than on the whole sentence. A span
        that points at more than its own evidence is a weaker receipt.
        """
        cursor = 0
        start_index: int | None = None
        end_index: int | None = None
        for index in range(lo, hi):
            token_start = cursor
            token_end = cursor + len(tokens[index].text)
            if start_index is None and token_end > match.start():
                start_index = index
            if token_start < match.end():
                end_index = index + 1
            cursor = token_end + 1
        return (lo if start_index is None else start_index, hi if end_index is None else end_index)

    def _numeric_event(
        self,
        transcript: Transcript,
        tokens: list[Token],
        lo: int,
        hi: int,
        match: re.Match[str],
        counter: int,
        patient_id: str,
        *,
        code: str,
        value: float | bool,
        uncertain: bool = False,
        negated: bool = False,
        historical: bool = False,
        duration_days: float | None = None,
        onset: str | None = None,
        value_range: list[float] | None = None,
        confidence: float = 0.8,
    ) -> ExtractedSymptomEvent:
        start, end = self._token_range_for_match(tokens, lo, hi, match)
        span = self._span(transcript, tokens, start, end)
        return ExtractedSymptomEvent(
            extraction_id=f"{transcript.recording_id}-x{counter:03d}",
            recording_id=transcript.recording_id,
            patient_id=patient_id,
            canonical_code=code,
            variable_name=self._variable_name(code),
            value=value,
            unit=self._canonical_unit(code),
            surface_form=match.group(0),
            category="reproductive",
            negated=negated,
            historical=historical,
            uncertain=uncertain,
            temporality="historical" if historical else "current",
            attribution="patient",
            speaker_role=self._speaker_role(tokens[start].speaker_role),
            onset=onset,
            duration_days=duration_days,
            value_range=value_range,
            evidence=span,
            extraction_confidence=round(confidence * (0.75 if uncertain else 1.0), 3),
            extractor_version=f"{self.name}/{self.version}",
        )

    # -- Helpers ------------------------------------------------------------

    @staticmethod
    def _sentence_bounds(tokens: list[Token]) -> dict[int, tuple[int, int]]:
        bounds: dict[int, tuple[int, int]] = {}
        for index, token in enumerate(tokens):
            lo, hi = bounds.get(token.sentence_id, (index, index))
            bounds[token.sentence_id] = (min(lo, index), index + 1)
        return bounds

    @staticmethod
    def _span(transcript: Transcript, tokens: list[Token], start: int, end: int) -> EvidenceSpan:
        char_start = tokens[start].char_start
        char_end = tokens[end - 1].char_end
        t_start, t_end = transcript.time_span_for_chars(char_start, char_end)
        segment = transcript.segment_at(char_start)
        return EvidenceSpan(
            text=transcript.text[char_start:char_end],
            char_start=char_start,
            char_end=char_end,
            start_seconds=t_start,
            end_seconds=t_end,
            segment_id=segment.segment_id if segment else None,
        )

    @staticmethod
    def _speaker_role(role: str) -> SpeakerRole:
        return role if role in {"patient", "clinician"} else "unknown"  # type: ignore[return-value]

    def _variable_name(self, code: str) -> str:
        spec = self._variables.get(code)
        return spec.canonical_name if spec else code

    def _canonical_unit(self, code: str) -> str | None:
        spec = self._variables.get(code)
        if spec is None:
            return None
        return spec.canonical_unit or spec.unit


class LlmExtractor(ExtractionAdapter):
    """Placeholder for an LLM-backed extractor. Never calls an API by default.

    The contract an implementation MUST satisfy, documented here so that adding a
    client later cannot quietly weaken the guarantees the rule-based extractor
    provides:

    1. **Strict JSON only.** The model returns a JSON array of objects matching
       :class:`ExtractedSymptomEvent` exactly. Prose, markdown fences or trailing
       commentary are a hard parse failure, not something to repair.
    2. **Verbatim evidence.** Every object carries ``evidence.text`` copied
       character-for-character from the transcript, plus the offsets. The caller
       re-verifies that ``transcript.text[char_start:char_end] == evidence.text``
       and *drops* any object that fails — a hallucinated span is an unsupported
       event and is counted as such (see ``evaluation/speech.py``).
    3. **Closed vocabulary.** ``canonical_code`` must be a code in
       ``registry/variables.yaml``. Unknown codes become
       :class:`UnmappedMention`, never new variables.
    4. **Attribution is explicit.** The prompt must require ``attribution`` and
       ``relation`` on every object, and the same post-check applies: a
       ``family_history_*`` code with non-family attribution is rejected.
    5. **No diagnosis synthesis.** The prompt forbids inferring a diagnosis the
       speaker did not state, and forbids merging two mentions into one event.
    6. **Confirmation unchanged.** Output is still a proposal: it flows through
       ``confirmation.py`` exactly like rule-based output.
    """

    name = "llm"
    version = "0.0.0"

    def __init__(self, client: Any = None, model: str | None = None) -> None:
        self.client = client
        self.model = model

    def extract(
        self, transcript: Transcript, *, patient_id: str, **kwargs: Any
    ) -> ExtractionResult:
        """Raise unless a client is configured. No network call is ever made here."""
        del transcript, patient_id, kwargs
        raise NotImplementedError(
            "LlmExtractor has no configured client. PRISM's offline default is "
            "RuleBasedExtractor; wire a client explicitly and re-read this class's "
            "docstring for the strict-JSON and evidence-verification contract."
        )
