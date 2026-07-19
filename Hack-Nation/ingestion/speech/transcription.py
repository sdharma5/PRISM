"""Transcription adapters.

Why an adapter boundary: ASR is the one part of the speech pipeline that is
genuinely a moving target (model versions, hosted vs local, language packs).
Everything downstream — extraction, confirmation, evidence linking — depends
only on the :class:`Transcript` contract below, so the ASR engine can be swapped
without touching a single extraction rule, and the evaluation corpus can run
with a scripted adapter that needs no model, no GPU and no network.

Word-level timings are mandatory rather than optional because PRISM requires
every speech-derived event to point back at the seconds of audio that produced
it. An ASR engine that cannot emit word timings cannot be used here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field, model_validator

TRANSCRIPT_SCHEMA_VERSION = "1.0.0"


class TranscriptWord(BaseModel):
    """One token with its audio timing and character offsets in the transcript."""

    text: str
    start_seconds: float = Field(ge=0.0)
    end_seconds: float = Field(ge=0.0)
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check(self) -> TranscriptWord:
        if self.end_seconds < self.start_seconds:
            raise ValueError(f"word '{self.text}': end_seconds precedes start_seconds.")
        if self.char_end < self.char_start:
            raise ValueError(f"word '{self.text}': char_end precedes char_start.")
        return self


class TranscriptSegment(BaseModel):
    """A contiguous utterance attributed to one speaker."""

    segment_id: str
    speaker_role: str = "unknown"
    text: str
    start_seconds: float = Field(ge=0.0)
    end_seconds: float = Field(ge=0.0)
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    words: list[TranscriptWord] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class Transcript(BaseModel):
    """Full transcript for one recording.

    ``text`` is the authoritative character space: every evidence span produced
    downstream indexes into this exact string, so it must never be re-normalized
    after transcription.
    """

    recording_id: str
    language: str = "en"
    text: str
    segments: list[TranscriptSegment] = Field(default_factory=list)
    engine: str = "unknown"
    engine_version: str = "unknown"
    schema_version: str = TRANSCRIPT_SCHEMA_VERSION

    def words(self) -> list[TranscriptWord]:
        """All words across segments, in transcript order."""
        return [w for seg in self.segments for w in seg.words]

    def segment_at(self, char_index: int) -> TranscriptSegment | None:
        """Return the segment containing ``char_index``, if any."""
        for seg in self.segments:
            if seg.char_start <= char_index < seg.char_end:
                return seg
        return None

    def time_span_for_chars(
        self, char_start: int, char_end: int
    ) -> tuple[float | None, float | None]:
        """Map a character span onto the audio seconds that produced it.

        Returns ``(None, None)`` when no word overlaps the span, which callers
        must treat as an ungrounded extraction rather than a zero timestamp.
        """
        overlapping = [
            w for w in self.words() if w.char_start < char_end and w.char_end > char_start
        ]
        if not overlapping:
            return (None, None)
        return (
            min(w.start_seconds for w in overlapping),
            max(w.end_seconds for w in overlapping),
        )


class TranscriptionAdapter(ABC):
    """Contract every ASR backend must satisfy."""

    name: str = "abstract"
    version: str = "0.0.0"

    @abstractmethod
    def transcribe(self, audio: Any, *, recording_id: str, language: str = "en") -> Transcript:
        """Transcribe ``audio`` into a :class:`Transcript` with word timings."""


def _tokenize_with_offsets(text: str, base_offset: int = 0) -> list[tuple[str, int, int]]:
    """Split ``text`` into whitespace-delimited tokens with character offsets."""
    tokens: list[tuple[str, int, int]] = []
    index = 0
    length = len(text)
    while index < length:
        while index < length and text[index].isspace():
            index += 1
        if index >= length:
            break
        start = index
        while index < length and not text[index].isspace():
            index += 1
        tokens.append((text[start:index], base_offset + start, base_offset + index))
    return tokens


class ScriptedTranscriptionAdapter(TranscriptionAdapter):
    """Returns a known transcript with synthetic, evenly-paced word timings.

    Used by the evaluation corpus and every unit test. It exists so that
    extraction accuracy can be measured independently of ASR accuracy: when the
    transcript is exactly the script, any extraction error is an extraction
    error. WER against a real engine is measured separately.
    """

    name = "scripted"
    version = "1.0.0"

    def __init__(self, words_per_second: float = 2.5, start_offset_seconds: float = 0.0) -> None:
        if words_per_second <= 0:
            raise ValueError("words_per_second must be positive.")
        self.words_per_second = words_per_second
        self.start_offset_seconds = start_offset_seconds

    def transcribe(self, audio: Any, *, recording_id: str, language: str = "en") -> Transcript:
        """Build a transcript from a script.

        Args:
            audio: Either a plain string script, or a sequence of
                ``{"speaker_role": ..., "text": ...}`` turn mappings.
            recording_id: Recording this transcript belongs to.
            language: BCP-47-ish language tag.
        """
        turns = self._normalize_turns(audio)
        seconds_per_word = 1.0 / self.words_per_second

        segments: list[TranscriptSegment] = []
        full_text_parts: list[str] = []
        char_cursor = 0
        clock = self.start_offset_seconds

        for index, (speaker_role, turn_text) in enumerate(turns):
            seg_start_char = char_cursor
            tokens = _tokenize_with_offsets(turn_text, base_offset=seg_start_char)
            words: list[TranscriptWord] = []
            seg_start_time = clock
            for token, c_start, c_end in tokens:
                words.append(
                    TranscriptWord(
                        text=token,
                        start_seconds=round(clock, 4),
                        end_seconds=round(clock + seconds_per_word, 4),
                        char_start=c_start,
                        char_end=c_end,
                    )
                )
                clock += seconds_per_word
            segments.append(
                TranscriptSegment(
                    segment_id=f"{recording_id}-seg{index:03d}",
                    speaker_role=speaker_role,
                    text=turn_text,
                    start_seconds=round(seg_start_time, 4),
                    end_seconds=round(clock, 4),
                    char_start=seg_start_char,
                    char_end=seg_start_char + len(turn_text),
                    words=words,
                )
            )
            full_text_parts.append(turn_text)
            char_cursor += len(turn_text) + 1  # +1 for the joining newline
            clock += 0.2  # small inter-turn gap

        return Transcript(
            recording_id=recording_id,
            language=language,
            text="\n".join(full_text_parts),
            segments=segments,
            engine=self.name,
            engine_version=self.version,
        )

    @staticmethod
    def _normalize_turns(audio: Any) -> list[tuple[str, str]]:
        if isinstance(audio, str):
            return [("patient", audio)]
        if isinstance(audio, dict):
            return [(str(audio.get("speaker_role", "patient")), str(audio["text"]))]
        turns: list[tuple[str, str]] = []
        for turn in audio:
            if isinstance(turn, str):
                turns.append(("patient", turn))
            else:
                turns.append((str(turn.get("speaker_role", "patient")), str(turn["text"])))
        return turns


def _word_field(raw_word: Any, field: str) -> Any:
    """Read a field from a word object, supporting both attribute and dict access.

    faster-whisper returns ``Word`` objects (attribute access); openai-whisper
    returns plain dicts (item access). Using ``getattr(...) or raw_word[field]``
    is wrong because a legitimate ``0.0`` start time is falsy and falls through
    to the subscript, which raises ``TypeError`` on a ``Word`` object. This
    helper checks for ``None`` explicitly instead.
    """
    val = getattr(raw_word, field, None)
    if val is None and isinstance(raw_word, dict):
        val = raw_word.get(field)
    return val


class WhisperTranscriptionAdapter(TranscriptionAdapter):
    """Whisper / faster-whisper backend, imported lazily.

    The import is deferred to :meth:`transcribe` so that merely importing this
    module — which the test suite does — never pulls in torch or a model file.
    """

    name = "whisper"
    version = "0.1.0"

    def __init__(
        self,
        model_size: str = "base",
        *,
        device: str = "cpu",
        implementation: str = "faster_whisper",
        compute_type: str | None = None,
        cpu_threads: int = 0,
    ) -> None:
        """
        Args:
            model_size: Whisper checkpoint size.
            device: "cpu" or "cuda".
            implementation: "faster_whisper" or "whisper".
            compute_type: ctranslate2 precision. Defaults to int8 on CPU, which
                is several times faster than the float32 that CPU inference
                otherwise falls back to -- float16 is unsupported there, so the
                weights get silently upconverted and every inference pays for it.
            cpu_threads: 0 lets ctranslate2 decide. Set explicitly when the
                process is confined to fewer cores than the machine reports,
                which is the normal case under a scheduler: ctranslate2 sizes its
                pool from the host core count and then thrashes inside a
                one-core allocation.
        """
        self.model_size = model_size
        self.device = device
        self.implementation = implementation
        self.compute_type = compute_type or ("int8" if device == "cpu" else "float16")
        self.cpu_threads = cpu_threads
        self._model: Any = None

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            if self.implementation == "faster_whisper":
                from faster_whisper import WhisperModel

                self._model = WhisperModel(
                    self.model_size,
                    device=self.device,
                    compute_type=self.compute_type,
                    cpu_threads=self.cpu_threads,
                )
            else:
                import whisper

                self._model = whisper.load_model(self.model_size, device=self.device)
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError(
                "WhisperTranscriptionAdapter requires an optional dependency that is not "
                "installed. Install it with `pip install '.[speech]'` plus one of "
                "`faster-whisper` or `openai-whisper`, or use "
                "ScriptedTranscriptionAdapter for offline evaluation."
            ) from exc
        return self._model

    def transcribe(
        self, audio: Any, *, recording_id: str, language: str = "en"
    ) -> Transcript:  # pragma: no cover - requires optional dependency
        """Transcribe a path or waveform with word-level timestamps."""
        model = self._load_model()
        if self.implementation == "faster_whisper":
            raw_segments, _info = model.transcribe(audio, language=language, word_timestamps=True)
            raw_segments = list(raw_segments)
            turns = [
                {"start": s.start, "end": s.end, "text": s.text, "words": list(s.words or [])}
                for s in raw_segments
            ]
        else:
            result = model.transcribe(audio, language=language, word_timestamps=True)
            turns = [
                {
                    "start": s["start"],
                    "end": s["end"],
                    "text": s["text"],
                    "words": s.get("words", []),
                }
                for s in result["segments"]
            ]
        return self._to_transcript(turns, recording_id=recording_id, language=language)

    def _to_transcript(
        self, turns: list[dict[str, Any]], *, recording_id: str, language: str
    ) -> Transcript:  # pragma: no cover - requires optional dependency
        segments: list[TranscriptSegment] = []
        text_parts: list[str] = []
        char_cursor = 0
        for index, turn in enumerate(turns):
            seg_text = str(turn["text"]).strip()
            words: list[TranscriptWord] = []
            search_from = 0
            for raw_word in turn.get("words", []):
                token = str(_word_field(raw_word, "word") or "").strip()
                if not token:
                    continue
                local = seg_text.find(token, search_from)
                if local < 0:
                    local = search_from
                search_from = local + len(token)
                start = float(_word_field(raw_word, "start") or 0.0)
                end = float(_word_field(raw_word, "end") or 0.0)
                words.append(
                    TranscriptWord(
                        text=token,
                        start_seconds=start,
                        end_seconds=max(start, end),
                        char_start=char_cursor + local,
                        char_end=char_cursor + local + len(token),
                    )
                )
            segments.append(
                TranscriptSegment(
                    segment_id=f"{recording_id}-seg{index:03d}",
                    speaker_role="unknown",
                    text=seg_text,
                    start_seconds=float(turn["start"]),
                    end_seconds=float(turn["end"]),
                    char_start=char_cursor,
                    char_end=char_cursor + len(seg_text),
                    words=words,
                )
            )
            text_parts.append(seg_text)
            char_cursor += len(seg_text) + 1
        return Transcript(
            recording_id=recording_id,
            language=language,
            text="\n".join(text_parts),
            segments=segments,
            engine=f"{self.name}:{self.implementation}",
            engine_version=self.model_size,
        )
