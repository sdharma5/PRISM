"""Speech transcription and clinical-event extraction.

Adapted from the ``prism_voice_input`` package, which shipped this as a second
standalone FastAPI app in ``server/main.py``. It is a router here instead, for
three reasons that are not stylistic:

* Two apps cannot both own port 8000, and the frontend already talks to this one.
* CORS, logging and error shape would otherwise be configured twice and drift.
* A patient's speech becomes clinical events, which belong in the same event
  ledger as everything else. A separate service would have its own.

**Models load lazily, not at startup.** The original loaded Whisper in a startup
hook, which means a missing ``faster-whisper`` takes down the entire API --
including inference, which has nothing to do with speech. Here the first
transcription request pays the load cost, and an absent dependency degrades this
one endpoint to a 503 that names what is missing.

Nothing this endpoint returns is confirmed. Events come back ``proposed``; the
patient confirms or rejects each one in the UI before it counts as evidence.
"""

from __future__ import annotations

import contextlib
import logging
import os
import tempfile
import uuid
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/speech", tags=["speech"])

#: Whisper model size. `base` is ~150MB and downloads once on first use.
WHISPER_MODEL = os.environ.get("PRISM_WHISPER_MODEL", "base")
WHISPER_DEVICE = os.environ.get("PRISM_WHISPER_DEVICE", "cpu")
#: ctranslate2 precision. int8 on CPU is several times faster than the float32
#: that CPU inference otherwise falls back to.
WHISPER_COMPUTE_TYPE = os.environ.get("PRISM_WHISPER_COMPUTE_TYPE") or None
#: Thread pool size. Left at 0 (auto) unless set, but auto reads the *host* core
#: count, which is wrong under a scheduler that confines the process to fewer.
WHISPER_CPU_THREADS = int(os.environ.get("PRISM_WHISPER_CPU_THREADS", "0"))

#: Populated on first successful load and reused for the process lifetime.
_transcriber: Any = None
_extractor: Any = None
_load_error: str | None = None


def speech_available() -> tuple[bool, str | None]:
    """Import check, not a load -- ``/models/status`` calls this on every page
    render and loading Whisper to answer it would cost hundreds of MB."""
    try:
        import faster_whisper  # noqa: F401
    except ImportError:
        return False, (
            "Speech transcription requires faster-whisper, which is not installed. "
            'Install it with: pip install -e ".[speech-realtime]"'
        )
    return True, None


def _load() -> tuple[Any, Any]:
    """Load the transcriber and extractor once, on first use."""
    global _transcriber, _extractor, _load_error

    if _transcriber is not None and _extractor is not None:
        return _transcriber, _extractor

    if _load_error is not None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error": {"code": "speech_unavailable", "message": _load_error}},
        )

    available, reason = speech_available()
    if not available:
        _load_error = reason
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error": {"code": "speech_unavailable", "message": reason}},
        )

    from ingestion.speech.extraction import RuleBasedExtractor
    from ingestion.speech.transcription import WhisperTranscriptionAdapter

    logger.info("Loading Whisper (model=%s, device=%s)...", WHISPER_MODEL, WHISPER_DEVICE)
    try:
        _transcriber = WhisperTranscriptionAdapter(
            model_size=WHISPER_MODEL,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE,
            cpu_threads=WHISPER_CPU_THREADS,
        )
        _extractor = RuleBasedExtractor()
    except Exception as exc:  # noqa: BLE001 - surfaced as 503, not a crash
        _load_error = f"Whisper model '{WHISPER_MODEL}' could not be loaded: {exc}"
        logger.exception("speech model load failed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error": {"code": "speech_model_load_failed", "message": _load_error}},
        ) from exc

    logger.info("Speech pipeline ready (whisper=%s).", WHISPER_MODEL)
    return _transcriber, _extractor


def _suffix_for(filename: str | None, content_type: str | None) -> str:
    """ffmpeg infers the demuxer from the extension, so a webm blob written to a
    `.wav` path fails to decode."""
    name = (filename or "").lower()
    for ext in (".webm", ".ogg", ".oga", ".m4a", ".mp4", ".wav", ".mp3", ".flac"):
        if name.endswith(ext):
            return ".ogg" if ext == ".oga" else ".m4a" if ext == ".mp4" else ext

    ct = (content_type or "").lower()
    for needle, ext in (
        ("webm", ".webm"),
        ("ogg", ".ogg"),
        ("mp4", ".m4a"),
        ("m4a", ".m4a"),
        ("wav", ".wav"),
        ("mpeg", ".mp3"),
        ("flac", ".flac"),
    ):
        if needle in ct:
            return ext

    return ".webm"  # the browser MediaRecorder default


def _decode_audio(path: str) -> tuple[Any, int] | None:
    """Decode to mono float64, whatever container the browser sent.

    libsndfile first, then PyAV. The fallback matters: libsndfile can't read
    webm/opus, which is what MediaRecorder produces by default -- so without it
    quality is absent for every real recording but fine in wav-based tests.
    """
    import numpy as np

    try:
        import soundfile as sf

        waveform, sample_rate = sf.read(path, dtype="float64", always_2d=False)
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=1)  # Whisper and the scorer want mono
        return waveform, int(sample_rate)
    except Exception:  # noqa: BLE001 - unsupported container, try PyAV
        pass

    try:
        import av

        with av.open(path) as container:
            stream = next((s for s in container.streams if s.type == "audio"), None)
            if stream is None:
                return None
            chunks = [
                frame.to_ndarray().astype("float64").reshape(-1)
                for frame in container.decode(stream)
            ]
        if not chunks:
            return None
        waveform = np.concatenate(chunks)
        # PyAV yields int16 for s16 streams; scale to the [-1, 1] the scorer
        # expects, or loudness thresholds are meaningless.
        if np.abs(waveform).max() > 1.0:
            waveform = waveform / 32768.0
        return waveform, int(stream.rate or 48000)
    except Exception as exc:  # noqa: BLE001
        logger.warning("PyAV decode failed: %s", exc)
        return None


def _best_effort_audio_quality(path: str) -> tuple[float | None, list[str]]:
    """Score audio quality. Never raises -- quality is informative, not gating."""
    try:
        from ingestion.speech.audio import assess_audio_quality

        decoded = _decode_audio(path)
        if decoded is None:
            return None, ["audio_quality_decode_failed"]
        waveform, sample_rate = decoded
        if waveform.size == 0:
            return None, ["audio_quality_empty_waveform"]
        report = assess_audio_quality(waveform, sample_rate)
        return report.quality_score, report.warnings
    except Exception as exc:  # noqa: BLE001 - must not fail the request
        logger.warning("audio quality assessment skipped: %s", exc)
        return None, [f"audio_quality_assessment_skipped:{type(exc).__name__}"]


@router.post("/transcribe")
async def transcribe(
    audio: UploadFile = File(...),
    patient_id: str = Form("demo-patient"),
    language: str = Form("en"),
) -> dict[str, Any]:
    """Transcribe an upload and extract proposed clinical events.

    Extraction handles negation and family-vs-patient attribution but isn't
    reliable enough to enter the record unreviewed.
    """
    from ingestion.speech.validation import drop_unsupported, validate_extractions

    transcriber, extractor = _load()

    raw = await audio.read()
    if not raw:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": {
                    "code": "empty_audio_upload",
                    "message": "The uploaded audio file is empty.",
                }
            },
        )

    suffix = _suffix_for(audio.filename, audio.content_type)
    tmp_path: str | None = None
    try:
        # delete=False because the path is handed to ffmpeg and soundfile, which
        # reopen it by name; cleanup happens in the finally block.
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(raw)
            tmp_path = tmp.name

        recording_id = f"rec-{uuid.uuid4().hex[:12]}"

        try:
            transcript = transcriber.transcribe(
                tmp_path, recording_id=recording_id, language=language
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("transcription failed")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail={"error": {"code": "transcription_failed", "message": str(exc)}},
            ) from exc

        try:
            result = extractor.extract(transcript, patient_id=patient_id)
        except Exception as exc:  # noqa: BLE001
            logger.exception("extraction failed")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail={"error": {"code": "extraction_failed", "message": str(exc)}},
            ) from exc

        # Extractions with no evidence span in the transcript are dropped.
        supported, unsupported = drop_unsupported(transcript, result.events)
        report = validate_extractions(transcript, supported)
        rejected = set(report.unsupported_extraction_ids)
        supported = [e for e in supported if e.extraction_id not in rejected]

        quality_score, quality_warnings = _best_effort_audio_quality(tmp_path)

        warnings: list[str] = [*result.warnings, *quality_warnings]
        if unsupported:
            warnings.append(f"{len(unsupported)} extraction(s) dropped for missing evidence spans.")
        if rejected:
            warnings.append(f"{len(rejected)} extraction(s) failed validation.")

        return {
            "recording_id": transcript.recording_id,
            "patient_id": patient_id,
            "transcript": {
                "recording_id": transcript.recording_id,
                "language": transcript.language,
                "text": transcript.text,
                "segments": [seg.model_dump(mode="json") for seg in transcript.segments],
                "engine": transcript.engine,
                "engine_version": transcript.engine_version,
            },
            "events": [e.model_dump(mode="json") for e in supported],
            "unsupported_count": len(unsupported) + len(rejected),
            "audio_quality_score": quality_score,
            # No diarization -- Whisper can't tell patient from clinician.
            "speaker_diarization": False,
            "warnings": warnings,
        }
    finally:
        if tmp_path is not None:
            # The upload is patient audio; remove it whatever happened above.
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
