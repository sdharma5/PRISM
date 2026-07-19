"""Audio recording metadata and offline audio-quality assessment.

Why this module exists: everything downstream of speech (transcription,
extraction, confirmation) inherits the quality ceiling of the recording. A
symptom extracted from a clipped, mostly-silent phone recording is not the same
evidence as one extracted from a clean clinic recording, and the modality token
must be able to say so. So quality is measured *once*, here, on the waveform,
and carried forward as a number rather than re-guessed by later stages.

Consent is enforced here rather than at the API boundary because this is the
first module that touches the waveform: if consent was not recorded, no code in
PRISM should ever have been able to read the samples at all.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

RecordingMode = Literal["patient_intake", "clinician_dictation", "consultation", "self_report"]
RetentionPolicy = Literal["discard_after_transcription", "retain_encrypted", "not_retained"]

#: Amplitude at/above which a normalized sample counts as clipped.
CLIPPING_THRESHOLD = 0.99

#: RMS below which a frame counts as silence, relative to the loudest frame.
SILENCE_RELATIVE_FLOOR = 0.02

DEFAULT_FRAME_MS = 25.0


class ConsentError(PermissionError):
    """Raised when audio processing is attempted without recorded consent.

    A distinct exception type (rather than a bare ``ValueError``) so that callers
    can never accidentally catch and swallow it alongside parsing errors.
    """


class AudioQualityReport(BaseModel):
    """Objective, waveform-level quality measures for one recording.

    Deliberately not a single opaque score: a reviewer needs to know *why*
    quality was low, because clipping (recording too hot) and silence fraction
    (microphone too far away) call for different fixes.
    """

    duration_seconds: float = Field(ge=0.0)
    sample_rate_hz: int = Field(gt=0)
    rms: float = Field(ge=0.0)
    peak_amplitude: float = Field(ge=0.0)
    clipping_fraction: float = Field(ge=0.0, le=1.0)
    silence_fraction: float = Field(ge=0.0, le=1.0)
    estimated_snr_db: float
    quality_score: float = Field(ge=0.0, le=1.0)
    warnings: list[str] = Field(default_factory=list)


class AudioRecording(BaseModel):
    """One consented audio recording and its provenance.

    The waveform itself is intentionally *not* a field: PRISM stores metadata and
    derived text, and the retention policy decides whether the bytes survive at
    all. Keeping samples out of the model makes it impossible to serialize raw
    patient audio into an artifact by accident.
    """

    recording_id: str
    patient_id: str
    mode: RecordingMode
    language: str = "en"
    duration_seconds: float = Field(ge=0.0)
    sample_rate_hz: int = Field(gt=0)
    consent_recorded: bool
    raw_audio_retention: RetentionPolicy = "discard_after_transcription"
    quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    source_dataset: str | None = None
    notes: str = ""

    def require_consent(self) -> None:
        """Fail closed when consent was not recorded.

        Called by every stage that reads audio or its transcript.
        """
        if not self.consent_recorded:
            raise ConsentError(
                f"recording {self.recording_id}: consent_recorded is False; "
                "PRISM refuses to process audio without recorded consent."
            )


def _frame_rms(waveform: np.ndarray, frame_len: int) -> np.ndarray:
    """Root-mean-square energy per fixed-length frame (trailing partial dropped)."""
    n_frames = max(1, len(waveform) // frame_len)
    usable = waveform[: n_frames * frame_len].reshape(n_frames, frame_len)
    return np.sqrt(np.mean(np.square(usable), axis=1))


def assess_audio_quality(
    waveform: np.ndarray,
    sample_rate_hz: int,
    *,
    frame_ms: float = DEFAULT_FRAME_MS,
) -> AudioQualityReport:
    """Measure clipping, silence and a coarse SNR from a mono waveform.

    The SNR estimate is deliberately crude: it compares the energy of frames
    judged to be speech against the energy of frames judged to be silence. It is
    not a calibrated acoustic measurement and must never be reported as one — it
    exists to rank recordings and to flag unusable ones, nothing more.

    Args:
        waveform: Mono samples. Any float or int dtype; scaled to [-1, 1] using
            the dtype's full scale for integers so clipping means what it says.
        sample_rate_hz: Sampling rate of ``waveform``.
        frame_ms: Analysis frame length in milliseconds.

    Returns:
        An :class:`AudioQualityReport`.
    """
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive.")

    samples = np.asarray(waveform).astype(np.float64).ravel()
    if np.issubdtype(np.asarray(waveform).dtype, np.integer):
        full_scale = float(np.iinfo(np.asarray(waveform).dtype).max)
        samples = samples / full_scale

    warnings: list[str] = []
    if samples.size == 0:
        return AudioQualityReport(
            duration_seconds=0.0,
            sample_rate_hz=sample_rate_hz,
            rms=0.0,
            peak_amplitude=0.0,
            clipping_fraction=0.0,
            silence_fraction=1.0,
            estimated_snr_db=-60.0,
            quality_score=0.0,
            warnings=["empty_waveform"],
        )

    duration = samples.size / sample_rate_hz
    peak = float(np.max(np.abs(samples)))
    rms = float(np.sqrt(np.mean(np.square(samples))))
    clipping_fraction = float(np.mean(np.abs(samples) >= CLIPPING_THRESHOLD))

    frame_len = max(1, int(sample_rate_hz * frame_ms / 1000.0))
    frames = _frame_rms(samples, frame_len)
    frame_peak = float(np.max(frames)) if frames.size else 0.0
    floor = frame_peak * SILENCE_RELATIVE_FLOOR
    silent = frames <= floor if frame_peak > 0 else np.ones_like(frames, dtype=bool)
    silence_fraction = float(np.mean(silent))

    speech_power = float(np.mean(np.square(frames[~silent]))) if np.any(~silent) else 0.0
    noise_power = float(np.mean(np.square(frames[silent]))) if np.any(silent) else 0.0
    if speech_power <= 0.0:
        snr_db = -60.0
    elif noise_power <= 0.0:
        # No measurable noise floor. Cap rather than report an infinite SNR.
        snr_db = 60.0
    else:
        snr_db = float(10.0 * np.log10(speech_power / noise_power))
    snr_db = float(np.clip(snr_db, -60.0, 60.0))

    if clipping_fraction > 0.01:
        warnings.append("clipping_detected")
    if silence_fraction > 0.8:
        warnings.append("mostly_silent")
    if snr_db < 10.0:
        warnings.append("low_snr")
    if peak < 0.05:
        warnings.append("very_low_level")

    quality = _quality_score(snr_db, clipping_fraction, silence_fraction)

    return AudioQualityReport(
        duration_seconds=duration,
        sample_rate_hz=sample_rate_hz,
        rms=rms,
        peak_amplitude=peak,
        clipping_fraction=clipping_fraction,
        silence_fraction=silence_fraction,
        estimated_snr_db=snr_db,
        quality_score=quality,
        warnings=warnings,
    )


def _quality_score(snr_db: float, clipping_fraction: float, silence_fraction: float) -> float:
    """Combine the three measures into one 0-1 score.

    SNR is mapped linearly from 0 dB (unusable) to 30 dB (clean); clipping and
    excess silence are multiplicative penalties because either one alone can
    render a recording untranscribable regardless of the others.
    """
    snr_component = float(np.clip((snr_db - 0.0) / 30.0, 0.0, 1.0))
    clip_penalty = float(np.clip(1.0 - clipping_fraction * 10.0, 0.0, 1.0))
    # Some silence is normal speech pacing; only penalise beyond half the file.
    excess_silence = max(0.0, silence_fraction - 0.5) / 0.5
    silence_penalty = float(np.clip(1.0 - excess_silence, 0.0, 1.0))
    return float(np.clip(snr_component * clip_penalty * silence_penalty, 0.0, 1.0))


def assess_recording(
    recording: AudioRecording,
    waveform: np.ndarray,
    *,
    frame_ms: float = DEFAULT_FRAME_MS,
) -> AudioQualityReport:
    """Consent-checked wrapper around :func:`assess_audio_quality`."""
    recording.require_consent()
    return assess_audio_quality(waveform, recording.sample_rate_hz, frame_ms=frame_ms)
