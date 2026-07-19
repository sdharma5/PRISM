"""Build encoders from config, refusing to silently substitute an untrained one.

The whole point of this module is one rule:

    A production inference run must never receive a heuristic encoder when it
    asked for a trained one.

The heuristic ultrasound segmenter is a genuinely useful scipy pipeline -- it
keeps CI torch-free and it exercises the counting, gating and measurement logic
where the clinically consequential bugs live. What it has not done is learn
anything. Numbers it produces look exactly like numbers from a trained model,
carry no marker distinguishing them, and would flow unchanged into a patient
report. So the fallback is available only when a config explicitly asks for it,
and when it is used the resulting token says so in its warnings.

A missing checkpoint is therefore a configuration error, not a degraded mode.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from registry.loader import _read_yaml  # noqa: PLC2701 - same loader the registry uses

__all__ = ["EncoderConfigurationError", "build_encoders", "load_encoder_config"]

REPO_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_PATH = REPO_ROOT / "configs" / "models" / "inference_encoders.yaml"


class EncoderConfigurationError(RuntimeError):
    """Raised when a configured encoder cannot be built as specified."""


def load_encoder_config(path: Path | None = None) -> dict[str, Any]:
    """Load the encoder implementation config."""
    return _read_yaml(path or _CONFIG_PATH)


def _resolve(path_like: str) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else (REPO_ROOT / path)


def _build_static(section: dict[str, Any]) -> Any:
    from models.tabular.encoder import StaticClinicalEncoder  # noqa: PLC0415

    directory = _resolve(section["checkpoint_path"])
    try:
        return StaticClinicalEncoder.load(directory)
    except FileNotFoundError as exc:
        raise EncoderConfigurationError(
            f"static_clinical_encoder is enabled but no trained artifact exists at "
            f"{directory}. Train one with scripts/train_static_encoder.py."
        ) from exc


def _build_ultrasound(section: dict[str, Any]) -> Any:
    implementation = section.get("implementation", "trained_unet")

    if implementation == "heuristic":
        if not section.get("allow_heuristic_fallback", False):
            raise EncoderConfigurationError(
                "ultrasound_encoder.implementation='heuristic' requires "
                "allow_heuristic_fallback=true. The heuristic segmenter is not a trained "
                "model and must be selected deliberately, never as a default."
            )
        from models.ultrasound.encoder import UltrasoundEncoder  # noqa: PLC0415

        return UltrasoundEncoder()

    if implementation != "trained_unet":
        raise EncoderConfigurationError(
            f"Unknown ultrasound implementation '{implementation}'; "
            "expected 'trained_unet' or 'heuristic'."
        )

    from models.ultrasound.trained_encoder import TrainedUltrasoundEncoder  # noqa: PLC0415

    checkpoint = _resolve(section["checkpoint_path"])
    try:
        return TrainedUltrasoundEncoder.load(checkpoint)
    except FileNotFoundError as exc:
        # Deliberately NOT falling back. See the module docstring.
        raise EncoderConfigurationError(
            f"ultrasound_encoder is configured as 'trained_unet' but no checkpoint exists "
            f"at {checkpoint}. Train one with:\n"
            "  python scripts/train_ultrasound_encoder.py "
            "--config configs/experiments/exp_usova3d_3d_unet.yaml\n"
            "To use the untrained heuristic instead, set implementation='heuristic' AND "
            "allow_heuristic_fallback=true — and understand that its output is not from "
            "a trained model."
        ) from exc


def _build_temporal(section: dict[str, Any]) -> Any:
    implementation = section.get("implementation", "target_specific_v1")

    if implementation == "target_specific_v1":
        from models.temporal.state_encoder import TemporalStateEncoder  # noqa: PLC0415

        directory = _resolve(section["checkpoint_path"])
        try:
            return TemporalStateEncoder.load(directory)
        except FileNotFoundError as exc:
            raise EncoderConfigurationError(
                f"temporal_encoder is enabled but no persisted encoder exists at {directory}. "
                "Fit one with scripts/train_temporal_state_encoder.py."
            ) from exc

    if implementation in ("echo_state", "trained_gru"):
        # Neither is available as a persisted, benchmarked artifact. The
        # echo-state model's recurrent weights are fixed random projections and
        # it has never been run on the frozen split; the GRU does not exist. Both
        # must beat the target-specific encoder on the same protocol before they
        # may be selected here.
        raise EncoderConfigurationError(
            f"temporal_encoder.implementation='{implementation}' is not available. Neither "
            "the echo-state model nor a learned GRU has been evaluated on the frozen "
            "participant split, so neither may be preferred over 'target_specific_v1', "
            "whose held-out numbers are recorded in "
            "artifacts/encoders/temporal_state_v1/benchmark_metrics.json."
        )

    raise EncoderConfigurationError(
        f"Unknown temporal implementation '{implementation}'; expected 'target_specific_v1'."
    )


def build_encoders(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Instantiate every enabled encoder.

    Args:
        config: Parsed encoder config; loaded from disk when omitted.

    Returns:
        ``{"static_encoder": ..., "ultrasound_encoder": ..., "temporal_encoder": ...}``
        with None for disabled branches, ready to splat into the orchestrator.

    Raises:
        EncoderConfigurationError: If an enabled encoder cannot be built.
    """
    config = config or load_encoder_config()
    builders = {
        "static_encoder": ("static_clinical_encoder", _build_static),
        "ultrasound_encoder": ("ultrasound_encoder", _build_ultrasound),
        "temporal_encoder": ("temporal_encoder", _build_temporal),
    }

    encoders: dict[str, Any] = {}
    for keyword, (section_name, builder) in builders.items():
        section = config.get(section_name, {}) or {}
        encoders[keyword] = builder(section) if section.get("enabled", False) else None
    return encoders
