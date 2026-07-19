"""Health and model-status routes.

``/models/status`` is the endpoint the frontend keys off to disable or qualify
features. It exists so that model availability is never hardcoded in a React
component: a branch gated off in ``inference_encoders.yaml`` must go dark in the
UI on the next request, without a redeploy.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from apps.api.deps import get_registry
from apps.api.registry import ModelRegistry
from apps.api.routers.speech import WHISPER_MODEL, speech_available
from apps.api.schemas.responses import (
    CalibrationStatusView,
    ModelStatusResponse,
    SpeechStatusView,
)

router = APIRouter(prefix="/api/v1", tags=["system"])


@router.get("/health")
def health() -> dict[str, str]:
    """Liveness only.

    Deliberately does not touch the registry: this answers "is the process up",
    and a readiness question that depends on the models belongs in
    ``/models/status``, where the answer can be specific about which branch.
    """
    return {"status": "ok"}


@router.get("/models/status", response_model=ModelStatusResponse)
def models_status(registry: ModelRegistry = Depends(get_registry)) -> ModelStatusResponse:
    """Per-branch availability, with a reason wherever a branch is unavailable."""
    branches = registry.status_payload()
    return ModelStatusResponse(
        static_clinical=branches["static_clinical"],
        temporal_state=branches["temporal_state"],
        ovarian_ultrasound=branches["ovarian_ultrasound"],
        calibration=CalibrationStatusView(
            available=registry.calibrator_loaded,
            method="platt_scaling" if registry.calibrator_loaded else None,
            note=(
                None
                if registry.calibrator_loaded
                else "No calibrator loaded; model scores are reported raw and uncalibrated."
            ),
        ),
        speech=_speech_status(),
        warnings=list(registry.warnings),
    )


def _speech_status() -> SpeechStatusView:
    """Speech availability, checked by import rather than by loading a model.

    The frontend disables the recorder from this, so it must be cheap enough to
    call on every status poll.
    """
    available, reason = speech_available()
    return SpeechStatusView(
        available=available,
        model=WHISPER_MODEL if available else None,
        reason=reason,
    )
