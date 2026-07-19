"""Inference routes.

Per-branch routes let a client exercise one encoder in isolation. They cannot
produce a whole-patient score: only the static branch is entitled to one, and
the mapper refuses to populate it without that branch.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status

from apps.api.deps import get_registry
from apps.api.registry import ModelRegistry
from apps.api.schemas.requests import (
    PatientInferenceRequest,
    StaticInferenceRequest,
    TemporalInferenceRequest,
    UltrasoundInferenceRequest,
)
from apps.api.schemas.responses import WebsitePMOSProfileResponse
from inference.patient_bundle import PatientDataBundle
from inference.presentation.website_mapper import to_website_response

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/patients", tags=["inference"])


def _run(registry: ModelRegistry, bundle: PatientDataBundle) -> WebsitePMOSProfileResponse:
    """Run the orchestrator and map the result.

    Per-encoder failures land in ``report.warnings`` rather than raising, so one
    optional branch falling over doesn't cost the whole report.
    """
    report = registry.orchestrator.run(bundle)
    return to_website_response(report)


@router.post("/infer", response_model=WebsitePMOSProfileResponse)
def infer(
    payload: PatientInferenceRequest,
    registry: ModelRegistry = Depends(get_registry),
) -> WebsitePMOSProfileResponse:
    """Main route: run every branch there is input for, skip the rest."""
    try:
        bundle = payload.to_bundle()
    except ValueError as exc:
        # Almost always a nested record naming a different patient.
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc

    return _run(registry, bundle)


@router.post("/infer/static", response_model=WebsitePMOSProfileResponse)
def infer_static(
    payload: StaticInferenceRequest,
    registry: ModelRegistry = Depends(get_registry),
) -> WebsitePMOSProfileResponse:
    """Static-clinical branch only."""
    if not registry.is_available("static_clinical"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The static clinical encoder is not available.",
        )
    try:
        bundle = payload.to_bundle()
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    return _run(registry, bundle)


@router.post("/infer/temporal", response_model=WebsitePMOSProfileResponse)
def infer_temporal(
    payload: TemporalInferenceRequest,
    registry: ModelRegistry = Depends(get_registry),
) -> WebsitePMOSProfileResponse:
    """Longitudinal branch only. Never yields a whole-patient PMOS score."""
    if not registry.is_available("temporal_state"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The temporal state encoder is not available.",
        )
    try:
        bundle = payload.to_bundle()
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    return _run(registry, bundle)


@router.post("/infer/ultrasound", response_model=WebsitePMOSProfileResponse)
def infer_ultrasound(
    payload: UltrasoundInferenceRequest,
    registry: ModelRegistry = Depends(get_registry),
) -> WebsitePMOSProfileResponse:
    """Ultrasound branch, gated off until the imaging audit is resolved.

    503 rather than an empty 200: the endpoint exists so the contract is stable,
    but an empty body would invite a client to render "no findings", which this
    branch cannot claim.
    """
    branch = registry.branch_status.get("ovarian_ultrasound")
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail={
            "message": "Ultrasound inference is not available.",
            "reason": (branch.reason if branch else None)
            or "The ultrasound branch is not validated for inference.",
            "validated_for_inference": bool(branch and branch.validated_for_inference),
        },
    )
