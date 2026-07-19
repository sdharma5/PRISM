"""Shared dependencies: access to the process-wide model registry."""

from __future__ import annotations

from fastapi import HTTPException, Request, status

from apps.api.registry import ModelRegistry

__all__ = ["get_registry"]


def get_registry(request: Request) -> ModelRegistry:
    """Return the registry built at startup.

    A 503 here means the service is running but has no usable models -- which
    should be unreachable, because startup refuses to complete without the
    static branch. It is kept as a guard rather than an assertion so that a
    future degraded-start mode surfaces as a clear service-unavailable response
    instead of an ``AttributeError`` inside a request handler.
    """
    registry = getattr(request.app.state, "registry", None)
    if registry is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models are not loaded; the service is not ready to score patients.",
        )
    return registry
