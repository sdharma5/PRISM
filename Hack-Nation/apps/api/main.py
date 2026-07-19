"""PRISM inference API.

Run with::

    uvicorn apps.api.main:app --reload --port 8000

Startup loads every configured encoder once and holds it for the process
lifetime. If the static clinical branch cannot be loaded, startup fails: it is
the only branch entitled to issue a whole-patient PCOS score, so a service
without it cannot do the job it exists for, and starting anyway would mean every
request returned a silently degraded result.
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from apps.api.registry import ModelRegistry, ModelRegistryError
from apps.api.routers import events as events_router
from apps.api.routers import inference as inference_router
from apps.api.routers import intake as intake_router
from apps.api.routers import jobs as jobs_router
from apps.api.routers import speech as speech_router
from apps.api.routers import system as system_router

logger = logging.getLogger(__name__)

#: Browser origins allowed to call this API. Defaults to the Next.js dev server.
#: An explicit list rather than "*": the response carries patient-derived
#: clinical content, so which origins may read it is a deliberate decision.
_DEFAULT_ORIGINS = ("http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:3001", "http://127.0.0.1:3001")


def _allowed_origins() -> list[str]:
    configured = os.environ.get("PRISM_CORS_ORIGINS")
    if not configured:
        return list(_DEFAULT_ORIGINS)
    return [origin.strip() for origin in configured.split(",") if origin.strip()]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load models once, then serve."""
    logging.basicConfig(level=os.environ.get("PRISM_LOG_LEVEL", "INFO"))
    logger.info("Loading PRISM model registry...")

    app.state.registry = ModelRegistry.load()
    app.state.event_store = events_router.build_event_store()
    app.state.jobs = {}

    for warning in app.state.registry.warnings:
        logger.warning("registry: %s", warning)
    logger.info("PRISM model registry ready.")

    yield

    # Nothing to release: the encoders are plain in-process objects and the
    # ledger is either in memory or already flushed to its JSONL file.
    logger.info("PRISM API shutting down.")


def create_app() -> FastAPI:
    """Application factory."""
    app = FastAPI(
        title="PRISM inference API",
        version="0.1.0",
        summary="Research-prototype PCOS evidence profiling. Not a diagnostic device.",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=_allowed_origins(),
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    app.include_router(system_router.router)
    app.include_router(inference_router.router)
    app.include_router(intake_router.router)
    app.include_router(events_router.router)
    app.include_router(jobs_router.router)
    app.include_router(speech_router.router)
    return app


app = create_app()

__all__ = ["ModelRegistryError", "app", "create_app"]
