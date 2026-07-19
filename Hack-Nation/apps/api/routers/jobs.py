"""Ingestion jobs: documents, speech, ultrasound.

Models the lifecycle a client needs (submit, poll, read) without pretending the
extraction pipelines are wired in. Jobs that cannot run park in a terminal
``unavailable`` state naming what is missing, rather than sitting at
``processing`` forever or returning fabricated extractions.

Ultrasound names the inference gate rather than an absence -- its checkpoint
exists and loads.
"""

from __future__ import annotations

import hashlib
import logging
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, status
from pydantic import BaseModel, ConfigDict, Field

from apps.api.deps import get_registry
from event_store.store import EventStore
from ingestion.documents.lab_extractor import LabExtractor, to_events
from ingestion.documents.parser import PdfPlumberParser

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/jobs", tags=["jobs"])

JobKind = Literal["documents", "speech", "ultrasound"]
JobStatus = Literal["queued", "processing", "completed", "failed", "unavailable"]


class JobSubmission(BaseModel):
    """What a client sends to open a job."""

    model_config = ConfigDict(extra="forbid")

    patient_id: str = Field(min_length=1)
    #: Opaque reference to already-uploaded content. Bytes are not accepted on
    #: this route: a large upload belongs on its own transport, and accepting it
    #: here would make the request timeout the de facto size limit.
    source_ids: list[str] = Field(default_factory=list)
    note: str | None = None


class JobRecord(BaseModel):
    """A job's observable state."""

    model_config = ConfigDict(extra="forbid")

    job_id: str
    kind: JobKind
    patient_id: str
    status: JobStatus
    created_at: str
    updated_at: str
    reason: str | None = None
    result: dict[str, Any] | None = None


#: Why each kind of job cannot complete yet. Stated per kind so a client can
#: show a specific message rather than a generic failure.
_UNAVAILABLE_REASONS: dict[str, str] = {
    "documents": (
        "Document extraction is not connected to this service. Documents can be "
        "submitted, but no extraction is performed and no events are produced."
    ),
    "speech": (
        "Speech extraction is not connected to this service. Recordings can be "
        "submitted, but no transcription or extraction is performed."
    ),
}


def _jobs(request: Request) -> dict[str, JobRecord]:
    store = getattr(request.app.state, "jobs", None)
    if store is None:  # pragma: no cover - startup always sets this
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The job store is not initialised.",
        )
    return store


def _create(request: Request, kind: JobKind, payload: JobSubmission, reason: str) -> JobRecord:
    now = datetime.now(UTC).isoformat()
    record = JobRecord(
        job_id=f"{kind}-{uuid.uuid4()}",
        kind=kind,
        patient_id=payload.patient_id,
        status="unavailable",
        created_at=now,
        updated_at=now,
        reason=reason,
    )
    _jobs(request)[record.job_id] = record
    return record


def _get(request: Request, job_id: str, kind: JobKind) -> JobRecord:
    record = _jobs(request).get(job_id)
    if record is None or record.kind != kind:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"No {kind} job with id '{job_id}'."
        )
    return record


@router.post("/documents", response_model=JobRecord, status_code=status.HTTP_202_ACCEPTED)
def create_document_job(payload: JobSubmission, request: Request) -> JobRecord:
    return _create(request, "documents", payload, _UNAVAILABLE_REASONS["documents"])


def _event_store(request: Request) -> EventStore:
    store = getattr(request.app.state, "event_store", None)
    if store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Event store not initialised."
        )
    return store


@router.post("/documents/upload", response_model=JobRecord, status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    request: Request,
    patient_id: str = Form(...),
    file: UploadFile = File(...),
) -> JobRecord:
    """Accept a PDF or image lab report, extract lab values, store as events."""
    contents = await file.read()
    file_hash = hashlib.sha256(contents).hexdigest()
    now = datetime.now(UTC).isoformat()
    job_id = f"documents-{uuid.uuid4()}"

    try:
        with tempfile.NamedTemporaryFile(
            suffix=Path(file.filename or "upload.pdf").suffix, delete=False
        ) as tmp:
            tmp.write(contents)
            tmp_path = Path(tmp.name)

        parser = PdfPlumberParser()
        parsed = parser.parse(tmp_path, document_id=file_hash[:16])
        tmp_path.unlink(missing_ok=True)

        extractor = LabExtractor()
        result = extractor.extract(parsed, patient_id=patient_id)
        events = to_events(
            result.results, source_dataset="document_upload", source_file_hash=file_hash
        )

        if events:
            _event_store(request).extend(events)
            logger.info(
                "document upload: extracted %d events from %s for patient %s",
                len(events),
                file.filename,
                patient_id,
            )

        record = JobRecord(
            job_id=job_id,
            kind="documents",
            patient_id=patient_id,
            status="completed",
            created_at=now,
            updated_at=now,
            reason=None,
            result={
                "extracted": len(result.results),
                "unsupported": len(result.unsupported),
                "events_stored": len(events),
                "warnings": result.warnings,
            },
        )
    except Exception as exc:
        logger.exception("document upload failed: %s", exc)
        record = JobRecord(
            job_id=job_id,
            kind="documents",
            patient_id=patient_id,
            status="failed",
            created_at=now,
            updated_at=now,
            reason=str(exc),
        )

    _jobs(request)[record.job_id] = record
    return record


@router.get("/documents/{job_id}", response_model=JobRecord)
def get_document_job(job_id: str, request: Request) -> JobRecord:
    return _get(request, job_id, "documents")


@router.post("/speech", response_model=JobRecord, status_code=status.HTTP_202_ACCEPTED)
def create_speech_job(payload: JobSubmission, request: Request) -> JobRecord:
    return _create(request, "speech", payload, _UNAVAILABLE_REASONS["speech"])


@router.get("/speech/{job_id}", response_model=JobRecord)
def get_speech_job(job_id: str, request: Request) -> JobRecord:
    return _get(request, job_id, "speech")


@router.post("/ultrasound", response_model=JobRecord, status_code=status.HTTP_202_ACCEPTED)
def create_ultrasound_job(payload: JobSubmission, request: Request) -> JobRecord:
    """Accept an ultrasound study; report the inference gate as the reason.

    Uploads are accepted so the interface can be built and exercised. The job
    then terminates as ``unavailable`` carrying the same reason
    ``/models/status`` gives, so the UI never has to compose its own explanation
    for why an imaging result did not arrive.
    """
    registry = get_registry(request)
    branch = registry.branch_status.get("ovarian_ultrasound")
    reason = (branch.reason if branch else None) or (
        "The ultrasound branch is not validated for inference."
    )
    return _create(request, "ultrasound", payload, reason)


@router.get("/ultrasound/{job_id}", response_model=JobRecord)
def get_ultrasound_job(job_id: str, request: Request) -> JobRecord:
    return _get(request, job_id, "ultrasound")
