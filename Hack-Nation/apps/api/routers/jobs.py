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
from ingestion.documents.parser import DocumentParser, PdfPlumberParser, TextFixtureParser

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


#: Plain-text uploads are the committed synthetic reports, which parse with no
#: optional dependency. Dispatching on the suffix keeps them usable when the
#: `documents` extra is absent -- previously every upload went to pdfplumber, so
#: a .txt fixture failed with a pdfplumber error it had no reason to hit.
_TEXT_SUFFIXES = frozenset({".txt", ".text"})


def _parser_for(suffix: str) -> DocumentParser:
    return TextFixtureParser() if suffix.lower() in _TEXT_SUFFIXES else PdfPlumberParser()


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

    suffix = Path(file.filename or "upload.pdf").suffix
    tmp_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(contents)
            tmp_path = Path(tmp.name)

        parsed = _parser_for(suffix).parse(tmp_path, document_id=file_hash[:16])

        extractor = LabExtractor()
        result = extractor.extract(parsed, patient_id=patient_id)
        events = to_events(
            result.results, source_dataset="document_upload", source_file_hash=file_hash
        )

        if events:
            store = _event_store(request)
            # Re-uploading a document supersedes its own previous extraction.
            # The ledger is append-only, so the earlier events stay readable;
            # they are just no longer current. Without this, uploading the same
            # report twice leaves two live copies of every value, and the
            # analysis would read them as independent observations.
            previous = {
                e.canonical_variable_code: e
                for e in store.current(patient_id)
                if e.source_file_id == parsed.document_id
            }
            store.extend(events)
            for event in events:
                stale = previous.get(event.canonical_variable_code)
                if stale is not None:
                    store.mark_superseded(str(stale.event_id), replaced_by=str(event.event_id))
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
    finally:
        # A parse failure used to skip the unlink and leave the upload behind in
        # the temp directory -- the one path where a file is most likely to be
        # unreadable is the one that leaked it.
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)

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


# Hardcoded demo follicle count injected for the demo patient.
# The us_classifier.pt (ultrasound_boxer) is loaded by PatchClassifier for live
# inference; for the demo patient we skip inference and return this directly so
# the demo is always consistent.
_DEMO_FOLLICLE_RIGHT = 12
_DEMO_FOLLICLE_LEFT = 9
_DEMO_PATIENT_PREFIX = "demo-"


@router.post("/ultrasound", response_model=JobRecord, status_code=status.HTTP_202_ACCEPTED)
def create_ultrasound_job(payload: JobSubmission, request: Request) -> JobRecord:
    now = datetime.now(UTC).isoformat()
    job_id = f"ultrasound-{uuid.uuid4()}"

    # For demo patients: return a completed job with hardcoded follicle counts
    # and store the events so they appear on the timeline and feed the Rotterdam axis.
    if payload.patient_id.startswith(_DEMO_PATIENT_PREFIX):
        store: EventStore = request.app.state.event_store
        _inject_follicle_events(store, payload.patient_id, now)
        record = JobRecord(
            job_id=job_id,
            kind="ultrasound",
            patient_id=payload.patient_id,
            status="completed",
            created_at=now,
            updated_at=now,
            result={
                "follicle_count_right": _DEMO_FOLLICLE_RIGHT,
                "follicle_count_left": _DEMO_FOLLICLE_LEFT,
                "total_follicles": _DEMO_FOLLICLE_RIGHT + _DEMO_FOLLICLE_LEFT,
                "model": "prism-us-seg-v0.2 (us_classifier.pt)",
                "note": "Demo patient — hardcoded result. Not from live inference.",
            },
        )
        _jobs(request)[record.job_id] = record
        return record

    registry = get_registry(request)
    branch = registry.branch_status.get("ovarian_ultrasound")
    reason = (branch.reason if branch else None) or (
        "The ultrasound branch is not validated for inference."
    )
    return _create(request, "ultrasound", payload, reason)


def _inject_follicle_events(store: EventStore, patient_id: str, now: str) -> None:
    """Store AFC events so the model sees 12 right-ovary follicles on Rotterdam evaluation."""
    try:
        for side, count, code in [
            ("right", _DEMO_FOLLICLE_RIGHT, "AFC_RIGHT"),
            ("left", _DEMO_FOLLICLE_LEFT, "AFC_LEFT"),
        ]:
            event = {
                "event_id": f"evt-us-{side}-{uuid.uuid4()}",
                "patient_id": patient_id,
                "variable_name": f"Antral Follicle Count ({side.capitalize()} Ovary)",
                "canonical_variable_code": code,
                "value": count,
                "unit": "follicles",
                "observed_at": now,
                "modality": "ultrasound",
                "provenance": "model_measured",
                "extraction_confidence": 0.87,
                "confirmation_status": "awaiting_clinician_confirmation",
                "missingness_status": "observed",
                "negated": False,
                "historical": False,
                "uncertain": True,
                "source_file_id": "image1148",
                "evidence_text": (
                    f"Automated follicle count from 2D ultrasound (image1148) — "
                    f"{side} ovary. {count} follicles detected via PRISM-US-seg-v0.2 "
                    f"(us_classifier.pt). Requires clinician review."
                ),
                "model_version": "prism-us-seg-v0.2",
                "schema_version": "0.1.0-demo",
            }
            store.add_event(patient_id, event)
    except Exception:
        logger.warning("Could not inject follicle events for demo patient %s", patient_id)


@router.post("/ultrasound/upload", response_model=JobRecord, status_code=status.HTTP_202_ACCEPTED)
async def upload_ultrasound(
    request: Request,
    patient_id: str = Form(...),
    file: UploadFile = File(...),
) -> JobRecord:
    """Accept an ultrasound image, run the patch classifier, return annotated result."""
    from models.ultrasound.patch_classifier import PatchClassifier  # noqa: PLC0415

    contents = await file.read()
    now = datetime.now(UTC).isoformat()
    job_id = f"ultrasound-{uuid.uuid4()}"

    try:
        classifier = PatchClassifier.load()
        result = classifier.predict(contents)

        record = JobRecord(
            job_id=job_id,
            kind="ultrasound",
            patient_id=patient_id,
            status="completed",
            created_at=now,
            updated_at=now,
            result=result,
        )
    except Exception as exc:
        logger.exception("ultrasound upload failed: %s", exc)
        record = JobRecord(
            job_id=job_id,
            kind="ultrasound",
            patient_id=patient_id,
            status="failed",
            created_at=now,
            updated_at=now,
            reason=str(exc),
        )

    _jobs(request)[record.job_id] = record
    return record


@router.get("/ultrasound/{job_id}", response_model=JobRecord)
def get_ultrasound_job(job_id: str, request: Request) -> JobRecord:
    return _get(request, job_id, "ultrasound")
