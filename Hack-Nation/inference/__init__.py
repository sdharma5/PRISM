"""Multimodal inference for a new, unseen patient.

Independently-trained encoders process whatever data a patient provides; a
deterministic coordination layer organises their outputs by clinical domain,
preserves disagreement, and abstains where evidence is thin. No jointly trained
multimodal model exists or is implied -- see ``docs/decisions/ADR-002``.
"""

from __future__ import annotations

from inference.evidence_coordinator import EvidenceCoordinator, load_coordination_config
from inference.orchestrator import (
    Encoder,
    PatientInferenceOrchestrator,
    coordinate_only,
    run_patient_inference,
)
from inference.patient_bundle import (
    DocumentInput,
    PatientDataBundle,
    SpeechInput,
    TemporalInput,
    UltrasoundInput,
)
from inference.report_schema import CoordinatedEvidence, DomainEvidence, PatientEvidenceReport

__all__ = [
    "CoordinatedEvidence",
    "DocumentInput",
    "DomainEvidence",
    "Encoder",
    "EvidenceCoordinator",
    "PatientDataBundle",
    "PatientEvidenceReport",
    "PatientInferenceOrchestrator",
    "SpeechInput",
    "TemporalInput",
    "UltrasoundInput",
    "coordinate_only",
    "load_coordination_config",
    "run_patient_inference",
]
