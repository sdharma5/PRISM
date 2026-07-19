"""The shared output envelope every modality encoder must emit.

At this stage of the project tokens are exported *independently*. They are not
concatenated or fused into one network, because the underlying datasets describe
different people. See ``docs/decisions/ADR-002-no-fake-pairing.md``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

TokenModality = Literal[
    "static_clinical",
    "speech_symptoms",
    "clinical_document",
    "ovarian_ultrasound",
    "longitudinal_hormonal_state",
]

StructuredValue = float | int | str | bool | None


class ModalityToken(BaseModel):
    """Standardized, serializable encoder output."""

    patient_id: str
    modality: TokenModality

    embedding: list[float] = Field(default_factory=list)
    structured_features: dict[str, StructuredValue] = Field(default_factory=dict)

    quality_score: float = Field(ge=0.0, le=1.0)
    confidence_score: float = Field(ge=0.0, le=1.0)

    observed_at: str | None = None
    model_version: str = "0.1.0"
    source_dataset: str | None = None

    provenance_ids: list[str] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    def write_json(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.model_dump(mode="json"), indent=2) + "\n")
        return path

    @classmethod
    def read_json(cls, path: Path) -> ModalityToken:
        return cls.model_validate(json.loads(Path(path).read_text()))
