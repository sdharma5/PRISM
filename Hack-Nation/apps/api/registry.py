"""Process-wide model registry. Loads the encoders once at startup.

Two rules:

* Load once. Reloading per request would make artifact availability a
  per-request property, so a deleted checkpoint would show up as sporadic
  failures rather than one clear one.
* Never substitute a heuristic for a trained model. ``build_encoders`` already
  refuses to; this adds the reporting side via ``/api/v1/models/status``.

An unavailable branch is fine. The static branch failing to load is not -- it is
the only source of a whole-patient score, so startup fails instead.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from evaluation.calibration import PlattCalibrator
from inference.encoder_registry import EncoderConfigurationError, build_encoders
from inference.evidence_coordinator import EvidenceCoordinator
from inference.orchestrator import PatientInferenceOrchestrator
from models.adapters.pmos.evidence_adapter import PmosEvidenceAdapter
from models.adapters.pmos.prototype_similarity import PrototypeSimilarityModel
from models.adapters.pmos.stability import PhenotypeStabilityEngine

__all__ = ["BranchStatus", "ModelRegistry", "ModelRegistryError"]

logger = logging.getLogger(__name__)

#: Config key -> the name the API reports the branch under.
_BRANCH_NAMES: dict[str, str] = {
    "static_clinical_encoder": "static_clinical",
    "temporal_encoder": "temporal_state",
    "ultrasound_encoder": "ovarian_ultrasound",
}

#: build_encoders() key for each config key.
_ENCODER_KEYS: dict[str, str] = {
    "static_clinical_encoder": "static_encoder",
    "temporal_encoder": "temporal_encoder",
    "ultrasound_encoder": "ultrasound_encoder",
}


class ModelRegistryError(RuntimeError):
    """Raised when the service cannot start with a defensible model set."""


@dataclass
class BranchStatus:
    """What the service can honestly say about one model branch.

    ``trained`` and ``validated_for_inference`` are separate because of
    ultrasound: the checkpoint loads fine, but its follicle Dice is
    oracle-assisted. Collapsing them would force it to be described as either
    absent or usable, and both are wrong.
    """

    available: bool
    trained: bool
    persisted: bool
    validated_for_inference: bool
    version: str | None = None
    implementation: str | None = None
    reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "available": self.available,
            "trained": self.trained,
            "persisted": self.persisted,
            "validated_for_inference": self.validated_for_inference,
        }
        if self.version is not None:
            payload["version"] = self.version
        if self.implementation is not None:
            payload["implementation"] = self.implementation
        if self.reason is not None:
            payload["reason"] = self.reason
        return payload


@dataclass
class ModelRegistry:
    """Everything the API needs to answer a request, built once."""

    orchestrator: PatientInferenceOrchestrator
    encoders: dict[str, Any]
    branch_status: dict[str, BranchStatus]
    config: dict[str, Any]
    repo_root: Path
    calibrator_loaded: bool = False
    warnings: list[str] = field(default_factory=list)

    # -- construction ------------------------------------------------------

    @classmethod
    def load(
        cls,
        *,
        repo_root: Path | None = None,
        config_path: Path | None = None,
        require_static: bool = True,
    ) -> ModelRegistry:
        """Build the registry from ``configs/models/inference_encoders.yaml``.

        Args:
            repo_root: Repository root; checkpoint paths in the config are
                resolved relative to it.
            config_path: Override for the encoder config.
            require_static: Refuse to start without the static branch. Only a
                test that is deliberately exercising the degraded path should
                pass False.

        Raises:
            ModelRegistryError: If the config is unusable, or if the static
                branch is required and did not load.
        """
        root = repo_root or _default_repo_root()
        path = config_path or root / "configs/models/inference_encoders.yaml"

        if not path.exists():
            raise ModelRegistryError(
                f"Encoder config not found at {path}. The API will not start with an "
                "implicit default: which checkpoint is loaded must be an explicit, "
                "reviewable decision."
            )

        config = yaml.safe_load(path.read_text()) or {}

        try:
            encoders = build_encoders(config)
        except EncoderConfigurationError as exc:
            # Not recoverable by substitution -- the caller asked for a
            # specific implementation and did not get it.
            raise ModelRegistryError(f"Encoder configuration rejected: {exc}") from exc

        warnings: list[str] = []
        status = _branch_status(config, encoders)

        static_encoder = encoders.get("static_encoder")
        if static_encoder is None:
            message = (
                "The static clinical encoder is not available. It is the only branch "
                "entitled to issue a whole-patient PMOS score, so the service cannot "
                "produce its primary output without it."
            )
            if require_static:
                raise ModelRegistryError(message)
            warnings.append(message)

        calibrator, calibrator_note = _load_calibrator(root, config)
        if calibrator_note:
            warnings.append(calibrator_note)

        adapter = PmosEvidenceAdapter(
            static_model=static_encoder,
            # Both ship working defaults. Omitting them yields empty
            # similarities and no stability verdict, which reads as "no
            # similarity found" rather than "never computed".
            prototype_model=PrototypeSimilarityModel(),
            stability_engine=PhenotypeStabilityEngine(),
            calibrator=calibrator,
        )

        orchestrator = PatientInferenceOrchestrator(
            static_encoder=encoders.get("static_encoder"),
            temporal_encoder=encoders.get("temporal_encoder"),
            ultrasound_encoder=encoders.get("ultrasound_encoder"),
            coordinator=EvidenceCoordinator(),
            adapter=adapter,
        )

        for name, branch in status.items():
            logger.info(
                "model branch %s: available=%s validated=%s version=%s",
                name,
                branch.available,
                branch.validated_for_inference,
                branch.version,
            )

        return cls(
            orchestrator=orchestrator,
            encoders=encoders,
            branch_status=status,
            config=config,
            repo_root=root,
            calibrator_loaded=calibrator is not None,
            warnings=warnings,
        )

    # -- queries -----------------------------------------------------------

    def status_payload(self) -> dict[str, Any]:
        """The body of ``GET /api/v1/models/status``."""
        return {name: branch.as_dict() for name, branch in self.branch_status.items()}

    def is_available(self, branch: str) -> bool:
        status = self.branch_status.get(branch)
        return bool(status and status.available)


def _default_repo_root() -> Path:
    """Repo root from this file's location, not the cwd -- the service should
    behave the same however it was started."""
    return Path(__file__).resolve().parents[2]


def _branch_status(config: dict[str, Any], encoders: dict[str, Any]) -> dict[str, BranchStatus]:
    """Describe each branch from the config and what actually got built."""
    status: dict[str, BranchStatus] = {}

    for config_key, name in _BRANCH_NAMES.items():
        section = config.get(config_key) or {}
        enabled = bool(section.get("enabled", False))
        built = encoders.get(_ENCODER_KEYS[config_key]) is not None
        implementation = section.get("implementation")
        version = section.get("model_version")

        # A disabled branch whose config still names a checkpoint is a
        # deliberate gate, not an absence -- report the difference.
        reason = _reason_for(config_key, section, enabled=enabled, built=built)

        status[name] = BranchStatus(
            available=built,
            trained=_looks_trained(implementation),
            persisted=bool(section.get("checkpoint_path")),
            validated_for_inference=built,
            version=version,
            implementation=implementation,
            reason=reason,
        )

    return status


def _looks_trained(implementation: str | None) -> bool:
    """``heuristic`` is the scipy threshold segmenter -- fine for CI, not a
    patient result, so it reports trained=False even though it loads."""
    if not implementation:
        return False
    return implementation != "heuristic"


def _reason_for(
    config_key: str, section: dict[str, Any], *, enabled: bool, built: bool
) -> str | None:
    if built:
        return None
    if config_key == "ultrasound_encoder" and not enabled:
        return (
            "Ovary segmentation head requires correction and held-out end-to-end "
            "evaluation. A checkpoint exists and loads, but its reported follicle "
            "Dice is oracle-assisted and is not a deployable patient result."
        )
    if not enabled:
        return "Disabled in configs/models/inference_encoders.yaml."
    return "Configured but not loaded; see service startup logs."


def _load_calibrator(
    root: Path, config: dict[str, Any]
) -> tuple[PlattCalibrator | None, str | None]:
    """Load the frozen Platt calibrator beside the static artifact.

    Returns ``(None, note)`` rather than raising: a raw score labelled as
    uncalibrated is still usable, and a secondary artifact should not take the
    service down. ``from_dict`` rejects a calibrator not fitted out-of-fold.
    """
    section = config.get("static_clinical_encoder") or {}
    checkpoint = section.get("checkpoint_path")
    if not checkpoint:
        return None, None

    path = root / checkpoint / "calibrator.json"
    if not path.exists():
        return None, (f"No calibrator at {path}; model scores are reported raw and uncalibrated.")

    try:
        calibrator = PlattCalibrator.from_dict(json.loads(path.read_text()))
    except (ValueError, json.JSONDecodeError) as exc:
        return None, (f"Calibrator at {path} was rejected ({exc}); model scores are reported raw.")
    return calibrator, None
