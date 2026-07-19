"""Run whichever encoders a new patient's data supports, then coordinate them.

This is the entry point for inference on an unseen patient. It owns exactly one
decision -- *which branches can run* -- and delegates everything else. Encoders
are injected rather than imported, so this module never depends on torch, on a
particular checkpoint, or on any encoder being trained at all.

Missing modalities are the normal case, not an error path. A patient with only a
questionnaire gets a static-only report; a patient with only an ultrasound gets
morphology evidence and no whole-patient PMOS statement. Both are legitimate
outputs, which is why every branch is guarded rather than required.

An encoder that raises is recorded as a warning and its branch is dropped. One
failing encoder must not deny the patient the report the other branches can
still support.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from inference.evidence_coordinator import EvidenceCoordinator
from inference.patient_bundle import PatientDataBundle
from inference.report_schema import CoordinatedEvidence, PatientEvidenceReport
from schemas.modality_token import ModalityToken

__all__ = ["Encoder", "PatientInferenceOrchestrator", "run_patient_inference"]


@runtime_checkable
class Encoder(Protocol):
    """The contract every modality encoder satisfies."""

    def export_token(self, payload: Any, *, patient_id: str) -> ModalityToken:
        """Run the encoder and return its token."""
        ...


class PatientInferenceOrchestrator:
    """Route a bundle through the available encoders and coordinate the result."""

    def __init__(
        self,
        *,
        static_encoder: Encoder | None = None,
        ultrasound_encoder: Encoder | None = None,
        temporal_encoder: Encoder | None = None,
        event_extractors: list[Any] | None = None,
        coordinator: EvidenceCoordinator | None = None,
        adapter: Any | None = None,
    ) -> None:
        """
        Args:
            static_encoder: Consumes clinical events. The only branch whose head
                is trained on labelled PMOS data.
            ultrasound_encoder: Consumes :class:`UltrasoundInput`.
            temporal_encoder: Consumes :class:`TemporalInput`.
            event_extractors: Speech/document extractors converting raw input to
                clinical events. They are ingestion, never their own branch.
            coordinator: Evidence coordinator; constructed by default.
            adapter: Optional PMOS adapter run on the coordinated evidence.
        """
        self.static_encoder = static_encoder
        self.ultrasound_encoder = ultrasound_encoder
        self.temporal_encoder = temporal_encoder
        self.event_extractors = event_extractors or []
        self.coordinator = coordinator or EvidenceCoordinator()
        self.adapter = adapter

    # -- ingestion ---------------------------------------------------------

    def _ingest_to_events(self, bundle: PatientDataBundle) -> tuple[list[Any], list[str]]:
        """Convert speech and documents into clinical events.

        Returns the bundle's own events plus everything extracted, and any
        warnings raised along the way.
        """
        events = list(bundle.clinical_events)
        warnings: list[str] = []

        raw_inputs = [*bundle.speech_recordings, *bundle.documents]
        if raw_inputs and not self.event_extractors:
            warnings.append(
                f"{len(raw_inputs)} speech/document input(s) supplied but no extractor is "
                "configured; their content did NOT reach the static clinical encoder."
            )
            return events, warnings

        for extractor in self.event_extractors:
            for raw in raw_inputs:
                try:
                    events.extend(extractor.extract(raw, patient_id=bundle.patient_id))
                except Exception as exc:  # noqa: BLE001 - one bad input must not sink the report
                    warnings.append(
                        f"Extractor {type(extractor).__name__} failed on one input: {exc}"
                    )
        return events, warnings

    # -- public API --------------------------------------------------------

    def run(self, bundle: PatientDataBundle, *, mode: str = "rule_based") -> PatientEvidenceReport:
        """Run inference for one patient.

        Args:
            bundle: The patient's supplied data.
            mode: Combination mode passed to the coordinator.

        Returns:
            A report carrying every token produced, the coordinated domain
            summary, and an explicit account of what was learned versus ruled.
        """
        tokens: list[ModalityToken] = []
        warnings: list[str] = []
        learned: list[str] = []
        rule_based: list[str] = ["evidence_coordinator.design_rule_weights"]

        events, ingest_warnings = self._ingest_to_events(bundle)
        warnings.extend(ingest_warnings)

        def _run(encoder: Encoder | None, payload: Any, label: str) -> None:
            if encoder is None or payload is None:
                return
            try:
                tokens.append(encoder.export_token(payload, patient_id=bundle.patient_id))
            except Exception as exc:  # noqa: BLE001 - degrade to fewer branches, never crash
                warnings.append(f"{label} encoder failed and was skipped: {exc}")

        if events:
            _run(self.static_encoder, events, "static_clinical")
            if any(token.modality == "static_clinical" for token in tokens):
                learned.append("static_clinical.pmos_head")
        elif bundle.has_static_input():
            warnings.append(
                "Speech/document input was supplied but produced no clinical events; "
                "the static clinical branch did not run."
            )

        _run(self.ultrasound_encoder, bundle.ultrasound_inputs or None, "ovarian_ultrasound")

        # When no pixel-based ultrasound ran but clinical events carry ultrasound
        # measurements (e.g. follicle count from the patch classifier), synthesise
        # a minimal ovarian_ultrasound token so the domain mapper and feature mapper
        # can apply the PCOM threshold rules.
        _US_PASSTHROUGH = {"follicle_number_per_ovary", "ovary_volume_ml"}
        if not any(t.modality == "ovarian_ultrasound" for t in tokens):
            us_features = {
                e.canonical_variable_code: float(e.value)
                for e in events
                if e.canonical_variable_code in _US_PASSTHROUGH
                and isinstance(e.value, (int, float))
            }
            if us_features:
                tokens.append(
                    ModalityToken(
                        patient_id=bundle.patient_id,
                        modality="ovarian_ultrasound",
                        structured_features=us_features,
                        quality_score=0.9,
                        confidence_score=0.9,
                        provenance_ids=["patch_classifier"],
                        warnings=["Derived from patch classifier count, not a radiologist AFC."],
                    )
                )
                rule_based.append("ultrasound.pcom_threshold_rules")

        _run(self.temporal_encoder, bundle.temporal_series, "longitudinal_hormonal_state")

        if any(token.modality == "ovarian_ultrasound" for token in tokens):
            rule_based.append("ultrasound.pcom_threshold_rules")
        if not tokens:
            warnings.append(
                "No encoder produced a token. The report contains no evidence and no "
                "domain may be interpreted."
            )

        evidence = self.coordinator.combine(tokens, patient_id=bundle.patient_id, mode=mode)
        warnings.extend(evidence.warnings)

        pmos_profile: dict[str, Any] = {}
        if self.adapter is not None and tokens:
            try:
                profile = self.adapter.predict(evidence)
                pmos_profile = (
                    profile.model_dump(mode="json") if hasattr(profile, "model_dump") else profile
                )
                rule_based.append("pmos_adapter.guideline_axes")
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"PMOS adapter failed and was skipped: {exc}")

        agreements = [
            f"{name}: {domain.agreement}"
            for name, domain in evidence.domain_evidence.items()
            if domain.agreement in ("strong", "moderate")
        ]
        conflicts = [note for domain in evidence.domain_evidence.values() for note in domain.notes]

        return PatientEvidenceReport(
            patient_id=bundle.patient_id,
            available_modalities=evidence.available_modalities,
            missing_modalities=evidence.missing_modalities,
            coverage=evidence.coverage,
            tokens={token.modality: token for token in tokens},
            domain_summary=evidence.domain_evidence,
            pmos_profile=pmos_profile,
            combination_mode=evidence.combination_mode,
            learned_components_used=learned,
            rule_based_components_used=rule_based,
            joint_model_used=False,
            agreements=agreements,
            conflicts=conflicts,
            provenance_ids=evidence.provenance_ids,
            warnings=list(dict.fromkeys(warnings)),
        )


def run_patient_inference(
    bundle: PatientDataBundle,
    *,
    static_encoder: Encoder | None = None,
    ultrasound_encoder: Encoder | None = None,
    temporal_encoder: Encoder | None = None,
    adapter: Any | None = None,
    mode: str = "rule_based",
) -> PatientEvidenceReport:
    """Convenience wrapper matching the prompt_4 signature."""
    orchestrator = PatientInferenceOrchestrator(
        static_encoder=static_encoder,
        ultrasound_encoder=ultrasound_encoder,
        temporal_encoder=temporal_encoder,
        adapter=adapter,
    )
    return orchestrator.run(bundle, mode=mode)


def coordinate_only(
    tokens: list[ModalityToken], *, mode: str = "rule_based"
) -> CoordinatedEvidence:
    """Coordinate pre-computed tokens without re-running any encoder."""
    return EvidenceCoordinator().combine(tokens, mode=mode)
