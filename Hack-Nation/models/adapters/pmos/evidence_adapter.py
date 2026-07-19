"""The PMOS interpretation layer over coordinated multimodal evidence.

This is the ``PMOSAdapter.predict(evidence) -> PMOSProfileOutput`` contract from
prompt_4. It is deliberately a *separate* class from the cohort-level
:class:`~models.adapters.pmos.adapter.PmosAdapter`, which discovers exploratory
profiles by clustering a whole cohort. The two answer different questions:

* ``PmosAdapter``          -- "what profiles exist in this cohort?" (fit on many)
* ``PmosEvidenceAdapter``  -- "what does this one patient's evidence show?"

Only the second runs at inference time for a new patient.

Composition of learned and rule-based parts, per prompt_4:

* **Part A, learned** -- the static clinical PMOS head, trained on the matched
  tabular cohort where symptoms, labs, history and derived measurements all
  belong to the same person. Genuinely supervised.
* **Part B, rule-based** -- everything crossing modality boundaries: guideline
  axis thresholds, coordinated domain weights, concordance assessment. Not
  learned, because no matched multimodal cohort exists to learn it from.

The adapter never blurs the two. ``learned_components_used`` and
``rule_based_components_used`` are populated on every output, and the schema
refuses a PMOS probability that is not backed by the learned head.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Protocol

from evaluation.calibration import PlattCalibrator
from features.phenotype_domains import load_domain_specs
from inference.report_schema import CoordinatedEvidence
from models.adapters.pmos.abstention import PmosAbstentionEngine
from models.adapters.pmos.evidence_rules import PmosEvidenceRules
from models.adapters.pmos.explanation import build_explanation
from models.adapters.pmos.feature_mapper import PmosFeatureMapper
from models.adapters.pmos.profile_output import (
    AxisEvidenceOutput,
    PMOSProfileOutput,
    PhenotypeDomainDetail,
)
from models.adapters.pmos.prototype_similarity import androgenic_evidence_source, summarize

__all__ = ["PmosEvidenceAdapter", "StaticPmosHead"]


@lru_cache(maxsize=1)
def _phenotype_domain_names() -> frozenset[str]:
    """Domain names the registry defines, used to read composites from a token."""
    return frozenset(load_domain_specs())


def _domain_detail(
    static: Any, domain_scores: dict[str, float | None]
) -> dict[str, PhenotypeDomainDetail]:
    """Coverage, labels and per-variable observation for each domain score.

    Coverage comes from the token. Observed/missing are derived here because
    ``structured_features`` is a flat scalar map and can't carry a list -- but it
    does carry every supplied variable, so membership is an exact check.
    """
    specs = load_domain_specs()
    features: dict[str, Any] = dict(static.structured_features) if static is not None else {}

    detail: dict[str, PhenotypeDomainDetail] = {}
    for name, score in domain_scores.items():
        spec = specs.get(name)
        codes = [f.code for f in spec.features] if spec is not None else []

        observed = [code for code in codes if isinstance(features.get(code), int | float | bool)]
        missing = [code for code in codes if code not in observed]

        # Applies only when every observed variable is report-class, mirroring
        # compute_domain_scores. Attaching it whenever anything was observed
        # stamps "symptoms only" onto a domain carrying a measured assay.
        observed_classes = {
            f.evidence_class
            for f in (spec.features if spec is not None else [])
            if f.code in observed
        }
        symptoms_only = bool(observed_classes) and observed_classes <= {"report"}

        coverage = features.get(f"{name}_coverage")
        detail[name] = PhenotypeDomainDetail(
            domain=name,
            label=getattr(spec, "label", None),
            score=score,
            coverage=float(coverage) if isinstance(coverage, int | float) else None,
            assessable=score is not None,
            evidence_source=getattr(spec, "evidence_source", "mixed"),
            evidence_qualifier=(
                getattr(spec, "symptom_only_qualifier", None) if symptoms_only else None
            ),
            observed_variables=observed,
            missing_variables=missing,
        )
    return detail


def _domain_evidence_sources(domain_scores: dict[str, float | None]) -> dict[str, str]:
    """Domain -> declared evidence kind, from the registry.

    Carried on every output so a consumer rendering a domain score never has to
    guess whether it came from a cutaneous sign, an assay, or an image.
    """
    try:
        specs = load_domain_specs()
    except Exception:  # noqa: BLE001 - a missing registry must not fail a patient
        return {}
    return {name: specs[name].evidence_source for name in domain_scores if name in specs}


def _stability_payload(report: Any) -> dict[str, Any]:
    """Serialize a stability report, or record that none was computed."""
    if report is None:
        return {
            "available": False,
            "reason": (
                "No stability engine was configured, so the phenotype assignment "
                "carries no fragility estimate and should be read as provisional."
            ),
        }
    return {
        "available": True,
        "bootstrap_agreement": round(report.bootstrap_agreement, 4),
        "n_bootstrap": report.n_bootstrap,
        "profile_flip_rate": round(report.profile_flip_rate, 4),
        "stability_score": round(report.stability_score, 4),
        "is_stable": report.is_stable,
        "n_observed_domains": report.n_observed_domains,
        "meets_minimum_domains": report.meets_minimum_domains,
        "domain_ablation": report.domain_ablation,
        "unstable_domains": report.unstable_domains,
        "modality_removal": report.modality_removal,
        "unstable_modalities": report.unstable_modalities,
        "temperature_sensitivity": report.temperature_sensitivity,
        "threshold_sensitivity": report.threshold_sensitivity,
        "abstain_from_profile": report.abstain_from_profile,
        "abstention_reason": report.abstention_reason,
        "note": (
            "Affinity scores are not calibrated probabilities. These statistics "
            "describe how fragile the ranking is, not how likely it is to be correct."
        ),
    }


class StaticPmosHead(Protocol):
    """The trained static clinical model, as the adapter needs it."""

    def predict_proba_from_features(self, values: dict[str, Any]) -> float:
        """Return P(PMOS) for one patient's canonical clinical variables."""
        ...


class PmosEvidenceAdapter:
    """Interpret coordinated evidence as a PMOS-specific profile."""

    model_version = "pmos-evidence-adapter-0.1.0"

    def __init__(
        self,
        *,
        static_model: StaticPmosHead | None = None,
        feature_mapper: PmosFeatureMapper | None = None,
        evidence_rules: PmosEvidenceRules | None = None,
        abstention_engine: PmosAbstentionEngine | None = None,
        prototype_model: Any | None = None,
        stability_engine: Any | None = None,
        calibrator: PlattCalibrator | None = None,
    ) -> None:
        """
        Args:
            static_model: Trained static clinical PMOS head. When None, the
                adapter still reports axis-level evidence but abstains from any
                whole-patient PMOS probability.
            feature_mapper: Token-to-variable mapper.
            evidence_rules: Guideline axis evaluator.
            abstention_engine: Insufficient-evidence policy.
            prototype_model: Optional research-prototype similarity model.
            stability_engine: Optional stability/flip-rate estimator.
            calibrator: Optional frozen Platt calibrator, fitted elsewhere on
                out-of-fold TRAINING predictions. The adapter only ever applies
                it; it has no fitting path, so an inference-time calibrator can
                never be fitted on the patients it scores.
        """
        self.static_model = static_model
        self.feature_mapper = feature_mapper or PmosFeatureMapper()
        self.evidence_rules = evidence_rules or PmosEvidenceRules()
        self.abstention_engine = abstention_engine or PmosAbstentionEngine()
        self.prototype_model = prototype_model
        self.stability_engine = stability_engine
        self.calibrator = calibrator

    # -- internals ---------------------------------------------------------

    def _calibrate(self, probability: float | None) -> float | None:
        """Apply the frozen calibrator, or return None when there is none."""
        if probability is None or self.calibrator is None or not self.calibrator.is_fitted:
            return None
        return float(self.calibrator.transform([probability])[0])

    def _static_probability(self, mapped: Any) -> tuple[float | None, list[str]]:
        """Run the learned head, degrading to None rather than raising."""
        if self.static_model is None:
            return None, []
        if "static_clinical" not in mapped.available_modalities:
            # The head is trained on clinical variables. Feeding it a patient
            # whose only evidence is an ultrasound would be extrapolation past
            # anything it saw in training.
            return None, []
        try:
            probability = float(self.static_model.predict_proba_from_features(mapped.values))
        except Exception as exc:  # noqa: BLE001 - a failed head must not sink the profile
            mapped.warnings.append(f"Static PMOS head failed and was skipped: {exc}")
            return None, []
        return max(0.0, min(1.0, probability)), ["static_clinical.pmos_head"]

    # -- public API --------------------------------------------------------

    def predict(self, evidence: CoordinatedEvidence) -> PMOSProfileOutput:
        """Produce a PMOS profile from coordinated evidence.

        Args:
            evidence: Output of the evidence coordinator.

        Returns:
            A profile, possibly abstaining, always declaring its method.
        """
        mapped = self.feature_mapper.transform(evidence)
        diagnostic = self.evidence_rules.evaluate(mapped)
        probability, learned = self._static_probability(mapped)

        decision = self.abstention_engine.evaluate(
            mapped=mapped,
            diagnostic_features=diagnostic,
            static_prediction=probability,
        )

        rule_based = [
            "pmos_adapter.guideline_axis_thresholds",
            "evidence_coordinator.design_rule_weights",
        ]

        # Continuous phenotype domain scores, read from the static token where the
        # domain scorer already produced them against TRAINING cohort statistics.
        # Recomputing here would need reference stats this object does not have.
        # Gated on registry domain names, not the `_score` suffix -- the registry
        # also has variables ending in `_score` (`ferriman_gallwey_score`), and
        # their raw values do not belong in a dict of cohort z-scores.
        known_domains = _phenotype_domain_names()
        domain_scores: dict[str, float | None] = {}
        static = evidence.static_token
        if static is not None:
            for key, value in static.structured_features.items():
                if not key.endswith("_score") or key == "pmos_evidence_probability":
                    continue
                name = key[: -len("_score")]
                if name not in known_domains:
                    continue
                domain_scores[name] = float(value) if isinstance(value, int | float) else None

        domain_detail = _domain_detail(static, domain_scores)

        phenotype: dict[str, float] = {}
        similarity = None
        if self.prototype_model is not None and domain_scores:
            try:
                similarity = self.prototype_model.predict(domain_scores)
                phenotype = dict(similarity.affinities)
                rule_based.append("pmos_adapter.prototype_similarity")
            except Exception as exc:  # noqa: BLE001
                mapped.warnings.append(f"Prototype similarity failed and was skipped: {exc}")

        stability_score = 0.0
        flip_rate: float | None = None
        stability_report = None
        is_stable: bool | None = None
        if self.stability_engine is not None and domain_scores and self.prototype_model is not None:
            try:
                stability_report = self.stability_engine.evaluate(
                    domain_scores,
                    self.prototype_model,
                    available_modalities=list(mapped.available_modalities),
                )
                stability_score = float(stability_report.stability_score)
                flip_rate = float(stability_report.profile_flip_rate)
                rule_based.append("pmos_adapter.stability_engine")
                is_stable = bool(
                    stability_report.is_stable and not stability_report.abstain_from_profile
                )
                mapped.warnings.extend(stability_report.warnings)
            except Exception as exc:  # noqa: BLE001
                mapped.warnings.append(f"Stability engine failed and was skipped: {exc}")

        # An unstable assignment must not be reported as a finding. The affinities
        # stay visible -- withholding them entirely would hide that the patient
        # sits between profiles -- but the dominant label is withdrawn, because
        # that is the part a reader would quote. `summarize` applies the gate and
        # records the reason, so the withdrawal is never silent.
        phenotype_summary: dict[str, Any] = {}
        if similarity is not None:
            phenotype_summary = summarize(similarity, is_stable=is_stable)
            dominant = phenotype_summary["dominant_profile"]
        else:
            dominant = None

        axis_outputs = {
            axis: AxisEvidenceOutput(
                axis=item.axis,
                level=item.level,
                score=item.score,
                axis_status=item.axis_status,
                supporting_evidence=item.supporting_evidence,
                missing_evidence=item.missing_evidence,
                threshold_sources=item.threshold_sources,
                assay_dependent=item.assay_dependent,
                caveats=item.caveats,
                notes=item.notes,
            )
            for axis, item in diagnostic.items()
        }

        agreements = [
            f"{name}: {domain.agreement}"
            for name, domain in mapped.domain_evidence.items()
            if domain.agreement in ("strong", "moderate")
        ]
        conflicts = [conflict.detail for conflict in mapped.conflicts]
        conflicts.extend(
            note for domain in mapped.domain_evidence.values() for note in domain.notes
        )

        missing_evidence = sorted(
            {code for item in diagnostic.values() for code in item.missing_evidence}
        )

        warnings = list(dict.fromkeys(mapped.warnings))
        warnings.append(
            "Cross-modal interpretation is rule-based. No jointly trained multimodal "
            "model was used, and no accuracy figure applies to the combined result."
        )
        if mapped.assay_dependent_present:
            warnings.append(
                "Assay-dependent androgen values were used with placeholder reference "
                f"ranges: {', '.join(mapped.assay_dependent_present)}."
            )

        explanation = build_explanation(
            mapped=mapped,
            diagnostic=diagnostic,
            similarity=similarity,
            static_probability=probability,
            decision=decision,
            stability=_stability_payload(stability_report),
        )

        return PMOSProfileOutput(
            patient_id=evidence.patient_id,
            pmos_evidence_probability=None if decision.abstain else probability,
            diagnostic_feature_evidence=axis_outputs,
            raw_model_score=probability,
            calibrated_model_score=self._calibrate(probability),
            phenotype_domain_scores=domain_scores,
            phenotype_domain_detail=domain_detail,
            domain_assessability={name: value is not None for name, value in domain_scores.items()},
            domain_evidence_source=_domain_evidence_sources(domain_scores),
            androgenic_evidence_source=androgenic_evidence_source(domain_scores),
            phenotype_affinities=phenotype,
            dominant_profile=dominant,
            indeterminate=bool(phenotype_summary.get("indeterminate", True)),
            assignment_is_stable=is_stable,
            profile_similarities=(
                {k: round(v, 4) for k, v in similarity.similarities.items()}
                if similarity is not None
                else {}
            ),
            eligible_profiles=list(phenotype_summary.get("eligible_profiles", [])),
            ineligible_profiles=dict(phenotype_summary.get("ineligible_profiles", {})),
            assignment_entropy=(round(similarity.entropy, 4) if similarity is not None else None),
            indeterminate_reasons=list(phenotype_summary.get("indeterminate_reasons", [])),
            profile_supporting_domains=(
                dict(similarity.supporting_evidence) if similarity is not None else {}
            ),
            explanation=explanation,
            stability_score=stability_score,
            profile_stability=_stability_payload(stability_report),
            subtype_flip_rate=flip_rate,
            abstain=decision.abstain,
            abstention_reason=decision.reason,
            available_modalities=mapped.available_modalities,
            missing_modalities=mapped.missing_modalities,
            agreements=agreements,
            conflicts=conflicts,
            missing_evidence=missing_evidence,
            learned_components_used=learned,
            rule_based_components_used=rule_based,
            provenance_ids=list(evidence.provenance_ids),
            warnings=list(dict.fromkeys(warnings)),
        )
