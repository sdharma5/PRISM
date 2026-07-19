"""Convert a :class:`PatientEvidenceReport` into the website contract.

The one place model output becomes something a person reads, so:

1. Never invent availability. A branch that didn't run says so, with a reason.
2. Qualifiers travel with the values they qualify -- symptoms-only androgenic
   evidence carries that fact in the same object as the score.
3. ``None`` survives. A domain never assessed must not arrive as 0.0, which
   reads as "exactly average".

The evidence bands are a presentation device, not a model output: a bare 0.696
invites "69.6% chance of PCOS".
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from functools import lru_cache
from typing import Any

from apps.api.schemas.responses import (
    AxisView,
    ConflictView,
    CurrentStateView,
    DomainScoreView,
    EvidenceStatementView,
    HormoneEstimateView,
    PcosAssessmentView,
    PhenotypeView,
    ProvenanceRecordView,
    ProvenanceView,
    StabilityView,
    WebsitePCOSProfileResponse,
)
from inference.report_schema import PatientEvidenceReport
from registry.loader import load_variable_registry

__all__ = ["evidence_level_for", "to_website_response", "translate_method"]

#: Upper-exclusive, walked in order. Bands rather than percentages: calibration
#: is only trustworthy where the reliability bins are populated.
_EVIDENCE_BANDS: tuple[tuple[float, str], ...] = (
    (0.25, "low"),
    (0.50, "moderate"),
    (0.75, "elevated"),
    (1.01, "high"),
)

#: LOCF must never read as a prediction -- it carries the last observed value
#: forward unchanged.
_METHOD_TRANSLATIONS: dict[str, str] = {
    "locf": "Based on the latest observed value",
    "ridge_window": "Estimated from the recent measurement pattern",
    "logistic": "Classified from recent longitudinal measurements",
}

#: Internal key -> (canonical code, display name, unit). Keyed by canonical code
#: so estimates join to events and registry entries.
_HORMONES: dict[str, tuple[str, str, str]] = {
    "lh": ("urinary_lh", "Luteinising hormone", "mIU/mL"),
    "e3g": ("e3g", "Estrone-3-glucuronide", "ng/mL"),
    "pdg": ("pdg", "Pregnanediol glucuronide", "ug/mL"),
}

#: Display order. Unlisted domains sort last, so new ones still render.
_DOMAIN_ORDER: tuple[str, ...] = (
    "reproductive",
    "clinical_androgenic_evidence",
    "biochemical_androgenic_evidence",
    "metabolic",
    "ovarian",
    "lh_amh_pattern",
    "symptom_burden",
)

_CYCLE_PHASE_PREFIX = "cycle_phase_probability_"


def evidence_level_for(score: float | None) -> str:
    """``None`` maps to ``not_available``, never ``low`` -- "low evidence" is a
    finding, "not available" is the absence of one."""
    if score is None:
        return "not_available"
    for upper, label in _EVIDENCE_BANDS:
        if score < upper:
            return label
    return "high"


def translate_method(code: str | None) -> str | None:
    """Unknown codes pass through rather than borrowing another method's wording."""
    if not code:
        return None
    return _METHOD_TRANSLATIONS.get(code, code)


def to_website_response(
    report: PatientEvidenceReport,
    *,
    generated_at: datetime | None = None,
    report_id: str | None = None,
) -> WebsitePCOSProfileResponse:
    """Map one internal report onto the website contract."""
    profile: dict[str, Any] = report.pcos_profile or {}
    stamp = generated_at or datetime.now(UTC)

    return WebsitePCOSProfileResponse(
        report_id=report_id or _report_id(report, stamp),
        patient_id=report.patient_id,
        generated_at=stamp.isoformat(),
        modality_coverage=_as_float(report.coverage),
        pcos_assessment=_assessment(profile, report),
        rotterdam_axes=_axes(profile),
        phenotype=_phenotype(profile),
        current_state=_current_state(report),
        androgenic_evidence_source=profile.get("androgenic_evidence_source"),
        supporting_evidence=_supporting_evidence(profile),
        conflicting_evidence=[_conflict(c) for c in (profile.get("conflicts") or [])],
        missing_evidence=list(profile.get("missing_evidence") or []),
        available_modalities=list(report.available_modalities),
        missing_modalities=list(report.missing_modalities),
        learned_components_used=list(report.learned_components_used),
        rule_based_components_used=list(report.rule_based_components_used),
        provenance=_provenance(report),
        # The report and the profile both carry the coordinator's warnings.
        warnings=list(
            dict.fromkeys(
                [str(w) for w in report.warnings]
                + [str(w) for w in (profile.get("warnings") or [])]
            )
        ),
    )


def _report_id(report: PatientEvidenceReport, stamp: datetime) -> str:
    """Derived from patient + timestamp so it's reproducible, hashed so it
    doesn't leak the patient id into logs."""
    digest = hashlib.sha256(f"{report.patient_id}|{stamp.isoformat()}".encode()).hexdigest()
    return f"rpt_{digest[:16]}"


# -- sections --------------------------------------------------------------


def _assessment(profile: dict[str, Any], report: PatientEvidenceReport) -> PcosAssessmentView:
    """The learned score, or an explicit statement of why there isn't one."""
    raw = profile.get("raw_model_score")
    calibrated = profile.get("calibrated_model_score")

    # Only the static branch yields a whole-patient score.
    static_ran = "static_clinical" in report.available_modalities
    if not static_ran or raw is None:
        return PcosAssessmentView(
            available=False,
            evidence_level="not_available",
            unavailable_reason=(
                profile.get("abstention_reason")
                or "A whole-profile model score could not be calculated because "
                "sufficient clinical evidence was not available."
            ),
        )

    # Band the calibrated score when present, or band and number disagree.
    banded = calibrated if calibrated is not None else raw

    return PcosAssessmentView(
        available=True,
        raw_model_score=raw,
        calibrated_model_score=calibrated,
        evidence_level=evidence_level_for(banded),
        calibrated=calibrated is not None,
        source="static_clinical",
        feature_coverage=_static_feature_coverage(report),
        qualifier=(
            "PCOS-related model score from the static-clinical branch. It reflects "
            "the clinical variables provided, not a diagnosis, and not a "
            "probability that you have PCOS."
        ),
    )


def _axes(profile: dict[str, Any]) -> dict[str, AxisView]:
    """The Rotterdam axes, androgenic split preserved.

    ``evidence_source`` is this axis's own evidence, not the report-level
    ``androgenic_evidence_source``. The two answer different questions and can
    legitimately disagree: an axis fires on a guideline threshold, while the
    matching domain also needs training reference stats, and the static cohort
    has no androgen assay.
    """
    raw_axes: dict[str, Any] = profile.get("diagnostic_feature_evidence") or {}

    biochemical = raw_axes.get("hyperandrogenism_biochemical") or {}
    biochemical_available = _axis_status(biochemical) not in {"not_assessable", None}

    out: dict[str, AxisView] = {}
    for name, body in raw_axes.items():
        if not isinstance(body, dict):
            continue
        is_androgenic = name.startswith("hyperandrogenism")
        out[name] = AxisView(
            status=_axis_status(body) or "not_assessable",
            level=body.get("level"),
            supporting_evidence=list(body.get("supporting_evidence") or []),
            missing_evidence=list(body.get("missing_evidence") or []),
            evidence_source=_axis_evidence_source(name),
            # Surfaced on both androgenic axes so a UI rendering only a combined
            # "hyperandrogenism" section still cannot omit that no assay exists.
            biochemical_evidence_available=biochemical_available if is_androgenic else None,
            reason=_axis_reason(body),
            caveats=list(body.get("caveats") or []),
            threshold_sources=dict(body.get("threshold_sources") or {}),
        )
    return out


def _axis_evidence_source(name: str) -> str | None:
    """What kind of evidence an axis is made of, from the axis itself."""
    if name.endswith("_clinical"):
        return "clinical"
    if name.endswith("_biochemical"):
        return "biochemical"
    if name == "polycystic_ovarian_morphology":
        return "imaging"
    return None


def _axis_status(body: dict[str, Any] | None) -> str | None:
    if not body:
        return None
    status = body.get("axis_status")
    return str(status) if status else None


def _axis_reason(body: dict[str, Any]) -> str | None:
    """Why an axis could not be assessed, phrased for a reader."""
    if body.get("axis_status") != "not_assessable":
        return None
    missing = list(body.get("missing_evidence") or [])
    if not missing:
        return "No evidence bearing on this axis was available."
    return "Not assessable: no result for " + ", ".join(missing[:4]) + "."


def _phenotype(profile: dict[str, Any]) -> PhenotypeView:
    """Continuous domains first, similarities second, stability visible."""
    scores: dict[str, Any] = profile.get("phenotype_domain_scores") or {}
    detail: dict[str, Any] = profile.get("phenotype_domain_detail") or {}
    assessability: dict[str, Any] = profile.get("domain_assessability") or {}
    sources: dict[str, Any] = profile.get("domain_evidence_source") or {}

    domains: dict[str, DomainScoreView] = {}
    for name, score in scores.items():
        body: dict[str, Any] = detail.get(name) or {}
        domains[name] = DomainScoreView(
            label=body.get("label"),
            score=score if isinstance(score, int | float) else None,
            available=bool(assessability.get(name, body.get("assessable", False))),
            coverage=_as_float(body.get("coverage")),
            evidence_source=body.get("evidence_source") or sources.get(name),
            # The qualifier is taken from the domain detail, which decides it the
            # same way the upstream composite does. Re-deriving it here from the
            # evidence_source string alone would mark every "symptoms" domain as
            # symptoms-only even when an assay contributed.
            qualifier=body.get("evidence_qualifier"),
            observed_variables=list(body.get("observed_variables") or []),
            missing_variables=list(body.get("missing_variables") or []),
            display_order=_domain_order(name),
        )

    indeterminate = bool(profile.get("indeterminate", True))
    stability = _stability(profile, indeterminate=indeterminate)

    return PhenotypeView(
        domain_scores=domains,
        profile_similarities={
            k: float(v)
            for k, v in (profile.get("profile_similarities") or {}).items()
            if isinstance(v, int | float)
        },
        # Determinate AND stable -- either alone is a coin flip dressed up.
        dominant_profile=(
            profile.get("dominant_profile")
            if not indeterminate and bool(profile.get("assignment_is_stable"))
            else None
        ),
        stable_dominant_profile=bool(profile.get("assignment_is_stable")) and not indeterminate,
        indeterminate=indeterminate,
        indeterminate_reasons=list(profile.get("indeterminate_reasons") or []),
        status=(
            "no_stable_dominant_profile"
            if indeterminate or not profile.get("assignment_is_stable")
            else "stable_dominant_profile"
        ),
        stability=stability,
    )


def _stability(profile: dict[str, Any], *, indeterminate: bool) -> StabilityView:
    """Plain language first, metrics after.

    ``indeterminate`` is taken into account because the two are independent and
    can disagree: resampling can be perfectly stable while the assignment is
    still a near-tie between two profiles. Reporting only the resampling verdict
    would put "the leading pattern held up" on screen directly beside "no stable
    dominant profile", which reads as a contradiction. When a profile was
    withheld, the plain-language line says so and the resampling result becomes
    the supporting detail rather than the headline.
    """
    detail: dict[str, Any] = profile.get("profile_stability") or {}
    if not detail.get("available", False):
        return StabilityView(
            label="not_assessed",
            plain_language="Stability was not assessed for this profile.",
        )

    score = detail.get("stability_score")
    agreement = detail.get("bootstrap_agreement")

    if not isinstance(score, int | float):
        label, plain = "not_assessed", "Stability was not assessed for this profile."
    elif score >= 0.75:
        label = "stable"
        plain = "The leading pattern held up when the evidence was resampled and re-checked."
    elif score >= 0.50:
        label = "moderately_stable"
        plain = "The leading pattern mostly held up, but changed under some checks."
    else:
        label = "unstable"
        plain = (
            "The leading pattern changed across resampling or evidence-removal "
            "checks, so no single profile is reported."
        )

    withheld = detail.get("abstention_reason")
    if indeterminate:
        reasons = [str(r) for r in (profile.get("indeterminate_reasons") or [])]
        withheld = withheld or (reasons[0] if reasons else None)
        plain = "No single profile is reported for this patient. " + (
            "The resampling checks themselves were consistent, but the leading "
            "patterns were too close together to separate."
            if label == "stable"
            else plain
        )

    return StabilityView(
        label=label,
        plain_language=plain,
        stability_score=score if isinstance(score, int | float) else None,
        bootstrap_agreement=agreement if isinstance(agreement, int | float) else None,
        profile_flip_rate=detail.get("profile_flip_rate"),
        unstable_domains=list(detail.get("unstable_domains") or []),
        withheld_reason=withheld,
    )


def _current_state(report: PatientEvidenceReport) -> CurrentStateView:
    """The longitudinal branch, or an explicit unavailability."""
    token = report.tokens.get("longitudinal_hormonal_state")
    if token is None:
        return CurrentStateView(
            available=False,
            unavailable_reason=(
                "No longitudinal measurements were provided, so a current-state "
                "estimate was not produced."
            ),
        )

    features: dict[str, Any] = dict(token.structured_features)

    probabilities = {
        key[len(_CYCLE_PHASE_PREFIX) :]: float(value)
        for key, value in features.items()
        if key.startswith(_CYCLE_PHASE_PREFIX) and isinstance(value, int | float)
    }

    estimates: dict[str, HormoneEstimateView] = {}
    methods: list[str] = []
    for hormone, (canonical, display, unit) in _HORMONES.items():
        value = features.get(f"predicted_{hormone}")
        if not isinstance(value, int | float):
            continue
        method_code = features.get(f"predicted_{hormone}_method")
        method_code = str(method_code) if method_code else None
        if method_code:
            methods.append(method_code)
        estimates[canonical] = HormoneEstimateView(
            code=canonical,
            display_name=display,
            value=float(value),
            method=translate_method(method_code),
            method_code=method_code,
            interval_low=_as_float(features.get(f"predicted_{hormone}_interval_low")),
            interval_high=_as_float(features.get(f"predicted_{hormone}_interval_high")),
            unit=unit,
        )

    phase = features.get("predicted_cycle_phase")
    if phase:
        methods.append("logistic")

    return CurrentStateView(
        available=True,
        predicted_cycle_phase=str(phase) if phase else None,
        cycle_phase_probabilities=probabilities,
        hormone_estimates=estimates,
        input_coverage=_as_float(features.get("input_coverage")),
        confidence=_as_float(getattr(token, "confidence_score", None)),
        observed_days=(
            int(features["observed_days"])
            if isinstance(features.get("observed_days"), int | float)
            else None
        ),
        # Deduplicated, order preserved.
        methods_used=list(dict.fromkeys(translate_method(m) or m for m in methods)),
    )


def _supporting_evidence(profile: dict[str, Any]) -> list[EvidenceStatementView]:
    """Threshold expressions split into parts, so a client renders a sentence
    rather than echoing ``"cycle_length > 35.0"`` at a patient."""
    seen: set[str] = set()
    out: list[EvidenceStatementView] = []

    for axis, body in (profile.get("diagnostic_feature_evidence") or {}).items():
        if not isinstance(body, dict):
            continue
        thresholds: dict[str, Any] = body.get("threshold_sources") or {}
        for item in body.get("supporting_evidence") or []:
            text = str(item)
            if text in seen:
                continue
            seen.add(text)
            out.append(
                EvidenceStatementView(
                    statement=text,
                    variable_code=_leading_code(text),
                    axis=str(axis),
                    guideline_source=(str(thresholds[text]) if text in thresholds else None),
                )
            )
    return out


@lru_cache(maxsize=1)
def _known_variable_codes() -> frozenset[str]:
    """Canonical codes, used to recognise the subject of a threshold expression."""
    try:
        return frozenset(load_variable_registry().variables)
    except Exception:  # noqa: BLE001 - a missing registry must not fail a report
        return frozenset()


def _leading_code(statement: str) -> str | None:
    """The code a threshold expression is about, checked against the registry
    rather than by shape -- many codes are single words (`acne`, `hirsutism`).
    Returns None for prose."""
    head = statement.split(" ", 1)[0].strip()
    return head if head in _known_variable_codes() else None


def _conflict(value: Any) -> ConflictView:
    body = _as_dict(value)
    return ConflictView(
        detail=str(body.get("detail") or body.get("message") or body),
        variable_code=body.get("canonical_variable_code") or body.get("variable_code"),
        modalities=[str(m) for m in (body.get("modalities") or [])],
        severity=body.get("severity"),
    )


def _domain_order(name: str) -> int:
    """Registry display order; unknown domains sort last but deterministically."""
    if name in _DOMAIN_ORDER:
        return _DOMAIN_ORDER.index(name)
    return len(_DOMAIN_ORDER)


def _static_feature_coverage(report: PatientEvidenceReport) -> float | None:
    """Read from the encoder rather than re-derived -- token keys can't
    distinguish a fitted feature slot from a passed-through variable."""
    token = report.tokens.get("static_clinical")
    if token is None:
        return None
    return _as_float(token.structured_features.get("observed_feature_fraction"))


def _provenance(report: PatientEvidenceReport) -> ProvenanceView:
    """One record per contributing branch -- the granularity the report is
    assembled at. Per-event provenance lives on the events themselves."""
    versions: dict[str, str] = {}
    records: list[ProvenanceRecordView] = []

    for name, token in report.tokens.items():
        if token is None:
            continue
        if token.model_version:
            versions[name] = str(token.model_version)
        records.append(
            ProvenanceRecordView(
                label=name,
                origin="model_estimate",
                source_id=token.source_dataset,
                observed_at=str(token.observed_at) if token.observed_at else None,
                model_version=str(token.model_version) if token.model_version else None,
                method=name,
                confidence=_as_float(token.confidence_score),
            )
        )

    for component in report.rule_based_components_used:
        records.append(
            ProvenanceRecordView(
                label=component,
                origin="rule_based_interpretation",
                method=component,
            )
        )

    return ProvenanceView(
        records=records,
        provenance_ids=list(report.provenance_ids),
        model_versions=versions,
        combination_mode=report.combination_mode,
        clinician_review_status=report.clinician_review_status,
    )


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return {"detail": str(value)}


def _as_float(value: Any) -> float | None:
    return float(value) if isinstance(value, int | float) else None
