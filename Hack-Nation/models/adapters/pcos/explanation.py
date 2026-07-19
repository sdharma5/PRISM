"""Structured, per-section explanation of a PCOS profile.

The sections are kept separate rather than merged into prose for one reason: a
reader must be able to tell *which modality* and *which method* produced each
statement. A paragraph that blends "the learned model gives 0.78" with "the scan
showed 22 follicles" with "the two branches disagree" reads as one coherent
finding, when in fact only the first came from anything trained on a PCOS label.

Every section therefore carries its own provenance, and the module never
generates a sentence that asserts a diagnosis. The strongest statement available
is that evidence for a Rotterdam feature was met, which is a statement about a
published threshold, not about the patient's disease status.
"""

from __future__ import annotations

from typing import Any

__all__ = ["build_explanation"]

_LEVEL_WORDS = {
    "high": "strong",
    "moderate": "moderate",
    "low": "limited",
    "insufficient_evidence": "insufficient",
    "not_combined": "not combined",
}


def _observed_evidence(mapped: Any) -> dict[str, Any]:
    """What was actually measured, and by which modality."""
    by_modality: dict[str, list[str]] = {}
    for code, modality in getattr(mapped, "sources", {}).items():
        by_modality.setdefault(modality, []).append(code)
    return {
        "n_variables_observed": len(mapped.observed_codes()),
        "variables_by_modality": {k: sorted(v) for k, v in sorted(by_modality.items())},
        "available_modalities": list(mapped.available_modalities),
        "note": (
            "Speech and document inputs appear here as the clinical variables they "
            "populated, not as separate predictions -- they are ingestion modalities."
        ),
    }


def _static_prediction(probability: float | None) -> dict[str, Any]:
    """The one genuinely learned component."""
    if probability is None:
        return {
            "available": False,
            "reason": (
                "The learned static clinical head did not run, so no whole-patient PCOS "
                "evidence probability is available. Axis-level findings remain valid."
            ),
        }
    return {
        "available": True,
        "pcos_evidence_probability": round(probability, 4),
        "method": "logistic regression on the matched static clinical cohort",
        "provenance": "static_clinical.pcos_head — the only component fit against a PCOS label",
        "caveat": (
            "Reproduces one clinic's recorded PCOS label on a cross-sectional cohort. "
            "That is not the same as diagnosing PCOS, and the figure carries no "
            "external validation."
        ),
    }


def _rotterdam(diagnostic: dict[str, Any]) -> dict[str, Any]:
    """Guideline axis assessment, with the thresholds that produced it."""
    axes: dict[str, Any] = {}
    met, not_met, not_assessable = [], [], []

    for axis, evidence in diagnostic.items():
        status = getattr(evidence, "axis_status", "not_assessable")
        axes[axis] = {
            "status": status,
            "level": getattr(evidence, "level", "insufficient_evidence"),
            "supporting": list(getattr(evidence, "supporting_evidence", []))[:6],
            "missing": list(getattr(evidence, "missing_evidence", []))[:6],
            "thresholds_applied": dict(getattr(evidence, "threshold_sources", {})),
            "assay_dependent": bool(getattr(evidence, "assay_dependent", False)),
        }
        {"met": met, "not_met": not_met}.get(status, not_assessable).append(axis)

    return {
        "axes": axes,
        "met": met,
        "not_met": not_met,
        "not_assessable": not_assessable,
        "summary": (
            f"{len(met)} axis/axes met, {len(not_met)} not met, "
            f"{len(not_assessable)} not assessable from the supplied evidence."
        ),
        "note": (
            "Axis thresholds come from published guidance and are applied as rules. "
            "Meeting an axis is not a diagnosis; Rotterdam requires clinical "
            "adjudication and exclusion of other causes."
        ),
    }


def _phenotype(similarity: Any) -> dict[str, Any]:
    """Profile similarity, hedged."""
    if similarity is None:
        return {"available": False, "reason": "No domain scores were available."}

    return {
        "available": True,
        "dominant_profile": similarity.dominant,
        "affinities": {k: round(v, 4) for k, v in similarity.affinities.items()},
        "assignment_entropy": round(similarity.entropy, 4),
        "driving_domains": similarity.supporting_evidence,
        "observed_domains": similarity.observed_domains,
        "missing_domains": similarity.missing_domains,
        "indeterminate_reasons": similarity.indeterminate_reasons,
        "note": (
            "These are similarities to described research patterns, NOT validated "
            "clinical subtypes. No subtype label exists in this repository and no "
            "external validation has been performed."
        ),
    }


def _modality_section(
    domain_evidence: dict[str, Any], domain: str, modality: str
) -> dict[str, Any]:
    """Evidence one modality contributed to one domain."""
    entry = domain_evidence.get(domain)
    if entry is None or modality not in getattr(entry, "modality_scores", {}):
        return {"available": False, "reason": f"No {modality} evidence for {domain}."}
    return {
        "available": True,
        "domain": domain,
        "score": entry.modality_scores.get(modality),
        "agreement_with_other_sources": entry.agreement,
        "supporting": list(entry.supporting_evidence)[:6],
    }


def build_explanation(
    *,
    mapped: Any,
    diagnostic: dict[str, Any],
    similarity: Any,
    static_probability: float | None,
    decision: Any,
    stability: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the full structured explanation.

    Args:
        mapped: Mapped PCOS features with per-code provenance.
        diagnostic: Per-axis evidence.
        similarity: Phenotype similarity result, or None.
        static_probability: Learned probability, or None.
        decision: The abstention decision.
        stability: Optional stability report.

    Returns:
        One dict per explanation section, each independently readable.
    """
    domain_evidence = getattr(mapped, "domain_evidence", {})

    conflicting = [conflict.detail for conflict in getattr(mapped, "conflicts", [])]
    conflicting.extend(
        note for entry in domain_evidence.values() for note in getattr(entry, "notes", [])
    )

    missing_variables = sorted(
        {code for e in diagnostic.values() for code in getattr(e, "missing_evidence", [])}
    )

    return {
        "observed_evidence": _observed_evidence(mapped),
        "learned_static_prediction": _static_prediction(static_probability),
        "rotterdam_axis_assessment": _rotterdam(diagnostic),
        "phenotype_profile_similarity": _phenotype(similarity),
        "ultrasound_morphology_evidence": _modality_section(
            domain_evidence, "ovarian_morphology", "ovarian_ultrasound"
        ),
        "current_temporal_state_evidence": _modality_section(
            domain_evidence, "current_state", "longitudinal_hormonal_state"
        ),
        "conflicting_evidence": {
            "n": len(conflicting),
            "items": conflicting[:10],
            "note": (
                "Conflicts are surfaced, never averaged away. A disagreement between a "
                "long-term history and a short temporal window is usually a property of "
                "the observation periods, not a measurement error."
            ),
        },
        "missing_evidence": {
            "variables": missing_variables[:20],
            "modalities": list(getattr(mapped, "missing_modalities", [])),
            "domains": list(getattr(similarity, "missing_domains", []) or []),
        },
        "uncertainty": {
            "assignment_entropy": (
                round(similarity.entropy, 4) if similarity is not None else None
            ),
            "stability": stability or {},
            "note": (
                "No confidence interval is attached to the learned probability: it was "
                "fit on 432 patients from one clinic and its calibration outside that "
                "population is unknown."
            ),
        },
        "abstention": {
            "abstained": bool(getattr(decision, "abstain", False)),
            "reason": getattr(decision, "reason", None),
            "partial_profile_permitted": bool(getattr(decision, "partial_profile_permitted", True)),
        },
        "method_summary": (
            "The PCOS probability is the only learned component. Rotterdam axes are "
            "published thresholds applied as rules. Cross-modal coordination uses "
            "declared design weights. No cross-modal relationship was learned, because "
            "no cohort in this repository has matched multimodal patients."
        ),
    }
