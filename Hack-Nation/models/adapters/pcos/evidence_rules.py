"""Transparent PCOS evidence rules over mapped multimodal features.

Two sources of truth are combined here, and they are not equals:

1. **Guideline axis assessment** (:mod:`models.adapters.pcos.diagnostic_features`)
   -- published thresholds with citations, applied to canonical variables. This
   is the authoritative signal. An axis that is *met* is met because a named
   guideline says so.
2. **Coordinated domain evidence** -- the design-rule weighted scores from the
   evidence coordinator. These give a graded reading where the axis gives a
   binary one, and cover the case where no single variable crosses a threshold
   but the overall picture is suggestive.

The rule: the axis decides the *level* when it is assessable; the domain score
only refines it. That ordering matters. A graded score derived from unvalidated
design weights must never override a published diagnostic threshold, and a
patient whose follicle count is 24 has met the PCOM criterion regardless of what
a weighted composite says.

Nothing here is learned. Every number is either a guideline threshold or a
declared weight, which is what makes this layer legal without matched
multimodal training data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from models.adapters.pcos.diagnostic_features import assess_all_axes
from models.adapters.pcos.feature_mapper import MappedPcosFeatures

__all__ = ["DiagnosticFeatureEvidence", "PcosEvidenceRules"]

#: Which coordinated domain informs which diagnostic axis.
_AXIS_DOMAIN = {
    "ovulatory_dysfunction": "reproductive",
    "hyperandrogenism_clinical": "androgenic",
    "hyperandrogenism_biochemical": "androgenic",
    "polycystic_ovarian_morphology": "ovarian_morphology",
}

_LEVEL_FOR_SCORE = ((0.66, "high"), (0.40, "moderate"))


@dataclass
class DiagnosticFeatureEvidence:
    """Evidence for one PCOS diagnostic axis."""

    axis: str
    level: str = "insufficient_evidence"
    score: float | None = None
    axis_status: str = "not_assessable"

    supporting_evidence: list[str] = field(default_factory=list)
    missing_evidence: list[str] = field(default_factory=list)
    threshold_sources: dict[str, str] = field(default_factory=dict)
    assay_dependent: bool = False
    caveats: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


class PcosEvidenceRules:
    """Evaluate PCOS diagnostic axes from mapped multimodal evidence."""

    def __init__(self, threshold_overrides: dict[str, dict[str, float]] | None = None) -> None:
        """
        Args:
            threshold_overrides: Per-axis threshold replacements. Biochemical
                androgen cut-points are assay-specific and the shipped defaults
                are placeholders, so a real deployment must override them.
        """
        self.threshold_overrides = threshold_overrides

    @staticmethod
    def _level_for(score: float) -> str:
        for cut, label in _LEVEL_FOR_SCORE:
            if score >= cut:
                return label
        return "low"

    def evaluate(self, mapped: MappedPcosFeatures) -> dict[str, DiagnosticFeatureEvidence]:
        """Assess every PCOS axis.

        Args:
            mapped: Canonical variables plus coordinated domain evidence.

        Returns:
            One :class:`DiagnosticFeatureEvidence` per axis.
        """
        assessments = assess_all_axes(mapped.values, self.threshold_overrides)
        results: dict[str, DiagnosticFeatureEvidence] = {}

        for axis, assessment in assessments.items():
            domain_name = _AXIS_DOMAIN.get(axis)
            domain = mapped.domain_evidence.get(domain_name) if domain_name else None
            domain_score = domain.score if domain is not None else None

            evidence = DiagnosticFeatureEvidence(
                axis=axis,
                axis_status=assessment.status,
                score=domain_score,
                threshold_sources=dict(assessment.threshold_sources),
                assay_dependent=assessment.assay_dependent,
                caveats=list(assessment.caveats),
            )

            # Supporting evidence: the guideline rules that fired, plus whatever
            # the coordinator recorded for the corresponding domain.
            evidence.supporting_evidence = list(assessment.satisfied_rules)
            if domain is not None:
                evidence.supporting_evidence.extend(domain.supporting_evidence)
                evidence.notes.extend(domain.notes)
                if domain.agreement == "conflicting":
                    evidence.notes.append(
                        f"Modalities disagree on {domain_name}; this axis rests on "
                        "contested evidence."
                    )

            evidence.missing_evidence = list(assessment.missing_codes)

            # Level: the guideline verdict dominates where it exists.
            if assessment.status == "met":
                evidence.level = "high"
            elif assessment.status == "not_met":
                # A negative guideline verdict with a suggestive graded score is
                # reported as low, not moderate: the threshold was not crossed.
                evidence.level = "low"
            elif domain_score is not None:
                evidence.level = self._level_for(domain_score)
                evidence.notes.append(
                    "No guideline threshold was assessable; this level comes from "
                    "design-rule weighted evidence only and is not a criterion verdict."
                )
            else:
                evidence.level = "insufficient_evidence"
                evidence.score = None

            if evidence.assay_dependent:
                evidence.caveats.append(
                    "Biochemical androgen thresholds are assay-specific; the applied "
                    "cut-points are placeholders unless overridden for this laboratory."
                )

            results[axis] = evidence

        return results
