"""Deterministic evidence coordination across independently-trained encoders.

This layer **does not learn cross-modal relationships**. It performs the eight
deterministic tasks in prompt_4: align patient identity, preserve modality
source, report quality and confidence, detect agreement and disagreement, show
missing modalities, organise output by clinical domain, apply predefined
evidence rules, and abstain where evidence is insufficient.

The aggregation, for domain *d* over modalities *m*:

    S_d = sum_m (w_md * q_m * c_m * s_md) / sum_m (w_md * q_m * c_m)

with ``w`` the design-rule relevance weight, ``q`` the encoder's quality score,
``c`` its confidence, and ``s`` its mapped domain score. Weighting by quality
and confidence means a low-quality ultrasound contributes proportionally less
without being excluded outright -- excluding it would discard real, if weak,
evidence, and including it unweighted would let a bad image outvote a good one.

``w`` is a design rule. It was not fit to anything, and the resulting score
therefore carries no validated accuracy. Every call records that in the report's
``rule_based_components_used``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from inference.disagreement import AgreementThresholds, classify_agreement, explain_disagreement
from inference.domain_mapper import DOMAINS, map_token_to_domains
from inference.report_schema import CoordinatedEvidence, DomainEvidence
from registry.loader import _read_yaml  # noqa: PLC2701 -- same loader the registry uses
from schemas.modality_token import ModalityToken

__all__ = ["EvidenceCoordinator", "load_coordination_config"]

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "models"
_EPS = 1e-9

#: Evidence-mass bands for the qualitative level reported alongside each score.
_LEVEL_CUTS = ((0.66, "high"), (0.40, "moderate"))


def load_coordination_config(path: Path | None = None) -> dict[str, Any]:
    """Load the design-rule weights and thresholds."""
    return _read_yaml(path or _CONFIG_PATH / "evidence_coordination.yaml")


class EvidenceCoordinator:
    """Combine available modality tokens into coordinated domain evidence."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or load_coordination_config()
        agreement = self.config.get("agreement", {})
        self.thresholds = AgreementThresholds(
            strong_max_spread=float(agreement.get("strong_max_spread", 0.15)),
            moderate_max_spread=float(agreement.get("moderate_max_spread", 0.30)),
        )
        self.weights: dict[str, dict[str, float]] = self.config.get("domain_weights", {})
        self.abstention = self.config.get("abstention", {})

    # -- internals ---------------------------------------------------------

    def _weight(self, domain: str, modality: str) -> float:
        return float(self.weights.get(domain, {}).get(modality, 0.0))

    @staticmethod
    def _level(score: float) -> str:
        for cut, label in _LEVEL_CUTS:
            if score >= cut:
                return label
        return "low"

    # -- public API --------------------------------------------------------

    def combine(
        self,
        tokens: list[ModalityToken],
        *,
        patient_id: str | None = None,
        mode: str = "rule_based",
    ) -> CoordinatedEvidence:
        """Coordinate all available tokens for one patient.

        Args:
            tokens: Tokens from whichever encoders ran. May be empty.
            patient_id: Overrides the id read from the tokens.
            mode: ``"separate"`` reports each encoder without combining;
                ``"rule_based"`` applies the design-rule weights;
                ``"calibrated"`` is unavailable and raises.

        Returns:
            Coordinated evidence ready for the PCOS adapter.

        Raises:
            ValueError: If tokens disagree on patient identity, or ``mode`` is
                ``"calibrated"`` (no matched validation data exists to fit it).
        """
        if mode == "calibrated":
            raise ValueError(
                "combination_mode='calibrated' requires weights fit on matched "
                "multimodal validation data, which does not exist in this repository "
                "(ADR-002). Use 'rule_based' and label the output as rule-based."
            )
        if mode not in ("separate", "rule_based"):
            raise ValueError(f"Unknown combination mode '{mode}'.")

        identities = {token.patient_id for token in tokens}
        if len(identities) > 1:
            raise ValueError(
                f"Tokens describe different patients {sorted(identities)}. Coordinating "
                "them would produce one report describing more than one person."
            )
        resolved_id = patient_id or (next(iter(identities)) if identities else "unknown")

        by_modality = {token.modality: token for token in tokens}
        available = sorted(by_modality)
        expected = ("static_clinical", "ovarian_ultrasound", "longitudinal_hormonal_state")
        missing = [name for name in expected if name not in by_modality]

        warnings: list[str] = [
            "Cross-modal evidence was combined using transparent rules rather than a "
            "jointly trained fusion model."
        ]
        provenance: list[str] = []
        for token in tokens:
            provenance.extend(token.provenance_ids)
            warnings.extend(token.warnings)

        # Map every token onto the shared domain scale.
        per_modality: dict[str, dict[str, float]] = {}
        per_modality_evidence: dict[str, list[str]] = {}
        for token in tokens:
            scores, evidence = map_token_to_domains(token)
            per_modality[token.modality] = scores
            per_modality_evidence[token.modality] = evidence

        domain_evidence: dict[str, DomainEvidence] = {}
        for domain in DOMAINS:
            contributions = {
                modality: scores[domain]
                for modality, scores in per_modality.items()
                if domain in scores and self._weight(domain, modality) > 0.0
            }
            domain_evidence[domain] = self._build_domain(
                domain=domain,
                contributions=contributions,
                tokens=by_modality,
                evidence_strings=per_modality_evidence,
                separate=mode == "separate",
            )

        coverage = len(available) / len(expected)
        if coverage <= float(self.abstention.get("low_coverage_threshold", 0.34)):
            warnings.append(
                f"Low modality coverage ({coverage:.0%}): this report rests on "
                f"{len(available)} of {len(expected)} branches."
            )

        return CoordinatedEvidence(
            patient_id=resolved_id,
            static_token=by_modality.get("static_clinical"),
            ultrasound_token=by_modality.get("ovarian_ultrasound"),
            temporal_token=by_modality.get("longitudinal_hormonal_state"),
            domain_evidence=domain_evidence,
            available_modalities=available,
            missing_modalities=missing,
            coverage=coverage,
            combination_mode=mode,  # type: ignore[arg-type]
            provenance_ids=sorted(set(provenance)),
            warnings=list(dict.fromkeys(warnings)),
        )

    def _build_domain(
        self,
        *,
        domain: str,
        contributions: dict[str, float],
        tokens: dict[str, ModalityToken],
        evidence_strings: dict[str, list[str]],
        separate: bool,
    ) -> DomainEvidence:
        """Aggregate one domain, abstaining when the evidence mass is too thin."""
        if not contributions:
            return DomainEvidence(
                domain=domain,
                level="insufficient_evidence",
                missing_evidence=[
                    f"no available modality contributes to {domain}",
                ],
            )

        numerator = 0.0
        mass = 0.0
        for modality, score in contributions.items():
            token = tokens[modality]
            weight = self._weight(domain, modality) * token.quality_score * token.confidence_score
            numerator += weight * score
            mass += weight

        supporting = sorted(contributions)
        evidence: list[str] = []
        for modality in supporting:
            evidence.extend(evidence_strings.get(modality, []))

        min_mass = float(self.abstention.get("min_domain_evidence_mass", 0.20))
        if mass < min_mass:
            return DomainEvidence(
                domain=domain,
                level="insufficient_evidence",
                modality_scores=contributions,
                supporting_modalities=supporting,
                agreement=classify_agreement(contributions, self.thresholds),
                supporting_evidence=evidence,
                missing_evidence=[
                    f"evidence mass {mass:.2f} below the {min_mass:.2f} floor "
                    "(low encoder quality or confidence)"
                ],
            )

        # In 'separate' mode no combined number is produced at all; the caller
        # asked to see each encoder on its own terms.
        score = None if separate else numerator / max(mass, _EPS)
        agreement = classify_agreement(contributions, self.thresholds)

        notes: list[str] = []
        note = explain_disagreement(domain, contributions, self.thresholds)
        if note is not None:
            notes.append(note.message)

        return DomainEvidence(
            domain=domain,
            score=score,
            level=self._level(score) if score is not None else "not_combined",
            modality_scores=contributions,
            supporting_modalities=supporting,
            agreement=agreement,
            supporting_evidence=evidence,
            notes=notes,
        )
