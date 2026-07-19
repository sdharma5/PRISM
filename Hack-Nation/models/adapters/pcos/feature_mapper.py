"""Map coordinated modality evidence into PCOS-specific variables.

The adapter reasons over canonical variable codes (``cycle_length``,
``follicle_number_per_ovary``, ``total_testosterone``, ...), while encoders emit
tokens. This module is the translation, and it is the place where a value's
*origin* must survive: two encoders may report the same variable, and which one
supplied it changes how much it is worth.

Conflicts are recorded, never resolved by overwriting. If the document encoder
extracted LH = 12.4 from a PDF and the static clinical record says LH = 9.1,
both are kept and an :class:`EvidenceConflict` is raised for human review --
the same rule the event store applies. Picking a winner here would make the
adapter the silent arbiter of a disagreement it has no basis to settle.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from inference.report_schema import CoordinatedEvidence, DomainEvidence
from schemas.evidence import EvidenceConflict

__all__ = ["MappedPcosFeatures", "PcosFeatureMapper"]

#: Which token field maps to which PCOS variable. Mirrors PCOS_FEATURE_MAP in
#: prompt_4; kept as data so it can be audited without reading code.
#:
#: Only codes the diagnostic axes actually consume appear here. A token field
#: with no entry is carried through unchanged under its own name, so a new
#: encoder output is never silently discarded.
PCOS_FEATURE_MAP: dict[str, str] = {
    "cycle_regularity": "cycle_irregularity",
    "average_cycle_length": "cycle_length",
    "estimated_follicle_number_per_ovary": "follicle_number_per_ovary",
    "hair_growth_face": "hirsutism",
}

#: Variables whose value is only meaningful with an assay-specific reference
#: range. The adapter reports these but flags them; see diagnostic_features.
_ASSAY_DEPENDENT = {"total_testosterone", "free_testosterone", "dheas", "shbg"}


@dataclass
class MappedPcosFeatures:
    """Canonical PCOS variables assembled from all available tokens."""

    patient_id: str
    #: code -> value, ready for ``assess_all_axes``.
    values: dict[str, float | bool | None] = field(default_factory=dict)
    #: code -> modality that supplied it.
    sources: dict[str, str] = field(default_factory=dict)
    #: Coordinated domain evidence, passed through for the rules layer.
    domain_evidence: dict[str, DomainEvidence] = field(default_factory=dict)

    conflicts: list[EvidenceConflict] = field(default_factory=list)
    available_modalities: list[str] = field(default_factory=list)
    missing_modalities: list[str] = field(default_factory=list)
    assay_dependent_present: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def observed_codes(self) -> list[str]:
        return sorted(code for code, value in self.values.items() if value is not None)


class PcosFeatureMapper:
    """Flatten :class:`CoordinatedEvidence` into PCOS variables."""

    def transform(self, evidence: CoordinatedEvidence) -> MappedPcosFeatures:
        """Assemble one patient's PCOS variables.

        Args:
            evidence: Coordinated output of the evidence coordinator.

        Returns:
            Canonical variables with per-code provenance and any conflicts.
        """
        mapped = MappedPcosFeatures(
            patient_id=evidence.patient_id,
            domain_evidence=dict(evidence.domain_evidence),
            available_modalities=list(evidence.available_modalities),
            missing_modalities=list(evidence.missing_modalities),
            warnings=list(evidence.warnings),
        )

        for modality, token in evidence.tokens().items():
            for raw_code, raw_value in token.structured_features.items():
                code = PCOS_FEATURE_MAP.get(raw_code, raw_code)
                if raw_value is None:
                    continue
                # Only scalar clinical variables participate in axis assessment.
                # Strings such as `predicted_cycle_phase` are state descriptions,
                # not measurements, and are carried but never thresholded.
                if not isinstance(raw_value, int | float | bool):
                    continue

                if code in mapped.values and mapped.values[code] != raw_value:
                    mapped.conflicts.append(
                        EvidenceConflict(
                            variable_name=code,
                            canonical_variable_code=code,
                            event_ids=[
                                *token.provenance_ids,
                                f"modality:{mapped.sources.get(code, 'unknown')}",
                                f"modality:{modality}",
                            ],
                            conflict_type="value_disagreement",
                            detail=(
                                f"{mapped.sources.get(code, 'unknown')} reported "
                                f"{mapped.values[code]}; {modality} reported {raw_value}."
                            ),
                            recommended_resolution=(
                                "Retain both values and ask a clinician which acquisition "
                                "should be used. The adapter does not choose."
                            ),
                            requires_human_review=True,
                        )
                    )
                    # Keep the FIRST value rather than the last so the outcome
                    # does not depend on dict iteration order, and record that a
                    # contested value is in play.
                    continue

                mapped.values[code] = raw_value
                mapped.sources[code] = modality
                if code in _ASSAY_DEPENDENT:
                    mapped.assay_dependent_present.append(code)

        mapped.assay_dependent_present = sorted(set(mapped.assay_dependent_present))
        if mapped.conflicts:
            mapped.warnings.append(
                f"{len(mapped.conflicts)} variable(s) were reported differently by two "
                "modalities and were NOT reconciled; see conflicts."
            )
        return mapped
