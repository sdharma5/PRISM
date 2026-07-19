"""When the PCOS adapter must decline to produce a profile.

Abstention is not a failure mode -- it is the correct output for a patient whose
evidence cannot support a statement. The rules below are ordered from most to
least severe, and the first one that fires decides, so the reported reason is
always the strongest objection rather than an arbitrary one.

The load-bearing rule is ``pcos_statement_requires_static``. A whole-patient
PCOS evidence probability may only be issued when the learned static clinical
head ran, because that head is the only component in this system fit against a
PCOS label. Ultrasound morphology alone establishes one Rotterdam feature; it
does not establish PCOS, and a system that reported a PCOS probability from an
image alone would be asserting something no component here was trained to
assert.
"""

from __future__ import annotations

from dataclasses import dataclass

from models.adapters.pcos.evidence_rules import DiagnosticFeatureEvidence
from models.adapters.pcos.feature_mapper import MappedPcosFeatures

__all__ = ["AbstentionDecision", "PcosAbstentionEngine"]


@dataclass(frozen=True)
class AbstentionDecision:
    """Whether to abstain, and from what."""

    abstain: bool
    reason: str | None = None
    #: True when domain-level evidence may still be reported even though the
    #: whole-patient PCOS statement is withheld. Partial output beats silence:
    #: an ultrasound-only patient should still learn what the scan showed.
    partial_profile_permitted: bool = True


class PcosAbstentionEngine:
    """Decide whether evidence supports a PCOS profile."""

    def __init__(
        self,
        *,
        min_observed_variables: int = 3,
        require_static_for_pcos_statement: bool = True,
        max_unresolved_conflicts: int = 3,
    ) -> None:
        self.min_observed_variables = min_observed_variables
        self.require_static_for_pcos_statement = require_static_for_pcos_statement
        self.max_unresolved_conflicts = max_unresolved_conflicts

    def evaluate(
        self,
        *,
        mapped: MappedPcosFeatures,
        diagnostic_features: dict[str, DiagnosticFeatureEvidence],
        static_prediction: float | None,
    ) -> AbstentionDecision:
        """Apply the abstention rules in severity order.

        Args:
            mapped: The mapped PCOS variables.
            diagnostic_features: Per-axis evidence.
            static_prediction: The learned static head's probability, if it ran.

        Returns:
            The decision, with the strongest applicable reason.
        """
        if not mapped.available_modalities:
            return AbstentionDecision(
                abstain=True,
                reason="No modality produced usable evidence for this patient.",
                partial_profile_permitted=False,
            )

        observed = mapped.observed_codes()
        if len(observed) < self.min_observed_variables:
            return AbstentionDecision(
                abstain=True,
                reason=(
                    f"Only {len(observed)} clinical variable(s) were observed; at least "
                    f"{self.min_observed_variables} are required to characterise any axis."
                ),
            )

        if len(mapped.conflicts) > self.max_unresolved_conflicts:
            return AbstentionDecision(
                abstain=True,
                reason=(
                    f"{len(mapped.conflicts)} unresolved inter-modality conflicts exceed the "
                    f"limit of {self.max_unresolved_conflicts}. The evidence base is "
                    "internally inconsistent and requires human reconciliation."
                ),
            )

        assessable = [
            axis
            for axis, evidence in diagnostic_features.items()
            if evidence.level != "insufficient_evidence"
        ]
        if not assessable:
            return AbstentionDecision(
                abstain=True,
                reason=("No PCOS diagnostic axis was assessable from the supplied evidence."),
            )

        if self.require_static_for_pcos_statement and static_prediction is None:
            return AbstentionDecision(
                abstain=True,
                reason=(
                    "The learned static clinical head did not run, so no whole-patient PCOS "
                    "evidence probability may be issued. Axis-level findings below remain "
                    "valid on their own terms."
                ),
                partial_profile_permitted=True,
            )

        return AbstentionDecision(abstain=False)
