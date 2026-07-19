"""Detect and *preserve* disagreement between modalities.

The governing rule from prompt_4 section 7: **do not average away
contradictions.** When the static history says persistent cycle irregularity and
a two-week wearable window says the cycle looks regular, the mean of the two is
a number that describes neither observation and hides the most clinically
interesting fact -- that the observation windows disagree.

So the coordinator still computes a combined score (a report needs one), but a
conflicting domain is *labelled* conflicting, the contributing scores are kept
individually, and a note explaining the likely reason is attached. A consumer
that reads only the score is at least warned that the score is contested.

Known structural disagreements get a specific explanation rather than a generic
"sources disagree", because the reason is usually a property of the modalities
rather than a measurement error -- a short temporal window genuinely cannot
characterise long-term regularity, and saying so is more useful than flagging it.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "AgreementThresholds",
    "DisagreementNote",
    "classify_agreement",
    "explain_disagreement",
]


@dataclass(frozen=True)
class AgreementThresholds:
    """Spread cut-points, loaded from ``configs/models/evidence_coordination.yaml``."""

    strong_max_spread: float = 0.15
    moderate_max_spread: float = 0.30


@dataclass(frozen=True)
class DisagreementNote:
    """One human-readable explanation of why two modalities differ."""

    domain: str
    modalities: tuple[str, ...]
    spread: float
    message: str


def classify_agreement(
    modality_scores: dict[str, float],
    thresholds: AgreementThresholds | None = None,
) -> str:
    """Classify agreement across the modalities scoring one domain.

    Returns:
        ``"none"`` (no contributor), ``"single_source"`` (exactly one),
        ``"strong"``, ``"moderate"`` or ``"conflicting"``.
    """
    limits = thresholds or AgreementThresholds()
    if not modality_scores:
        return "none"
    if len(modality_scores) == 1:
        return "single_source"

    spread = max(modality_scores.values()) - min(modality_scores.values())
    if spread <= limits.strong_max_spread:
        return "strong"
    if spread <= limits.moderate_max_spread:
        return "moderate"
    return "conflicting"


#: Structural explanations for disagreements that arise from what a modality can
#: observe, not from an error. Keyed by the frozenset of modalities involved.
_STRUCTURAL_EXPLANATIONS: dict[frozenset[str], dict[str, str]] = {
    frozenset({"static_clinical", "longitudinal_hormonal_state"}): {
        "reproductive": (
            "The static history and the short-term temporal window are not fully "
            "aligned. The temporal observation period may be too short to "
            "characterise long-term cycle regularity, which is what the history "
            "describes. Neither observation supersedes the other."
        ),
    },
    frozenset({"static_clinical", "ovarian_ultrasound"}): {
        "ovarian_morphology": (
            "Imaging-derived morphology and report-transcribed follicle counts "
            "disagree. The transcribed counts are second-hand and may come from a "
            "different scan, operator or scanner than the images supplied here."
        ),
    },
}


def explain_disagreement(
    domain: str,
    modality_scores: dict[str, float],
    thresholds: AgreementThresholds | None = None,
) -> DisagreementNote | None:
    """Return an explanation when a domain's contributors conflict.

    Args:
        domain: Domain name.
        modality_scores: Per-modality score for this domain.
        thresholds: Agreement cut-points.

    Returns:
        A note when agreement is ``"conflicting"``, else None.
    """
    if classify_agreement(modality_scores, thresholds) != "conflicting":
        return None

    modalities = tuple(sorted(modality_scores))
    spread = max(modality_scores.values()) - min(modality_scores.values())

    structural = _STRUCTURAL_EXPLANATIONS.get(frozenset(modalities), {}).get(domain)
    if structural is None:
        ranked = sorted(modality_scores.items(), key=lambda kv: kv[1])
        low, high = ranked[0], ranked[-1]
        structural = (
            f"Sources disagree on {domain}: {high[0]} scores {high[1]:.2f} while "
            f"{low[0]} scores {low[1]:.2f}. The combined score below is reported "
            "with this disagreement unresolved, not as a consensus."
        )

    return DisagreementNote(domain=domain, modalities=modalities, spread=spread, message=structural)
