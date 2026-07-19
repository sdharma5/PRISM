"""Map heterogeneous encoder outputs onto shared clinical domains.

Each encoder speaks its own language: the static clinical encoder emits
weighted z-score composites, the ultrasound encoder emits follicle counts in
millimetres, the temporal encoder emits cycle-phase probabilities. The
coordinator can only compare them once they are on one scale.

That translation is where meaning is most easily lost, so every rule below is
explicit, anchored to a published threshold where one exists, and reported in
``DomainEvidence.supporting_evidence`` so a reader sees what produced a number.

Scale convention: every domain score is in [0, 1] and means *strength of
evidence for the abnormal / PCOS-consistent direction*. 0.5 is "uninformative",
not "normal" -- a genuinely normal finding scores below 0.5.

**These are not calibrated probabilities.** A reproductive score of 0.8 does not
mean an 80% chance of ovulatory dysfunction; it means the evidence sits high on
a documented ramp. Only the learned static PCOS head emits a real probability,
and it is kept separate for exactly that reason.
"""

from __future__ import annotations

import math

from schemas.modality_token import ModalityToken

__all__ = [
    "ANDROGENIC_SOURCES",
    "DOMAIN_MAP",
    "DOMAINS",
    "androgenic_evidence_source",
    "map_token_to_domains",
    "squash_z",
]

#: Shared domains. ``current_state`` is deliberately not a PCOS domain -- it
#: describes where in the cycle a patient is right now, which is a different
#: kind of statement from "how much evidence of PCOS is there".
DOMAINS: tuple[str, ...] = (
    "reproductive",
    "androgenic",
    "metabolic",
    "ovarian_morphology",
    "current_state",
)

#: Declarative record of which token field feeds which domain. Mirrors the
#: PCOS_FEATURE_MAP in prompt_4 section 5; kept as data so the mapping can be
#: audited and tested without reading the functions below.
DOMAIN_MAP: dict[str, str] = {
    "static_clinical.reproductive": "reproductive",
    "static_clinical.clinical_androgenic_evidence": "androgenic",
    "static_clinical.biochemical_androgenic_evidence": "androgenic",
    "static_clinical.metabolic": "metabolic",
    "static_clinical.ovarian": "ovarian_morphology",
    "ovarian_ultrasound.follicle_number_per_ovary": "ovarian_morphology",
    "ovarian_ultrasound.estimated_follicle_number_per_ovary": "ovarian_morphology",
    "ovarian_ultrasound.ovary_volume_ml": "ovarian_morphology",
    "longitudinal_hormonal_state.cycle_irregularity": "reproductive",
    "longitudinal_hormonal_state.predicted_cycle_phase": "current_state",
}

#: 2023 International Evidence-based Guideline: a follicle number per ovary of
#: >=20, OR an ovarian volume >=10 mL, meets the polycystic ovarian morphology
#: criterion on a transvaginal scan with adequate resolution.
PCOM_FOLLICLE_THRESHOLD = 20.0
PCOM_VOLUME_THRESHOLD_ML = 10.0

_EPS = 1e-9


def squash_z(z: float, *, scale: float = 2.0) -> float:
    """Map a z-score composite onto [0, 1] with 0 -> 0.5.

    A logistic rather than a linear clip: composites are unbounded, and a clip
    would make every patient beyond the boundary look identical. ``scale`` sets
    how many standard deviations correspond to a decisive score -- at the
    default, z = +2 maps to about 0.73.
    """
    return 1.0 / (1.0 + math.exp(-float(z) / max(scale, _EPS)))


def _ramp(value: float, *, threshold: float, width: float) -> float:
    """Evidence ramp centred on a published threshold.

    Exactly at ``threshold`` the score is 0.5, because a value sitting on a
    diagnostic cut-point is genuinely equivocal. ``width`` controls how quickly
    evidence accumulates on either side.
    """
    return 1.0 / (1.0 + math.exp(-(float(value) - threshold) / max(width, _EPS)))


def _static_domains(token: ModalityToken) -> tuple[dict[str, float], list[str]]:
    """Domain scores from the static clinical token's composite z-scores."""
    scores: dict[str, float] = {}
    evidence: list[str] = []
    features = token.structured_features

    # The static encoder exports composites under `<domain>_score`, matching
    # registry/phenotype_domains.yaml. Absent composites are skipped, never
    # defaulted -- a missing domain is not a domain scoring zero.
    for registry_name, domain in (
        ("reproductive", "reproductive"),
        ("metabolic", "metabolic"),
        ("ovarian", "ovarian_morphology"),
    ):
        raw = features.get(f"{registry_name}_score")
        if raw is None or not isinstance(raw, int | float):
            continue
        scores[domain] = squash_z(float(raw))
        evidence.append(f"static {registry_name} composite z={float(raw):+.2f}")

    androgenic_score, androgenic_evidence = _combined_androgenic(features)
    if androgenic_score is not None:
        scores["androgenic"] = androgenic_score
    evidence.extend(androgenic_evidence)

    return scores, evidence


#: The shared `androgenic` domain is fed by two separate registry composites.
#: They are combined here, but the combination always declares its source --
#: cutaneous signs and a measured androgen level are not interchangeable
#: evidence, and a reader must never have to guess which one produced a number.
ANDROGENIC_SOURCES = ("symptoms_only", "biochemical_only", "both", "unavailable")


def androgenic_evidence_source(features: dict[str, object]) -> str:
    """Which androgenic evidence backs this token: one of ``ANDROGENIC_SOURCES``."""
    clinical = isinstance(features.get("clinical_androgenic_evidence_score"), int | float)
    biochemical = isinstance(features.get("biochemical_androgenic_evidence_score"), int | float)
    if clinical and biochemical:
        return "both"
    if clinical:
        return "symptoms_only"
    if biochemical:
        return "biochemical_only"
    return "unavailable"


def _combined_androgenic(features: dict[str, object]) -> tuple[float | None, list[str]]:
    """Combine clinical and biochemical androgenic composites into one score.

    When both are present they are averaged: neither is a gold standard for the
    other, and the guideline treats clinical and biochemical hyperandrogenism as
    alternative routes to the same criterion. When only one is present it stands
    alone -- and the evidence string says so, because a symptom-only score read
    as a measured androgen level is precisely the misreading this split exists
    to prevent.
    """
    source = androgenic_evidence_source(features)
    if source == "unavailable":
        return None, [
            "androgenic evidence unavailable: neither cutaneous signs nor an "
            "androgen assay were scored for this patient"
        ]

    evidence: list[str] = []
    parts: list[float] = []
    for key, label in (
        ("clinical_androgenic_evidence_score", "clinical (cutaneous signs)"),
        ("biochemical_androgenic_evidence_score", "biochemical (androgen assay)"),
    ):
        raw = features.get(key)
        if isinstance(raw, int | float):
            parts.append(float(raw))
            evidence.append(f"static {label} androgenic composite z={float(raw):+.2f}")

    evidence.append(f"androgenic evidence source: {source}")
    return squash_z(sum(parts) / len(parts)), evidence


def _ultrasound_domains(token: ModalityToken) -> tuple[dict[str, float], list[str]]:
    """Ovarian morphology evidence from the ultrasound token.

    Only counts the acquisition is entitled to report are used. A per-section
    count from a single frame is NOT an antral follicle count and must not be
    compared against the per-ovary PCOM threshold, so it is ignored here.
    """
    scores: dict[str, float] = {}
    evidence: list[str] = []
    features = token.structured_features

    count = features.get("follicle_number_per_ovary")
    if count is None:
        count = features.get("estimated_follicle_number_per_ovary")
        if count is not None:
            evidence.append("count is a cine-tracked ESTIMATE, not a complete census")

    if isinstance(count, int | float):
        # Width 5: the ramp reaches ~0.88 at 30 follicles and ~0.12 at 10.
        scores["ovarian_morphology"] = _ramp(
            float(count), threshold=PCOM_FOLLICLE_THRESHOLD, width=5.0
        )
        evidence.append(
            f"follicle number per ovary = {float(count):.0f} "
            f"(PCOM threshold {PCOM_FOLLICLE_THRESHOLD:.0f})"
        )

    volume = features.get("ovary_volume_ml")
    if isinstance(volume, int | float):
        volume_score = _ramp(float(volume), threshold=PCOM_VOLUME_THRESHOLD_ML, width=2.5)
        # The guideline treats count and volume as ALTERNATIVE routes to the same
        # criterion, so take the stronger rather than averaging: a patient meeting
        # it on volume alone still meets it.
        scores["ovarian_morphology"] = max(scores.get("ovarian_morphology", 0.0), volume_score)
        evidence.append(
            f"ovarian volume = {float(volume):.1f} mL "
            f"(PCOM threshold {PCOM_VOLUME_THRESHOLD_ML:.0f} mL)"
        )

    return scores, evidence


def _temporal_domains(token: ModalityToken) -> tuple[dict[str, float], list[str]]:
    """Reproductive and current-state evidence from the temporal token."""
    scores: dict[str, float] = {}
    evidence: list[str] = []
    features = token.structured_features

    irregularity = features.get("cycle_irregularity")
    if isinstance(irregularity, int | float):
        scores["reproductive"] = float(min(max(irregularity, 0.0), 1.0))
        evidence.append(f"temporal cycle irregularity = {float(irregularity):.2f}")

    # Phase entropy is a statement about how confidently the current state is
    # known, which is `current_state` evidence -- NOT evidence about PCOS.
    entropy = features.get("cycle_phase_entropy")
    phase = features.get("predicted_cycle_phase")
    if isinstance(entropy, int | float):
        # Entropy over 4 phases is at most log(4); low entropy = confident state.
        confidence = 1.0 - min(float(entropy) / math.log(4.0), 1.0)
        scores["current_state"] = confidence
        evidence.append(f"predicted phase '{phase}' with state confidence {confidence:.2f}")

    return scores, evidence


def map_token_to_domains(token: ModalityToken) -> tuple[dict[str, float], list[str]]:
    """Translate one token into per-domain scores on the shared [0, 1] scale.

    Args:
        token: Any modality token.

    Returns:
        ``(scores, evidence_strings)``. Domains the token cannot speak to are
        simply absent from ``scores``.

    Raises:
        ValueError: If the token's modality has no mapping rule. Silently
            returning nothing would let a new encoder be wired in and quietly
            contribute to no domain at all.
    """
    handlers = {
        "static_clinical": _static_domains,
        "ovarian_ultrasound": _ultrasound_domains,
        "longitudinal_hormonal_state": _temporal_domains,
        # Speech and documents are ingestion modalities. If one reaches the
        # coordinator as its own token, that is a wiring error: its content
        # belongs in the static clinical encoder's input.
    }
    handler = handlers.get(token.modality)
    if handler is None:
        raise ValueError(
            f"No domain mapping for modality '{token.modality}'. Speech and document "
            "tokens must be converted to clinical events and fed to the static "
            "clinical encoder, not coordinated as independent branches."
        )
    return handler(token)
