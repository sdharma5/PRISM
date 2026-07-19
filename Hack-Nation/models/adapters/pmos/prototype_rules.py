"""Named research profiles used *only* to label discovered clusters post hoc.

Scientific WHY
--------------
The PMOS literature repeatedly describes a small set of recurring patterns — a
metabolic/insulin-resistant leaning group, an LH/AMH-driven reproductive leaning
group, a strongly androgenic group, and a lean group with preserved metabolic
health. Those descriptions are useful vocabulary and useless as a classifier.

So the direction of use here is strictly one-way:

    discovered cluster  ->  enrichment vector  ->  best-matching profile name

and **never**

    patient  ->  rules  ->  profile name.

A patient is only ever placed by the clustering model, with soft membership and
an abstain option. These enrichment patterns exist to give a discovered cluster a
human-readable *nickname*, and the match quality (cosine similarity) is reported
alongside it so a weak match reads as weak. If nothing matches well the cluster
keeps its neutral index name — an unnamed group is far better than a misnamed one.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "MIN_MATCH_SIMILARITY",
    "PROTOTYPE_PROFILES",
    "PrototypeMatch",
    "PrototypeProfile",
    "match_prototype",
    "name_clusters",
]

#: Below this cosine similarity a cluster is left with its neutral index name.
MIN_MATCH_SIMILARITY: float = 0.35


@dataclass(frozen=True)
class PrototypeProfile:
    """A named enrichment pattern in standardized (z-score) units."""

    name: str
    label: str
    rationale: str
    #: canonical variable code (or domain name) -> expected z-score direction and
    #: rough magnitude. Magnitudes are indicative only; matching is by cosine
    #: similarity, which is scale-invariant.
    enrichment: Mapping[str, float]
    literature_note: str = ""


PROTOTYPE_PROFILES: dict[str, PrototypeProfile] = {
    "metabolic_leaning": PrototypeProfile(
        name="metabolic_leaning",
        label="Metabolic-leaning research profile",
        rationale=(
            "Adiposity and insulin resistance dominate: higher BMI, waist "
            "circumference, fasting insulin and HOMA-IR, with an atherogenic lipid "
            "pattern (higher triglycerides, lower HDL)."
        ),
        enrichment={
            "bmi": 1.0,
            "waist_circumference": 0.9,
            "waist_hip_ratio": 0.6,
            "fasting_insulin": 1.0,
            "homa_ir": 1.0,
            "fasting_glucose": 0.5,
            "triglycerides": 0.6,
            "hdl_cholesterol": -0.6,
            "shbg": -0.7,
            "metabolic": 1.0,
        },
        literature_note=(
            "Corresponds to the 'metabolic' cluster described in genotype- and "
            "phenotype-based subtyping analyses (e.g. Dapas et al. 2020). Reported "
            "in independent cohorts, but not established as a clinical entity."
        ),
    ),
    "lh_amh_leaning": PrototypeProfile(
        name="lh_amh_leaning",
        label="LH/AMH-leaning research profile",
        rationale=(
            "Reproductive-axis dominant: higher LH, higher LH/FSH ratio, higher AMH "
            "and follicle counts, with comparatively unremarkable metabolic markers."
        ),
        enrichment={
            "luteinizing_hormone": 1.0,
            "lh_fsh_ratio": 1.0,
            "anti_mullerian_hormone": 1.0,
            "follicle_number_per_ovary": 0.8,
            "ovary_volume_ml": 0.5,
            "follicle_stimulating_hormone": -0.3,
            "bmi": -0.3,
            "fasting_insulin": -0.3,
            "ovarian": 1.0,
            "reproductive": 0.7,
        },
        literature_note=(
            "Corresponds to the 'reproductive' cluster of the same literature, "
            "characterised by high LH and SHBG with relatively low BMI."
        ),
    ),
    "androgenic_leaning": PrototypeProfile(
        name="androgenic_leaning",
        label="Androgen-leaning research profile",
        rationale=(
            "Androgen markers dominate: higher total and free testosterone, higher "
            "DHEAS, lower SHBG, and higher hirsutism scoring."
        ),
        enrichment={
            "total_testosterone": 1.0,
            "free_testosterone": 1.0,
            "dheas": 0.7,
            "shbg": -0.8,
            "ferriman_gallwey_score": 0.8,
            "hirsutism": 0.7,
            "acne": 0.4,
            # Either half of the split androgenic axis can carry this profile.
            # Listing only the biochemical one would make the profile impossible
            # to support in a cohort with no androgen assays, and listing a
            # single merged key would hide which kind of evidence it rested on.
            "clinical_androgenic_evidence": 1.0,
            "biochemical_androgenic_evidence": 1.0,
        },
        literature_note=(
            "Androgen-predominant groupings recur across cohorts but overlap "
            "substantially with the metabolic pattern via SHBG suppression."
        ),
    ),
    "lean_reproductive": PrototypeProfile(
        name="lean_reproductive",
        label="Lean reproductive research profile",
        rationale=(
            "Low adiposity and preserved insulin sensitivity alongside cycle "
            "irregularity: lower BMI, waist circumference, insulin and HOMA-IR, with "
            "elevated reproductive-axis markers."
        ),
        enrichment={
            "bmi": -0.9,
            "waist_circumference": -0.8,
            "fasting_insulin": -0.8,
            "homa_ir": -0.8,
            "hdl_cholesterol": 0.5,
            "shbg": 0.6,
            "cycle_irregularity": 0.7,
            "luteinizing_hormone": 0.5,
            "metabolic": -0.9,
            "reproductive": 0.7,
        },
        literature_note=(
            "'Lean PMOS' is widely described clinically; it is best treated as a "
            "descriptive observation rather than a distinct mechanism."
        ),
    ),
}


@dataclass(frozen=True)
class PrototypeMatch:
    """The best-matching named profile for one discovered cluster."""

    cluster: str
    profile_name: str | None
    similarity: float
    label: str
    #: Every profile's similarity, so a near-tie is visible rather than hidden.
    all_similarities: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _cosine(a: Mapping[str, float], b: Mapping[str, float]) -> float:
    """Cosine similarity over the *union* of the two vectors' features.

    Taking the union with zero-fill rather than only the shared features is
    essential. On the shared-feature intersection a profile that happens to
    mention a single variable would score a perfect 1.0 against any cluster
    enriched in that variable, no matter how badly it failed to account for
    everything else the cluster is enriched in. Zero-filling means a profile is
    penalized both for what it gets wrong and for what it stays silent about.
    """
    keys = sorted(set(a) | set(b))
    if not keys:
        return 0.0
    va = np.array([float(a.get(k, 0.0)) for k in keys])
    vb = np.array([float(b.get(k, 0.0)) for k in keys])
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom <= 0:
        return 0.0
    return float(np.dot(va, vb) / denom)


def match_prototype(
    cluster: str,
    enrichment: Mapping[str, float],
    profiles: Mapping[str, PrototypeProfile] | None = None,
    min_similarity: float = MIN_MATCH_SIMILARITY,
) -> PrototypeMatch:
    """Find the named research profile a discovered cluster most resembles.

    Returns ``profile_name=None`` when no profile clears ``min_similarity``, and
    warns when the top two are within 0.10 of each other — a near-tie means the
    nickname is not carrying real information and should be read as decoration.
    """
    catalogue = dict(profiles or PROTOTYPE_PROFILES)
    similarities = {
        name: _cosine(enrichment, profile.enrichment) for name, profile in catalogue.items()
    }
    if not similarities:
        return PrototypeMatch(cluster, None, 0.0, cluster, {}, ["no prototype profiles supplied"])

    ordered = sorted(similarities.items(), key=lambda kv: kv[1], reverse=True)
    best_name, best_score = ordered[0]
    warnings: list[str] = []

    if best_score < min_similarity:
        return PrototypeMatch(
            cluster=cluster,
            profile_name=None,
            similarity=float(best_score),
            label=cluster,
            all_similarities=similarities,
            warnings=[
                f"no named research profile matched cluster '{cluster}' above "
                f"{min_similarity:.2f} (best {best_name} at {best_score:.2f}); "
                "the cluster keeps its neutral index name"
            ],
        )

    if len(ordered) > 1 and abs(ordered[0][1] - ordered[1][1]) < 0.10:
        warnings.append(
            f"cluster '{cluster}' matches '{ordered[0][0]}' ({ordered[0][1]:.2f}) and "
            f"'{ordered[1][0]}' ({ordered[1][1]:.2f}) almost equally; the name is a "
            "convenience label only"
        )

    return PrototypeMatch(
        cluster=cluster,
        profile_name=best_name,
        similarity=float(best_score),
        label=f"{cluster} (resembles the {catalogue[best_name].label})",
        all_similarities=similarities,
        warnings=warnings,
    )


def name_clusters(
    enrichment_by_cluster: Mapping[str, Mapping[str, float]],
    min_similarity: float = MIN_MATCH_SIMILARITY,
) -> dict[str, PrototypeMatch]:
    """Label every discovered cluster post hoc. Never used to assign patients."""
    return {
        cluster: match_prototype(cluster, enrichment, min_similarity=min_similarity)
        for cluster, enrichment in enrichment_by_cluster.items()
    }
