"""The PMOS adapter: the single entry point that ties Step 5 together.

Scientific WHY
--------------
Everything above this file is condition-agnostic — it clusters matrices, resamples
them, and reports how much survived. All PMOS-specific knowledge (which feature
axes exist, which thresholds apply, which named research profiles to compare
discovered clusters against) lives in this package and nowhere else. That
separation is what makes the pipeline reusable for another hormonal condition
without rewriting the statistics, and it is what stops PMOS assumptions leaking
silently into the discovery step.

The order of operations is deliberate:

1. Discover groups on an **explicit PMOS-positive training subset** only.
2. Choose K on measured evidence, never on the literature's favourite number.
3. Quantify how fragile the result is (bootstrap, ablation, perturbation).
4. Calibrate membership probabilities against measured bootstrap agreement.
5. Only then describe a participant — hedged, with abstention available, and with
   the missing evidence stated alongside the observed evidence.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np

from models.adapters.pmos.diagnostic_features import AxisAssessment, assess_all_axes
from models.adapters.pmos.output_schema import (
    AbstentionReport,
    MissingEvidence,
    ModelOrganizedPhenotype,
    ObservedEvidence,
    PmosResearchOutput,
    UncertaintyReport,
)
from models.adapters.pmos.phenotype_heads import (
    build_phenotype_profile,
    compute_domain_scores,
    defining_feature_coverage,
)
from models.adapters.pmos.prototype_rules import PrototypeMatch, name_clusters
from models.phenotype.clustering import (
    ClusteringInput,
    FittedClustering,
    KSelection,
    fit_base_partition,
    fit_clustering,
    run_clustering_benchmark,
    select_k,
)
from models.phenotype.prototype_mapping import (
    ClusterCharacterization,
    assert_hedged_language,
    characterize_clusters,
)
from models.stability.ablation import AblationResult, run_ablation
from models.stability.abstention import (
    AbstentionEvidence,
    AbstentionThresholds,
    evaluate_abstention,
    scaled_distance_percentile,
)
from models.stability.bootstrap import BootstrapResult, align_labels, bootstrap_clustering
from models.stability.calibration import (
    CalibrationResult,
    fit_temperature,
    membership_from_distances,
)
from models.stability.perturbation import PerturbationResult, run_perturbations
from schemas.phenotype import INDETERMINATE, PhenotypeProfile, StabilityReport

__all__ = ["PmosAdapter", "PmosAdapterConfig"]

LIMITATIONS = [
    "Groups were discovered by unsupervised clustering in a single research cohort "
    "and have not been replicated in an independent cohort.",
    "No exclusion of other causes of oligo-anovulation or androgen excess (thyroid "
    "disease, hyperprolactinaemia, non-classical CAH, Cushing syndrome) was possible.",
    "Biochemical thresholds are assay- and laboratory-specific; the encoded defaults "
    "are placeholders.",
    "Missing data are imputed for clustering; participants with sparse measurements "
    "receive memberships driven substantially by the imputation.",
    "Cohort composition determines what clusters exist at all; the result does not "
    "transfer to a population with a different referral pattern.",
]


@dataclass
class PmosAdapterConfig:
    """Every analyst choice, in one inspectable object."""

    algorithms: tuple[str, ...] = ("kmeans", "gaussian_mixture", "agglomerative", "consensus")
    k_values: tuple[int, ...] = (2, 3, 4, 5, 6)
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    n_bootstrap: int = 30
    consensus_resamples: int = 30
    n_noise_replicates: int = 5
    enrichment_threshold: float = 0.4
    thresholds: AbstentionThresholds = field(default_factory=AbstentionThresholds)
    threshold_overrides: dict[str, dict[str, float]] = field(default_factory=dict)
    modality_of: dict[str, str] = field(default_factory=dict)
    model_version: str = "0.1.0"
    source_dataset: str | None = None


@dataclass
class PmosDiscovery:
    """The fitted state produced by :meth:`PmosAdapter.fit`."""

    data: ClusteringInput
    selection: KSelection
    fitted: FittedClustering
    benchmarks: list[Any]
    characterizations: dict[str, ClusterCharacterization]
    prototype_names: dict[str, PrototypeMatch]
    bootstrap: BootstrapResult
    ablation: AblationResult
    perturbations: list[PerturbationResult]
    calibration: CalibrationResult
    membership: np.ndarray
    profile_names: list[str]
    alternative_assignments: dict[str, list[str]] = field(default_factory=dict)


class PmosAdapter:
    """Discover exploratory PMOS-related profiles and describe one participant.

    Usage is two-phase: :meth:`fit` on a cohort, then :meth:`profile` per
    participant. ``profile`` returns the triple
    ``(PhenotypeProfile, StabilityReport, PmosResearchOutput)``.
    """

    def __init__(self, config: PmosAdapterConfig | None = None) -> None:
        self.config = config or PmosAdapterConfig()
        self.discovery: PmosDiscovery | None = None

    # ---------------------------------------------------------------- fitting

    def fit(
        self,
        representations: Sequence[ClusteringInput],
        pmos_positive_training_ids: Sequence[str],
    ) -> PmosAdapter:
        """Run the full discovery + stability pipeline on an explicit subset.

        ``pmos_positive_training_ids`` is required and is passed straight through
        to :func:`run_clustering_benchmark`, which refuses an empty subset. The
        subset must be *training* participants only: selecting K on data that will
        later be used to evaluate the grouping is circular.
        """
        cfg = self.config
        benchmarks = run_clustering_benchmark(
            representations,
            cluster_subset_ids=pmos_positive_training_ids,
            algorithms=cfg.algorithms,
            k_values=cfg.k_values,
            seeds=cfg.seeds,
            n_bootstrap=cfg.n_bootstrap,
            consensus_resamples=cfg.consensus_resamples,
        )
        selection = select_k(benchmarks)

        chosen = next(r for r in representations if r.label == selection.representation)
        data = chosen.subset(pmos_positive_training_ids)
        fitted = fit_clustering(
            data,
            selection.algorithm,
            selection.k,
            seed=int(cfg.seeds[0]),
            consensus_resamples=cfg.consensus_resamples,
        )

        characterizations = characterize_clusters(
            data.matrix,
            fitted.labels,
            list(data.feature_names) or [f"f{i}" for i in range(data.matrix.shape[1])],
            threshold=cfg.enrichment_threshold,
        )
        prototype_names = name_clusters(
            {name: char.enrichment for name, char in characterizations.items()}
        )

        boot = bootstrap_clustering(
            data,
            selection.algorithm,
            selection.k,
            reference_labels=fitted.labels,
            n_bootstrap=cfg.n_bootstrap,
            seed=int(cfg.seeds[0]),
        )
        ablation = run_ablation(
            data,
            fitted.labels,
            selection.algorithm,
            selection.k,
            modality_of=cfg.modality_of,
            seed=int(cfg.seeds[0]),
        )
        perturbations = run_perturbations(
            data,
            fitted.labels,
            selection.algorithm,
            selection.k,
            n_noise_replicates=cfg.n_noise_replicates,
            seed=int(cfg.seeds[0]),
        )

        raw_membership = (
            fitted.responsibilities
            if fitted.responsibilities is not None
            else membership_from_distances(data.matrix, np.asarray(fitted.centers))
        )
        agreement = [boot.agreement_rate.get(pid, 0.0) for pid in data.participant_ids]
        calibration = fit_temperature(raw_membership, agreement)
        from models.stability.calibration import temperature_scale

        membership = temperature_scale(raw_membership, calibration.temperature)

        profile_names = [f"profile_{label}" for label in sorted(np.unique(fitted.labels))]

        self.discovery = PmosDiscovery(
            data=data,
            selection=selection,
            fitted=fitted,
            benchmarks=list(benchmarks),
            characterizations=characterizations,
            prototype_names=prototype_names,
            bootstrap=boot,
            ablation=ablation,
            perturbations=perturbations,
            calibration=calibration,
            membership=membership,
            profile_names=profile_names,
            alternative_assignments=self._alternative_assignments(
                representations, pmos_positive_training_ids, selection, fitted
            ),
        )
        return self

    def _alternative_assignments(
        self,
        representations: Sequence[ClusteringInput],
        subset_ids: Sequence[str],
        selection: KSelection,
        fitted: FittedClustering,
    ) -> dict[str, list[str]]:
        """Refit every other (representation, algorithm) at the chosen K.

        This is the evidence for abstention rule 2: a participant placed
        differently by a defensible alternative analysis has not been placed.
        """
        out: dict[str, list[str]] = {pid: [] for pid in fitted.participant_ids}
        for representation in representations:
            try:
                alt_data = representation.subset(subset_ids)
            except ValueError:
                continue
            if list(alt_data.participant_ids) != list(fitted.participant_ids):
                continue
            for algorithm in self.config.algorithms:
                if (
                    representation.label == selection.representation
                    and algorithm == selection.algorithm
                ):
                    continue
                base = "kmeans" if algorithm == "consensus" else algorithm
                try:
                    labels = np.asarray(
                        fit_base_partition(
                            alt_data.matrix, base, selection.k, int(self.config.seeds[0])
                        )
                    )
                except (ValueError, np.linalg.LinAlgError):
                    continue
                mapping = align_labels(fitted.labels, labels)
                for i, pid in enumerate(alt_data.participant_ids):
                    mapped = mapping.get(int(labels[i]), -1)
                    out[pid].append(f"profile_{mapped}" if mapped >= 0 else "unmatched")
        return out

    # --------------------------------------------------------------- profiling

    def profile(
        self,
        patient_id: str,
        raw_values: Mapping[str, float | bool | None] | None = None,
        standardized_values: Mapping[str, float | None] | None = None,
    ) -> tuple[PhenotypeProfile, StabilityReport, PmosResearchOutput]:
        """Describe one participant of the fitted cohort. The single entry point.

        ``raw_values`` are canonical-unit measurements used for the feature-axis
        assessments; ``standardized_values`` are cohort z-scores used for domain
        scores. Either may be omitted, in which case the corresponding section
        reports "not assessable" rather than guessing.
        """
        if self.discovery is None:
            raise RuntimeError("PmosAdapter.profile() called before fit().")
        d = self.discovery
        if patient_id not in d.data.participant_ids:
            raise KeyError(
                f"'{patient_id}' was not in the fitted cohort. Out-of-cohort scoring is "
                "not supported: the profiles are only defined relative to this cohort."
            )
        index = d.data.participant_ids.index(patient_id)

        axes = assess_all_axes(raw_values or {}, self.config.threshold_overrides)
        domain_scores = compute_domain_scores(standardized_values or {})
        defining = list(d.data.feature_names) or []
        observed_defining, total_defining, coverage_assessable = self._defining_coverage(
            defining, standardized_values, domain_scores
        )

        membership = {
            d.profile_names[j]: float(d.membership[index, j]) for j in range(d.membership.shape[1])
        }
        dominant_label = int(d.fitted.labels[index])
        dominant_name = f"profile_{dominant_label}"

        flipping = d.ablation.per_participant_flips.get(patient_id, [])
        distance_percentile = scaled_distance_percentile(
            d.data.matrix[index], np.asarray(d.fitted.centers), d.data.matrix
        )
        flip_rate = float(
            np.mean(
                [
                    1.0 if p.per_participant_flipped.get(patient_id, False) else 0.0
                    for p in d.perturbations
                ]
            )
            if d.perturbations
            else 0.0
        )
        mean_js = float(
            np.mean([p.per_participant_js.get(patient_id, 0.0) for p in d.perturbations])
            if d.perturbations
            else 0.0
        )

        stability = evaluate_abstention(
            AbstentionEvidence(
                patient_id=patient_id,
                dominant_profile=dominant_name,
                probabilities=membership,
                alternative_assignments=d.alternative_assignments.get(patient_id, []),
                bootstrap_agreement=d.bootstrap.agreement_rate.get(patient_id),
                cluster_jaccard=d.bootstrap.cluster_jaccard_for(dominant_label),
                flipping_features=flipping,
                n_features_tested=len(d.ablation.flip_rate),
                fragility_by_feature=dict(d.ablation.flip_rate),
                distance_percentile=distance_percentile,
                n_defining_features_observed=observed_defining if coverage_assessable else None,
                n_defining_features_total=total_defining if coverage_assessable else None,
                subtype_flip_rate=flip_rate,
            ),
            self.config.thresholds,
        )

        indeterminate_mass = 1.0 if stability.abstain else 0.0
        phenotype = build_phenotype_profile(
            patient_id=patient_id,
            domain_scores=domain_scores,
            membership=membership,
            representation=d.selection.representation,
            n_profiles=d.selection.k,
            indeterminate_mass=indeterminate_mass,
            model_version=self.config.model_version,
            extra_warnings=list(stability.warnings),
        )

        output = self._build_output(
            patient_id,
            axes,
            domain_scores,
            phenotype,
            stability,
            observed_defining,
            total_defining,
            mean_js,
            flip_rate,
            raw_values or {},
            dominant_name,
        )
        return phenotype, stability, output

    @staticmethod
    def _defining_coverage(
        defining: list[str],
        standardized_values: Mapping[str, float | None] | None,
        domain_scores: Mapping[str, Any],
    ) -> tuple[int, int, bool]:
        """How much of the profile-defining evidence this participant actually has.

        The columns that define the profiles depend on the winning representation,
        and they are not always canonical variable codes. When the representation
        is ``domain_scores`` its columns are *domain names*, so counting them
        against a variable-code dictionary would score every participant 0 and
        abstain on the entire cohort for a purely clerical reason. We therefore
        resolve coverage against whichever key space the defining columns live in,
        and return ``assessable=False`` when they match neither — an unrunnable
        check must report itself as unrun, not as failed.
        """
        if not defining:
            return 0, 0, False
        if all(name in domain_scores for name in defining):
            observed = sum(1 for name in defining if domain_scores[name].score is not None)
            return observed, len(defining), True
        if standardized_values and any(name in standardized_values for name in defining):
            observed, total = defining_feature_coverage(standardized_values, defining)
            return observed, total, True
        return 0, len(defining), False

    # ----------------------------------------------------------------- output

    def _resemblance_statement(self, profile_name: str, abstained: bool) -> str:
        """Hedged one-liner, run through the banned-phrase guard before returning."""
        if abstained:
            text = (
                "This participant could not be placed with enough confidence to name a "
                "group. The result is reported as indeterminate."
            )
            return assert_hedged_language(text, context="resemblance statement")
        d = self.discovery
        assert d is not None
        match = d.prototype_names.get(profile_name)
        if match is None or match.profile_name is None:
            text = (
                f"This participant is most similar to the data-driven group "
                f"'{profile_name}' found in this cohort, which did not closely "
                f"resemble any named research profile."
            )
        else:
            text = (
                f"This participant is most similar to the data-driven group "
                f"'{profile_name}', which itself has overlap with the "
                f"{match.profile_name.replace('_', ' ')} research profile "
                f"(pattern similarity {match.similarity:.2f}). This is a descriptive "
                f"resemblance within one research cohort."
            )
        return assert_hedged_language(text, context="resemblance statement")

    def _build_output(
        self,
        patient_id: str,
        axes: Mapping[str, AxisAssessment],
        domain_scores: Mapping[str, Any],
        phenotype: PhenotypeProfile,
        stability: StabilityReport,
        observed_defining: int,
        total_defining: int,
        mean_js: float,
        flip_rate: float,
        raw_values: Mapping[str, float | bool | None],
        dominant_name: str,
    ) -> PmosResearchOutput:
        d = self.discovery
        assert d is not None

        threshold_sources: dict[str, str] = {}
        caveats: list[str] = []
        for axis in axes.values():
            threshold_sources.update(axis.threshold_sources)
            caveats.extend(axis.caveats)

        observed = ObservedEvidence(
            observed_variables=sorted(k for k, v in raw_values.items() if v is not None),
            axis_status={name: axis.status for name, axis in axes.items()},
            axis_evidence_available={name: axis.evidence_available for name, axis in axes.items()},
            threshold_sources=threshold_sources,
            assay_dependent_axes=[n for n, a in axes.items() if a.assay_dependent],
            caveats=sorted(set(caveats)),
        )

        missing = MissingEvidence(
            missing_variables=sorted(
                {code for axis in axes.values() for code in axis.missing_codes}
            ),
            not_assessable_axes=[n for n, a in axes.items() if a.status == "not_assessable"],
            domains_below_coverage=[
                name for name, score in domain_scores.items() if score.score is None
            ],
            n_defining_features_observed=observed_defining,
            n_defining_features_total=total_defining,
            notes=[
                "A 'not_assessable' axis means the inputs were not measured. It must "
                "never be read as evidence that the feature is absent."
            ],
        )

        organized = ModelOrganizedPhenotype(
            representation=d.selection.representation,
            algorithm=d.selection.algorithm,
            n_profiles=d.selection.k,
            profile_probabilities=phenotype.phenotype_probabilities,
            dominant_profile=phenotype.dominant_profile,
            dominant_probability=phenotype.dominant_probability,
            resemblance_statement=self._resemblance_statement(dominant_name, stability.abstain),
            profile_descriptions={
                name: char.description for name, char in d.characterizations.items()
            },
            k_selection_rationale=d.selection.rationale,
        )

        uncertainty = UncertaintyReport(
            assignment_entropy=stability.assignment_entropy,
            stability_score=stability.stability_score,
            subtype_flip_rate=stability.subtype_flip_rate,
            bootstrap_jaccard=stability.bootstrap_jaccard,
            cohort_mean_bootstrap_jaccard=d.bootstrap.mean_jaccard,
            calibration_temperature=d.calibration.temperature,
            expected_calibration_error=d.calibration.ece_after,
            highest_fragility_feature=d.ablation.highest_fragility_feature,
            mean_perturbation_flip_rate=flip_rate,
            mean_perturbation_js_divergence=mean_js,
        )

        abstention = AbstentionReport(
            abstained=stability.abstain,
            reasons=stability.abstain_reasons,
            indeterminate_probability=float(
                phenotype.phenotype_probabilities.get(INDETERMINATE, 0.0)
            ),
        )

        warnings = list(
            dict.fromkeys(phenotype.warnings + stability.warnings + d.selection.warnings)
        )
        for axis in axes.values():
            warnings.extend(axis.warnings)

        return PmosResearchOutput(
            patient_id=patient_id,
            adapter_version=self.config.model_version,
            source_dataset=self.config.source_dataset,
            generated_at=datetime.now(UTC).isoformat(),
            observed_evidence=observed,
            model_organized_phenotype=organized,
            missing_evidence=missing,
            uncertainty=uncertainty,
            abstention=abstention,
            limitations=list(LIMITATIONS),
            warnings=list(dict.fromkeys(warnings)),
        )
