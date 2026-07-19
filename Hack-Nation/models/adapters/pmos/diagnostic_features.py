"""The recognized PMOS feature axes, their inputs, and their documented thresholds.

Scientific WHY
--------------
PMOS has no single confirmatory test. Consensus frameworks describe it through a
small number of *feature axes* — ovulatory dysfunction, biochemical
hyperandrogenism, clinical hyperandrogenism, polycystic ovarian morphology — each
of which is assessed separately and each of which has real threshold controversy.

This module encodes those axes as **research-only feature assessments**, with
three hard rules:

1. Every threshold carries its source string. A number without a provenance is
   not usable evidence.
2. When an axis's inputs are missing, the assessment is ``not_assessable``. It is
   never ``not_met``. Treating an unmeasured axis as absent is the mechanism by
   which under-investigated patients get told they are fine, and it is the most
   consequential silent bug possible in this domain.
3. Nothing here is a diagnosis. An axis being "met" means a measured value
   crossed a documented research threshold, nothing more. Diagnosis additionally
   requires exclusion of other causes (thyroid disease, hyperprolactinaemia,
   non-classical congenital adrenal hyperplasia, Cushing syndrome), which no
   dataset here supports.

Assay-dependence caveat
-----------------------
Biochemical androgen thresholds are **not transferable between laboratories**.
Direct immunoassays for total and free testosterone perform poorly at female
concentrations, and consensus guidance recommends high-quality assays
(LC-MS/MS, extraction/chromatography) with laboratory-specific reference ranges
derived from a well-phenotyped population. The defaults below are therefore
placeholders that must be overridden per dataset, and the assessment records
``assay_dependent=True`` so a downstream report can say so out loud.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

__all__ = [
    "FEATURE_AXES",
    "AxisAssessment",
    "AxisSpec",
    "ThresholdRule",
    "assess_all_axes",
    "assess_axis",
]

AxisStatus = Literal["met", "not_met", "not_assessable"]


@dataclass(frozen=True)
class ThresholdRule:
    """One measurable condition contributing to an axis."""

    code: str
    comparator: Literal[">=", ">", "<=", "<", "==", "truthy"]
    threshold: float | None
    source: str
    assay_dependent: bool = False
    notes: str = ""

    def evaluate(self, value: float | bool | None) -> bool | None:
        """True/False when assessable, ``None`` when the input is missing."""
        if value is None:
            return None
        if self.comparator == "truthy":
            return bool(value)
        if self.threshold is None:
            return None
        v = float(value)
        t = float(self.threshold)
        return {
            ">=": v >= t,
            ">": v > t,
            "<=": v <= t,
            "<": v < t,
            "==": v == t,
        }[self.comparator]


@dataclass(frozen=True)
class AxisSpec:
    """A recognized PMOS feature axis: its rules and how they combine."""

    name: str
    label: str
    description: str
    rules: tuple[ThresholdRule, ...]
    #: "any" — one satisfied rule is enough (the usual case: several equivalent
    #: operationalizations of the same clinical concept).
    combination: Literal["any"] = "any"
    caveats: tuple[str, ...] = ()

    @property
    def codes(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(rule.code for rule in self.rules))


FEATURE_AXES: dict[str, AxisSpec] = {
    "ovulatory_dysfunction": AxisSpec(
        name="ovulatory_dysfunction",
        label="Ovulatory dysfunction",
        description=(
            "Irregular or absent ovulation, operationalized through menstrual cycle "
            "pattern. Cycle-based criteria are age-dependent because cycles are "
            "normally irregular for the first years after menarche and again in the "
            "perimenopausal transition."
        ),
        rules=(
            ThresholdRule(
                code="cycle_length",
                comparator=">",
                threshold=35.0,
                source=(
                    "2023 International Evidence-based Guideline for the Assessment and "
                    "Management of PMOS: cycles >35 days in adults are irregular."
                ),
            ),
            ThresholdRule(
                code="cycle_length",
                comparator="<",
                threshold=21.0,
                source=(
                    "2023 International Evidence-based Guideline: cycles <21 days in "
                    "adults are irregular."
                ),
            ),
            ThresholdRule(
                code="menstrual_frequency_per_year",
                comparator="<",
                threshold=9.0,
                source=(
                    "Rotterdam 2003 / 2023 International Guideline: fewer than 8-9 "
                    "cycles per year indicates oligo-anovulation."
                ),
            ),
            ThresholdRule(
                code="cycle_irregularity",
                comparator="truthy",
                threshold=None,
                source="Participant- or clinician-reported cycle irregularity flag.",
                notes="Report-based evidence, weaker than a measured cycle length.",
            ),
            ThresholdRule(
                code="amenorrhea",
                comparator="truthy",
                threshold=None,
                source=(
                    "2023 International Guideline: primary or secondary amenorrhoea is "
                    "ovulatory dysfunction by definition."
                ),
            ),
        ),
        caveats=(
            "Regular cycles do not exclude anovulation; confirmation requires a "
            "mid-luteal progesterone measurement, which most datasets here lack.",
            "Hormonal contraception masks the natural cycle and invalidates this axis.",
        ),
    ),
    "hyperandrogenism_biochemical": AxisSpec(
        name="hyperandrogenism_biochemical",
        label="Biochemical hyperandrogenism",
        description=(
            "Elevated circulating androgens. Calculated free testosterone or the free "
            "androgen index is preferred over total testosterone because SHBG varies "
            "widely with adiposity and insulin resistance."
        ),
        rules=(
            ThresholdRule(
                code="total_testosterone",
                comparator=">",
                threshold=2.5,
                source=(
                    "Placeholder upper reference limit in nmol/L for adult females. "
                    "Assay- and laboratory-specific; must be replaced with the "
                    "reference range of the measuring laboratory."
                ),
                assay_dependent=True,
            ),
            ThresholdRule(
                code="free_testosterone",
                comparator=">",
                threshold=0.031,
                source=(
                    "Placeholder upper reference limit in nmol/L for calculated free "
                    "testosterone. Assay- and calculation-formula-specific."
                ),
                assay_dependent=True,
            ),
            ThresholdRule(
                code="dheas",
                comparator=">",
                threshold=9.2,
                source=(
                    "Placeholder upper reference limit in umol/L. DHEAS is strongly "
                    "age-dependent and is a secondary marker only."
                ),
                assay_dependent=True,
                notes="Marked adrenal androgen excess should prompt exclusion of other causes.",
            ),
        ),
        caveats=(
            "Hormonal contraception suppresses androgens and raises SHBG; values "
            "measured on treatment cannot assess this axis.",
            "Direct free-testosterone immunoassays are unreliable at female "
            "concentrations and should not be used for this assessment.",
        ),
    ),
    "hyperandrogenism_clinical": AxisSpec(
        name="hyperandrogenism_clinical",
        label="Clinical hyperandrogenism",
        description=(
            "Androgen excess visible on examination: hirsutism, acne, or female "
            "pattern hair loss. Hirsutism scoring cut-offs differ substantially by "
            "ancestry because terminal hair density does."
        ),
        rules=(
            ThresholdRule(
                code="ferriman_gallwey_score",
                comparator=">=",
                threshold=4.0,
                source=(
                    "2023 International Guideline: modified Ferriman-Gallwey cut-offs "
                    "of 4-6 depending on ethnicity; the lower bound is used here so "
                    "the axis is not silently insensitive in populations with lower "
                    "baseline terminal hair density."
                ),
                notes="Self-scored mFG is less reliable than clinician scoring.",
            ),
            ThresholdRule(
                code="hirsutism",
                comparator="truthy",
                threshold=None,
                source="Clinician- or participant-reported hirsutism flag.",
            ),
            ThresholdRule(
                code="androgenic_alopecia",
                comparator="truthy",
                threshold=None,
                source=(
                    "Female pattern hair loss; low specificity in isolation per the "
                    "2023 International Guideline."
                ),
            ),
        ),
        caveats=(
            "Acne alone is a poor discriminator in adolescents and is not sufficient "
            "for this axis.",
            "Cosmetic hair removal masks hirsutism and biases this axis toward 'not met'.",
        ),
    ),
    "polycystic_ovarian_morphology": AxisSpec(
        name="polycystic_ovarian_morphology",
        label="Polycystic ovarian morphology",
        description=(
            "Ultrasound appearance of multiple small antral follicles and/or "
            "increased ovarian volume. Thresholds depend directly on transducer "
            "frequency, so a count from an older or transabdominal scan is not "
            "comparable to one from a modern transvaginal scan."
        ),
        rules=(
            ThresholdRule(
                code="follicle_number_per_ovary",
                comparator=">=",
                threshold=12.0,
                source=(
                    "Rotterdam 2003 consensus: FNPO >=12 on either ovary. "
                    "The 2023 International Guideline raised this to >=20 for modern "
                    "high-frequency transvaginal transducers; 12 is used here as the "
                    "conservative threshold for patch-classifier-derived counts."
                ),
                notes="Not applicable within 8 years of menarche.",
            ),
            ThresholdRule(
                code="ovary_volume_ml",
                comparator=">=",
                threshold=10.0,
                source=(
                    "Rotterdam 2003 and retained in the 2023 International Guideline: "
                    "ovarian volume >=10 mL on either ovary, with no dominant follicle, "
                    "corpus luteum or cyst present."
                ),
            ),
            ThresholdRule(
                code="anti_mullerian_hormone",
                comparator=">",
                threshold=None,
                source=(
                    "The 2023 International Guideline accepts serum AMH as an "
                    "alternative to ultrasound for defining PCOM in adults, but "
                    "explicitly gives no universal cut-off: thresholds are assay- and "
                    "population-specific. No default threshold is encoded, so this "
                    "rule stays not-assessable until a dataset supplies one."
                ),
                assay_dependent=True,
            ),
        ),
        caveats=(
            "Should not be assessed within 8 years of menarche, when multifollicular "
            "ovaries are common and non-pathological.",
            "Not assessable on hormonal contraception, which alters ovarian morphology.",
        ),
    ),
}


@dataclass
class AxisAssessment:
    """The outcome of assessing one feature axis for one participant."""

    axis: str
    status: AxisStatus
    evidence_available: bool
    observed_codes: list[str] = field(default_factory=list)
    missing_codes: list[str] = field(default_factory=list)
    satisfied_rules: list[str] = field(default_factory=list)
    threshold_sources: dict[str, str] = field(default_factory=dict)
    assay_dependent: bool = False
    caveats: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_met(self) -> bool:
        """True only for an affirmative assessment — never for not_assessable."""
        return self.status == "met"


def assess_axis(
    spec: AxisSpec,
    values: Mapping[str, float | bool | None],
    threshold_overrides: Mapping[str, float] | None = None,
) -> AxisAssessment:
    """Assess one axis, returning ``not_assessable`` whenever inputs are missing.

    A rule with no observed input contributes nothing. If *no* rule could be
    evaluated the axis is ``not_assessable``; if at least one rule was evaluated
    and any was satisfied the axis is ``met``; otherwise it is ``not_met`` — and
    even then we record which inputs were missing, because "not met on the two
    variables we happened to have" is materially weaker than "not met on all four".
    """
    overrides = dict(threshold_overrides or {})
    observed: list[str] = []
    missing: list[str] = []
    satisfied: list[str] = []
    sources: dict[str, str] = {}
    assay_dependent = False
    evaluated_any = False

    for rule in spec.rules:
        value = values.get(rule.code)
        threshold = overrides.get(rule.code, rule.threshold)
        effective = ThresholdRule(
            code=rule.code,
            comparator=rule.comparator,
            threshold=threshold,
            source=rule.source,
            assay_dependent=rule.assay_dependent,
            notes=rule.notes,
        )
        outcome = effective.evaluate(value)
        key = f"{rule.code} {rule.comparator} {threshold if threshold is not None else 'true'}"
        sources[key] = rule.source
        if outcome is None:
            if value is None:
                missing.append(rule.code)
            continue
        evaluated_any = True
        if rule.code not in observed:
            observed.append(rule.code)
        if rule.assay_dependent:
            assay_dependent = True
        if outcome:
            satisfied.append(key)

    missing = [c for c in dict.fromkeys(missing) if c not in observed]

    if not evaluated_any:
        status: AxisStatus = "not_assessable"
        warnings = [
            f"'{spec.name}' could not be assessed: none of {list(spec.codes)} were observed. "
            "This is reported as not_assessable and must never be read as absent."
        ]
    elif satisfied:
        status = "met"
        warnings = []
    else:
        status = "not_met"
        warnings = (
            [
                f"'{spec.name}' evaluated on a partial input set; missing {missing}. "
                "A negative assessment here is weaker than a fully measured one."
            ]
            if missing
            else []
        )

    if assay_dependent:
        warnings.append(
            f"'{spec.name}' used assay-dependent thresholds; these are placeholders "
            "unless overridden with the measuring laboratory's reference ranges."
        )

    return AxisAssessment(
        axis=spec.name,
        status=status,
        evidence_available=evaluated_any,
        observed_codes=observed,
        missing_codes=missing,
        satisfied_rules=satisfied,
        threshold_sources=sources,
        assay_dependent=assay_dependent,
        caveats=list(spec.caveats),
        warnings=warnings,
    )


def assess_all_axes(
    values: Mapping[str, float | bool | None],
    threshold_overrides: Mapping[str, Mapping[str, float]] | None = None,
) -> dict[str, AxisAssessment]:
    """Assess every recognized PMOS feature axis for one participant."""
    overrides = dict(threshold_overrides or {})
    return {
        name: assess_axis(spec, values, overrides.get(name)) for name, spec in FEATURE_AXES.items()
    }
