# ADR-003: Hedged language for discovered phenotype profiles

**Status:** Accepted · **Date:** 2026-07-18

## Context

Unsupervised clustering on a PMOS cohort will produce groups. Those groups will
be interpretable — one will have higher BMI, glucose and insulin; another higher
LH and AMH. It is extremely tempting to call these "the metabolic subtype" and
"the LH–AMH subtype", especially because published work describes similar
groups, and especially because *K = 4* is a number the literature reports.

That temptation is the failure mode. A cluster is a partition of one dataset
under one representation, one algorithm, one K, and one seed. It is not a
validated clinical entity, and describing it as one converts an exploratory
result into an unsupported diagnostic claim.

## Decision

1. **K is chosen by evidence, not by publication.** `select_k()` evaluates
   K ∈ {2,…,6} across representations and algorithms on internal validity and
   stability. Four is never the default.
2. **Membership is probabilistic**, with an explicit `indeterminate` mass. No
   patient is forced into a group.
3. **Groups are labelled post hoc by enrichment**, and descriptions may only use
   hedged constructions: *resembles*, *is most similar to*, *has overlap with*.
4. **Banned phrases are enforced in code.** `models/phenotype/prototype_mapping.py`
   holds a banned-phrase list ("clinically validated subtype", "diagnosis",
   "confirmed subtype", …) and raises if a generated description contains one.
   `tests/unit/test_prototype_language.py` asserts the guard fires.
5. **Stability travels with the assignment.** Every profile export carries a
   stability score, a flip rate, the most fragile feature, and an abstention
   decision. A dominant probability of 0.61 with a flip rate of 0.21 is reported
   as exactly that, never as a category.

## Consequences

Outputs read as weaker than a subtype label — because they are weaker. The
honest phrasing is *"this participant's profile resembles a metabolic-leaning
pattern (p = 0.61, stability 0.73, unstable to removal of fasting insulin)"*.

The word "validated" becomes available again only if labels and genuine external
validation ever support it. Until then, the guard stays in the code path, not
just in the style guide — a reviewer skimming a diff will not catch a phrase
that a unit test will.
