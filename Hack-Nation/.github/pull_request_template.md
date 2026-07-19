## What changed?

<!-- One paragraph. -->

## Why is it scientifically valid?

<!-- What justifies this choice? If it changes a modeling decision, what would
falsify it? -->

## Which module is affected?

- [ ] schemas / registry
- [ ] ingestion
- [ ] event store
- [ ] features
- [ ] models — tabular / phenotype / stability / speech / documents / ultrasound / temporal
- [ ] PMOS adapter
- [ ] training / evaluation
- [ ] docs / CI

## Which schema or data contract changed?

<!-- If any: state the version bump, backward compatibility, and the
registry/schema_versions.yaml + CHANGELOG entries. "None" is a valid answer. -->

## Which tests were added?

## Which documentation was updated?

## Does this change alter any medical or scientific claim?

<!-- Answer explicitly. If yes, say which claim, and what evidence now supports
it. This is the question most likely to block the PR. -->

---

### Model PRs must also include

- [ ] Config file
- [ ] Tests
- [ ] Documentation
- [ ] Artifact contract satisfied
- [ ] Metric definitions
- [ ] Known limitations
- [ ] Reproduction command

### Checklist

- [ ] `make ci` passes
- [ ] Preprocessing is fitted inside the training fold
- [ ] Longitudinal splits are grouped by participant
- [ ] No clinical data committed
- [ ] Output language stays research-oriented and non-diagnostic
