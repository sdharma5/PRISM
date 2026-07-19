# Document pipeline (Step 7)

Converts laboratory and clinical reports into structured evidence with source
traceability. Supported: lab reports, ultrasound reports, medication lists, short
clinical summaries. Arbitrary full-EHR ingestion is deliberately out of scope.

## Pipeline

```text
PDF or image → text and table extraction → document classification →
candidate variable extraction → unit and date normalization →
reference-range extraction → source-page linking → human verification →
structured events → document token
```

## Both values are kept

```python
class ExtractedLabResult(BaseModel):
    canonical_test_code: str
    source_test_name: str
    value: float | str          # as printed
    source_unit: str | None     # as printed
    canonical_value: float | None
    canonical_unit: str | None
    reference_low: float | None
    reference_high: float | None
    source_page: int
    source_text: str
    unit_conversion_applied: bool
    ...
```

Unit normalization is registry-driven — `registry/units.yaml`, every factor unit
tested. Testosterone reported in nmol/L and in ng/dL must land on the same
canonical number, and the corpus deliberately contains both so the path is
exercised.

## Grounding is a hard gate

Every extracted value carries a page number and a text span. **A value that
cannot be grounded is dropped and counted as unsupported, never silently
added.** `unsupported-value rate` is a headline metric because a confidently
hallucinated lab value is worse than a missing one — the missing one is visible.

## Metrics

Test-name · numeric-value · unit · date · reference-range extraction ·
page grounding · unsupported-value rate.

Measured on 20–30 committed synthetic reports with manually verified ground
truth. Synthetic — no real-world performance claim.
