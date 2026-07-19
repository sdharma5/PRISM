# Ultrasound pipeline (Step 8)

Produces semantic ovary and follicle measurements. The target is **not** a PMOS
diagnosis — it is segmentation, counting, morphology, and knowing when not to
measure.

## 2D is the primary pathway

Routine clinical PMOS ovarian assessment is **2D transvaginal imaging**. The
sonographer sweeps the probe through the ovary and reads follicles off individual
B-mode frames. A dedicated 3D volume acquisition is an optional extra that most
clinics do not perform.

The module is built around that reality. 2D segmentation, 2D quality control and
cross-frame follicle tracking are the default path; the 3D U-Net and volumetric
measurement are retained as an **optional enhanced mode** for the rare study that
arrives as a genuine volume.

USOVA3D is still used, but for what it actually is: one of very few public
datasets carrying expert ovary **and individual-follicle** labels. It is a
*pretraining and label resource*, not a model of the clinical input. Its volumes
are sliced into labelled 2D frames (see [Slice extraction](#usova3d-2d-slice-extraction)).

### Input priority

| Priority | Input | What it supports |
|---|---|---|
| 1 | 2D transvaginal **cine loop** | best realistic input; tracked unique-follicle estimate |
| 2 | **Multiple 2D stills** with measurements | per-section counts, weaker tracking |
| 3 | **Single 2D still** | per-section count and area only — limited output |
| 4 | **3D volume** | optional enhanced mode; the only true per-ovary count |

## Pipeline

```text
                    DICOM / frames / array
                             │
                    de-identification gate      (fail-loud; burned-in text refused)
                             │
              acquisition-mode detection        ← decides everything below
                             │
        ┌────────────────────┼────────────────────────┐
        │ single_frame       │ cine_loop / multi_frame│ volume_3d (optional)
        ▼                    ▼                        ▼
   preprocess           preprocess per frame     preprocess
        │                    │                        │
   ovary_detector_2d    ovary_detector_2d        segmenter_3d
        │                    │                        │
   segmenter_2d         segmenter_2d per frame   quality (3D gate)
        │                    │                        │
   qc_2d (frame)        qc_2d (per frame +       follicle_instances
        │                aggregate; usable count)     │
        │                    │                        │
        │               cine_tracking             morphology_3d
        │               (IoU + centroid + size         │
        │                → unique tracks)              │
        ▼                    ▼                        ▼
  morphology_2d        morphology_2d (cine)      volumetric measures
        │                    │                        │
        └────────────────────┴────────────────────────┘
                             │
                       output_schema
             (sets acquisition_mode + follicle_count_method)
                             │
                    encoder → ModalityToken
```

An ambiguous rank-3 array with no through-plane geometry is treated as a **cine
loop, never a volume**. The two errors are not symmetric: calling a real volume a
sweep merely forfeits measurements, while calling a sweep a volume fabricates an
ovarian volume and a per-ovary count that nothing in the data supports.

## The three-way counting distinction

This is the core of the module. Follicle counting is **three non-interchangeable
quantities**, and the method always travels with the number.

| Field | Meaning | Supported by |
|---|---|---|
| `follicle_number_per_section` | Follicles visible in **one cross-section** | any single frame |
| `estimated_follicle_number_per_ovary` | Unique follicles **estimated** by tracking across frames | cine loop / multi-frame |
| `follicle_number_per_ovary` | A **true** per-ovary count | full 3D volume only |

`OvarianMorphologyOutput.reportable_follicle_count` returns `(count, method)` as a
pair. Callers must never unpack only the integer — the method is what makes the
number interpretable.

The 2023 international guideline treats per-section and per-ovary counts as
distinct, and permits per-section counting precisely *because* complete counting
is often unreliable. Collapsing them into one integer would let a single still
frame silently claim a whole-ovary count.

### Outputs by input type

| | `single_frame` | `cine_loop` / `multi_frame` | `volume_3d` |
|---|---|---|---|
| `follicle_number_per_section` | yes | yes (representative frame) | — |
| `estimated_follicle_number_per_ovary` | **refused** | yes | — |
| `follicle_number_per_ovary` | **refused** | **refused** | yes |
| `ovary_area_mm2` | yes | yes (largest cross-section) | — |
| `ovary_volume_ml` | **refused** | withheld | yes |
| `ovary_dimensions_mm` (3-tuple) | — | — | yes |
| in-plane dimensions | — | yes (in warnings) | — |
| follicle size distribution | yes | yes | yes |
| `tracking_coverage`, `frames_analyzed` | — | yes | — |

The **refused** rows are enforced **twice, independently**:

1. the builders in `models/ultrasound/output_schema.py` never pass the forbidden
   field; and
2. the `model_validator` on `schemas/imaging.py:OvarianMorphologyOutput` raises if
   anything else tries to.

Neither guard substitutes for the other. `tests/unit/test_acquisition_mode_guards.py`
asserts both.

Note `ovary_dimensions_mm` stays `None` for 2D. It is a 3-tuple describing a
volume; a sweep measures two dimensions, and padding the third with a placeholder
would be a fabrication. The in-plane pair is reported in the warnings instead.

## Segmentation

2D U-Net with semantic classes `0 = background`, `1 = ovary`, `2 = follicle`
(`segmenter_2d.py`, the primary path), plus the optional 3D U-Net
(`segmenter_3d.py`). Both have a torch-free threshold + morphology fallback, so
the pipeline, its tracking and its gating run in CI without torch. The fallback is
not a toy: follicles are anechoic fluid inside echogenic stroma, and that contrast
is the same physical signal a learned model exploits.

The loss includes an "outside" term penalizing follicle predictions outside the
ovary:

$$ \mathcal{L}_{outside} = \sum_i P_i(\text{follicle})\left[1-P_i(\text{ovary})\right] $$

alongside ovary and follicle Dice, a boundary term, and the quality-head loss. It
needs no ground truth, so it regularises unlabelled studies too, and it applies
unchanged to 2D frames.

## Cine tracking — how a follicle on frames 3-7 is counted once

A pipeline that summed per-frame counts over a 60-frame sweep would report a
number an order of magnitude too large, and would report it confidently, because
every individual per-section count would be correct.

**The matching rule.** Each candidate on the current frame is compared against the
most recent observation of every active track. A pair is admissible only if it
passes two hard gates:

1. **Spatial continuity** — mask IoU >= `min_iou` **or** centroid displacement <=
   `max_centroid_shift_mm`. A disjunction: a small follicle moving by its own
   diameter has IoU 0 but is obviously the same follicle, while a large one can
   have high IoU despite a large displacement. Requiring both fragments small
   follicles; requiring neither links anything to anything.
2. **Size plausibility** — smaller/larger cross-sectional area ratio >=
   `min_size_ratio`. A two-fold jump between adjacent frames is a different
   structure.

Admissible pairs are scored `0.5*IoU + 0.3*proximity + 0.2*size_ratio` and assigned
greedily. Unmatched candidates start new tracks; tracks shorter than
`min_track_frames` are discarded.

**Failure modes — read before trusting the number.**

* **Re-imaging the same anatomy (over-count).** A probe that sweeps forward then
  back produces two disjoint tracks for one follicle. The tracker has no
  out-of-plane position and **cannot detect this**. This is the single largest
  reason the result is an estimate.
* **Fast probe motion (over-count).** Displacement beyond threshold with IoU 0
  breaks a track in two.
* **Adjacent similar-sized follicles (under-count).** Greedy assignment can swap
  identities or merge two follicles into one track.
* **Tangential planes of section.** A follicle clipped near its pole shrinks below
  the minimum-diameter filter and vanishes; `max_frame_gap` bridges brief dropouts
  but not long ones.
* **Incomplete sweeps (under-count).** `tracking_coverage` is **frame** coverage,
  not anatomical coverage. A perfectly tracked loop that imaged half the ovary
  reports coverage 1.0 and half the true count. No 2D acquisition can rule this
  out — which is the deeper reason a true per-ovary count needs a volume.
* **Single-frame follicles are dropped**, biasing towards under-counting. A
  one-frame blob is more often speckle than a follicle, and an inflated antral
  follicle count is the more consequential error.

Confidence degrades multiplicatively with coverage (`coverage^1.5`), loop length,
and the share of discarded short tracks.

## The quality gate

`qc_2d.py` for frames and loops, `quality.py` for volumes; route via
`assess_quality_for_mode`.

Predicts ovary visible / whole ovary visible / laterality available / pixel
spacing available / follicle counting feasible / ovarian volume feasible /
overall quality score.

**If quality is insufficient, the module abstains** — `measurement_feasible=False`,
`None` measurements, reasons attached. If pixel spacing is unknown, no physical
measurement is emitted at all. A follicle count from an image where half the ovary
is out of plane is not a low-confidence count; it is a different quantity.

The 3D gate reaches a **0.0 unsafe-acceptance rate** through two defensive checks,
both carried into 2D:

* an **ovary-fraction ceiling** (40% of a volume, 60% of a frame) — without it, a
  structureless noise image is split roughly in half by any thresholding step and
  the larger half is confidently measured as an ovary;
* an **ovary-vs-background contrast floor** — if candidate and surroundings have
  the same echogenicity, nothing has been detected, whatever the mask says.

The 2D ceiling is looser because an ovary *cross-section* legitimately fills far
more of a frame than an ovary fills a volume.

`ovarian_volume_feasible` is **always False** from `qc_2d`, regardless of image
quality: a cross-section has two dimensions and volume needs three.

For cine loops, per-frame assessments are aggregated by taking the **median score
over usable frames**, not the mean over all frames — a loop with 40 excellent
frames on the ovary and 60 approaching it is a good loop. The usable fraction is
reported separately as `tracking_coverage`.

## Instances and morphology

Connected components -> **watershed separation of touching follicles** (the
commonest cause of undercounting; applied in 2D as well as 3D) -> removal below a
documented physical-size threshold -> physical sizing via spacing -> cross-frame
tracking -> count and size distribution.

A track's diameter is its **maximum** cross-section across frames: the plane
through a follicle's centre gives its true diameter and every other plane
under-reports, so the mean would systematically under-size every follicle.

**Large or uncertain cystic structures** above a documented diameter threshold are
flagged, excluded from the small-follicle count, and sent for review. They are
never assigned a pathological diagnosis — this version does not attempt
ovarian-mass characterization.

## USOVA3D → 2D slice extraction

`ingestion/ultrasound/slice_extraction.py` cuts labelled volumes into labelled 2D
slices for pretraining the 2D segmenter.

**This is legitimate for learning appearance.** Stroma/lumen contrast, speckle
statistics and follicle cross-section shape are all genuine.

**It is not an independent test set**, for three reasons:

1. Adjacent slices of one volume share nearly all their anatomy — a slice-level
   split measures memorisation. Splitting is therefore **by subject**, enforced in
   `assign_slice_splits`.
2. A resliced plane is not a native 2D frame: reconstruction interpolation changes
   speckle texture and resolution anisotropy, and a model can exploit that.
3. A 3D probe sweep and a freehand 2D sweep sample the ovary at differently
   distributed angles.

**A test set should come from an independent 2D clinical source.** Every extracted
slice carries `VOLUME_DERIVED_PROVENANCE` so the caveat survives into artifacts.

## Staged training strategy

Declared in `configs/models/ultrasound_segmentation.yaml`. The ordering follows
from what data exists.

!!! warning "Declared, not implemented"

    This is a plan, not a capability. `scripts/train_ultrasound.py` prints these
    stages and then states that no weights are fit — it evaluates the assembled
    pipeline on phantoms. There is no `--stage` flag. Stage 2 needs a real 2D
    transvaginal dataset with a manually labelled subset, which this repository
    does not have.

| Stage | Data | Trains | Why |
|---|---|---|---|
| 1 — pretrain | USOVA3D-derived 2D slices | `segmenter_2d` | the only public source of individual-follicle labels; teaches ovarian appearance |
| 2 — fine-tune | real 2D transvaginal scans | `segmenter_2d` | the actual deployment input; low LR preserves stage-1 priors |
| 3 — cine tracking | annotated cine loops | `cine_tracking` (segmenter **frozen**) | joint training would let the segmenter produce whatever masks make tracking easy |

## Model-generated until reviewed

`OvarianMorphologyOutput.clinician_review_status` starts at `model_generated`.
Only a clinician moves it to `clinician_confirmed`, and until then every exported
token carries the warning "Clinician confirmation pending".

## Metrics

Counts are scored **per method and never pooled**:

* `per_section_count_mae` — follicles in one cross-section
* `unique_track_count_mae` (+ signed `unique_track_count_bias`) — distinct
  follicles across a loop
* `per_ovary_count_mae` — the true volumetric count

Averaging the first two would produce a number describing nothing and would hide a
model that reads frames well but tracks badly.

Tracking is scored further by the two **directional** failures, because they
cancel in a count:

* `tracking_fragmentation_rate` — excess tracks per true follicle (inflates)
* `tracking_merge_rate` — excess true follicles per track (deflates)

Plus: ovary Dice / follicle Dice / instance precision/recall / ovary-area MAPE /
ovarian-volume absolute error / false follicle voxels outside the ovary /
`tracking_coverage` / quality-gate sensitivity / **quality-gate unsafe acceptance
rate** (the safety metric, which must stay at 0).
