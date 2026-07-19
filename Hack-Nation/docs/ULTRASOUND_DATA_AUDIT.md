# Ultrasound Data Audit

**Date:** 2026-07-18 · **Method:** direct inspection of files on disk.
Where this document and earlier repository documentation disagree, **this
document is correct** — every statement below was read from the filesystem, not
from a config or a README claim.

---

## 1. What is actually present

| Dataset | Path | Size | Usable for segmentation? |
|:--|:--|--:|:--|
| USOVA3D | `datasets/usova3d/raw/` | 228 MB | **Yes** — the only source with masks |
| PMOSGen (2D) | `datasets/ovarian_2d/raw/` | 188 MB | **No** — no masks, no spacing |

### USOVA3D

```
datasets/usova3d/raw/
├── split.json                     pre-existing 12-2-2 volume split
├── load_split.py
├── README.md                      per-volume black-slice statistics
└── <split>/                       split = train | val | test
    ├── images/Vol<N>/slice_NNN.png
    └── labels/Vol<N>/
        ├── follicle_r1/           binary mask, mode L, {0, 255}
        ├── follicle_r1_color/     RGB overlay, one colour per instance
        ├── follicle_r1_labels/    INSTANCE-ID mask, {0, 1, 2, ...}
        ├── follicle_r2/ …         same three variants, expert 2
        ├── ovary_r1/              binary mask, {0, 255}
        └── ovary_r2/              binary mask, {0, 255}
```

* **Volumes:** 16 total — 12 train / 2 val / 2 test (pre-existing `split.json`).
* **Slices:** 3,419 (train 2,593 · val 398 · test 428).
* **Image format:** 8-bit greyscale PNG (`mode='L'`), observed range 0–230.
* **Per-volume shape:** varies, `(Z, Y, X)` from `(181, 89, 199)` to `(247, 162, 208)`.
* **Ovary masks:** yes, binary, **two annotators** (`ovary_r1`, `ovary_r2`).
* **Follicle masks:** yes, binary **and instance-level**, two annotators.
* **Annotation type:** semantic *and* instance — `follicle_rN_labels` carries
  integer instance IDs, so instance metrics are computable without inventing a
  splitting heuristic.

### Two sub-cohorts, and they are not equivalent

| Volumes | `description` | Spacing (mm, isotropic) | n |
|:--|:--|:--|--:|
| Vol1–Vol6 | `UKCMB, UZ1`–`UZ6` | **0.25** | 6 |
| Vol101–Vol119 | `UZ-volumen` | **1.0** | 10 |

**The `1.0, 1.0, 1.0` spacing is almost certainly a placeholder, not a
measurement.** Exactly-unit isotropic spacing is the conventional default when
no calibration was recorded. This matters more than any modelling choice:

* every physical measurement (ovarian volume in mL, follicle diameter in mm,
  the 2–10 mm antral-follicle definition) is **unreliable for 10 of 16 volumes**;
* voxel-level Dice is unaffected;
* so **segmentation metrics are trustworthy and morphometric metrics are not**,
  for the majority of the cohort. This must be confirmed against the USOVA3D
  source before any mm-denominated number is reported.

---

## 2. Identifiers, laterality, and the grouping risk

* **There is no patient identifier.** The only key is `vol_id`. `meta.json`
  contains `vol_id, raw_file, shape_zyx, slice_axis, n_slices, spacing, origin,
  description, annotations` — nothing patient-level.
* **Laterality is not recorded.** No left/right field exists anywhere.
* **Whether two volumes belong to one patient cannot be determined from the
  data.** The `UZ1`–`UZ6` naming is consistent with six distinct studies, but
  nothing confirms that two volumes are not the left and right ovary of one
  person.

**Consequence for splitting.** We group by `vol_id` and treat volume = subject.
If any two volumes are in fact the same patient, a subject-level split is not
guaranteed and some leakage is possible. This is an **unresolvable-from-disk
limitation**, recorded in the split manifest and the model card rather than
papered over. It is *not* the same class of error as slice-level splitting,
which would be catastrophic and which we avoid outright.

---

## 3. Inter-annotator disagreement — the effective performance ceiling

Measured directly from `meta.json` `nonzero_voxels` and `max_label`:

| Volume | follicles r1 | follicles r2 | ovary voxel disagreement |
|:--|--:|--:|--:|
| Vol117 | 11 | 5 | 23.7% |
| Vol119 | 8 | 3 | 14.9% |
| Vol110 | 8 | 4 | 25.1% |
| Vol4 | 7 | 5 | 39.3% |
| Vol5 | 4 | 3 | 37.9% |
| Vol2 | 3 | 3 | 34.2% |

* **Exact follicle-count agreement: 5 of 16 volumes.**
* **Mean ovary-volume disagreement: 16.5%** (worst 39.3%).

Two trained experts disagree on the follicle count in **11 of 16 volumes**,
sometimes by more than 2×. Any model scored against a single arbitrarily-chosen
annotator inherits that annotator's idiosyncrasies. Therefore:

* the model is evaluated against **both** annotation sets and both numbers are
  reported;
* a Dice of ~0.8 against one rater is near the inter-rater ceiling, not a
  shortfall;
* published accuracies in the high 90s on this dataset should be read with the
  split protocol in mind — patch-level random splits of 16 volumes leak.

---

## 4. PMOSGen (2D) — present but unusable for segmentation

```
datasets/ovarian_2d/raw/
├── pmosgen_train/images/*.jpg     +  class_label.xlsx
└── pmosgen_test/images/*.jpg      +  "class label.csv"
```

* **4,679 JPEG images**, 300×300, no DICOM header, no calibration bar.
* Labels are **image-level only**: "Appears normal / abnormal" and "polycystic
  ovary Visible / Not-visible".
* **No segmentation masks. No pixel spacing.**

Empirically confirmed: running the existing heuristic segmenter over 30 PMOSGen
images yields a median of **208 "follicles" per image** (range 145–483) against a
clinical antral follicle count of 5–30. The cause is not the segmenter — without
mm calibration the 2 mm minimum-size filter cannot be applied, so speckle is
counted. `follicle_instances.py:200` emits precisely this warning.

**No segmentation model, of any architecture, can produce a follicle count from
this dataset.** The blocker is the absence of calibration, not model capacity.
`DATASET_REGISTRY.md:19` already lists `follicle_instance_segmentation` as
unsupported for this dataset; that entry is correct.

---

## 5. Licensing and redistribution

* **No licence file is present** in `datasets/usova3d/raw/` and there is no git
  remote configured.
* USOVA3D originates from UM FERI, Maribor (`usova3d.um.si`).
* **Checkpoint redistribution status: UNKNOWN and therefore treated as
  restricted.** Trained weights are written to `artifacts/` and are *not* to be
  published or redistributed until the upstream licence is confirmed. This is
  recorded in the model card.

---

## 6. Environment

| Item | Status |
|:--|:--|
| PyTorch | **2.13.0+cpu**, installed into `.venv` |
| CUDA on this node | **Not available** (`torch.cuda.is_available() == False`) |
| Cores on this node | **1** |
| SLURM GPU partitions | `l40s` (8), `a100` (3), `nvl`/`h100` (16 each) |

Real training runs on SLURM GPU; this node supports CPU smoke tests only.

---

## 7. Conclusions that drive the implementation

1. **USOVA3D is the only supervised source.** 16 volumes, ~12 for training.
2. **Keep the model small.** 12 training subjects cannot support a large network;
   capacity spent here buys memorisation, not accuracy.
3. **Group by volume, never by slice.** Adjacent slices of one volume are near-
   duplicates.
4. **Report against both annotators.** A single-rater number overstates certainty.
5. **Segmentation metrics yes; mm-denominated metrics conditionally.** Volume and
   diameter are only meaningful for the six 0.25 mm volumes until the spacing of
   the `UZ-volumen` cohort is confirmed.
6. **Two heads, not three-class softmax.** Follicles are structures *inside* the
   ovary, and follicle voxels are a small minority; a shared encoder with
   separate ovary and follicle heads avoids forcing a mutually-exclusive
   softmax over nested anatomy.
