# Training guide

How to train every PRISM model, locally and on a cluster.

<div align="center">

**Nothing here is validated for clinical use. Smoke-run numbers are not results.**

</div>

---

## 0. Before you train anything

### Get the data

No clinical data ships with this repository. Obtain each dataset under its own
terms (see [`DATASET_REGISTRY.md`](DATASET_REGISTRY.md)) and store it **outside**
the repository tree.

```bash
cp .env.example .env
# edit .env:
#   PRISM_DATA_ROOT=/scratch/$USER/prism-data
```

Expected layout under `PRISM_DATA_ROOT`:

```text
$PRISM_DATA_ROOT/
├── pmos_tabular/pmos_data.csv
├── mcphases/                     # credentialed PhysioNet access + DUA
├── nhanes_2021_2023/
├── usova3d/                      # 3D volumes + ovary/follicle masks
└── ovarian_ultrasound_2d/        # primary 2D pathway (dataset TBD)
```

### Confirm the environment works

```bash
make install
make validate      # registries agree with each other
make test          # unit + contract, no data needed
make smoke         # tiny synthetic end-to-end runs
```

If `make smoke` passes, every pipeline executes. It says nothing about whether
any model is any good — that is the point of the rest of this document.

---

## 1. The order to train in

Dependencies are real; do not skip ahead.

```text
Step 3  static baselines ─────┐
                              ├──> Step 5  clustering + stability
Step 4  domains + autoencoder ┘
                                   (needs the label + an embedding)

Step 6  speech      ┐
Step 7  documents   ├── independent; no training dependency on the above
Step 8  ultrasound  │
Step 9  temporal    ┘
```

Steps 6–9 are genuinely independent — different datasets, different people.
They can run in parallel, and on a cluster they should.

---

## 2. Step 3 — static baselines

**Question it answers:** does the pipeline and evaluation methodology work at
all? The binary target is a methodology check, not a scientific goal.

```bash
python scripts/train_static_baselines.py \
  --config configs/experiments/exp_static_baselines.yaml
```

Runs logistic regression, random forest, gradient boosting, MLP, majority-class
and a single-feature rule, under repeated stratified patient-level CV across ≥5
seeds. Writes the full artifact directory.

**What to check before believing it:**
- Does the nonlinear model beat the **single-feature rule** by a meaningful
  margin? If not, you have a one-variable problem, not a machine-learning one.
- Is the calibration slope near 1.0? An AUROC of 0.85 with a slope of 0.4 is not
  a usable probability.
- Did any fold produce a wildly different metric? With small n that is normal
  and it is why ≥5 seeds are mandatory.

Runtime: minutes on CPU. **Do not send this to a GPU.**

---

## 3. Step 4 — phenotype domains + masked autoencoder

```bash
python scripts/train_tabular_autoencoder.py \
  --config configs/experiments/exp_phenotype_domains.yaml
```

Transparent domain scores are deterministic — they need no training. The
autoencoder masks 10–30% of observed variables and reconstructs them.

**The acceptance bar is explicit:** the embedding must reconstruct withheld
variables **better than mean imputation**. The script reports both. If it does
not beat the baseline, the embedding is adding parameters and nothing else —
do not carry it into Step 5.

Runtime: minutes on CPU (numpy Adam, no torch).

---

## 4. Step 5 — clustering and stability

```bash
python scripts/discover_phenotypes.py    --config configs/experiments/exp_subtype_stability.yaml
python scripts/run_stability_analysis.py --config configs/experiments/exp_subtype_stability.yaml
```

Sweeps representations × algorithms × K ∈ {2..6}, then bootstrap, ablation and
perturbation stability.

**Read the stability output before the cluster labels.** A configuration with a
good silhouette and a 0.4 flip rate has found noise. Two guards exist because
both failure modes are easy to hit:
- reproducibility alone cannot detect *absence* of structure — cutting one
  Gaussian blob in half is perfectly reproducible, so an absolute silhouette
  floor is applied separately;
- K is chosen on evidence. Four is never a default.

This is the most expensive CPU step: the sweep is embarrassingly parallel across
seeds and bootstraps. Good array-job candidate.

---

## 5. Steps 6–7 — speech and documents

```bash
python scripts/evaluate_speech_pipeline.py   --config configs/experiments/exp_speech_extraction.yaml
python scripts/evaluate_document_pipeline.py --config configs/experiments/exp_document_extraction.yaml
```

These are **evaluation**, not training — the default extractors are rule-based
and deterministic. Nothing to schedule; they run in seconds on a laptop.

**A warning that matters more than the numbers.** The current corpora are
synthetic and were written in the same session as the extractors, so the metrics
partly measure the extractor against its own assumptions. A first sweep scored
F1 1.0 for exactly this reason; adding cases the extractor genuinely fails
brought it to 0.9733 with `coverage_limit` recall at 0.1667.

If you extend the corpus, **write the utterances first, from real clinical
phrasing, before looking at the lexicon.** Otherwise you will reproduce the
circularity and the number will rise for no reason.

---

## 6. Step 8 — ultrasound (2D-primary)

### What runs today

```bash
python scripts/prepare_ultrasound.py --config configs/data/ultrasound.yaml
python scripts/train_ultrasound.py   --config configs/experiments/exp_ultrasound.yaml
```

`prepare_ultrasound.py` loads, de-identifies, validates and preprocesses each
study and writes `prepared_manifest.json` — the audit trail of which studies were
eligible for measurement at all. `train_ultrasound.py` runs the assembled
pipeline over the three acquisition pathways and scores each separately:

```bash
# Score one pathway at a time. Default is all three.
python scripts/train_ultrasound.py --config configs/experiments/exp_ultrasound.yaml \
  --modes single_frame
python scripts/train_ultrasound.py --config configs/experiments/exp_ultrasound.yaml \
  --modes cine_loop volume_3d
```

With no dataset present both run on synthetic phantoms, which is the only mode CI
uses. The phantoms carry exact ground-truth counts, frame spans, diameters and
volumes, so the reported numbers are genuine absolute error rather than "it
completed".

### What does NOT run yet

`configs/models/ultrasound_segmentation.yaml` declares a three-stage training
strategy under `training.stages` — pretrain on USOVA3D-derived slices, fine-tune
on real 2D scans, then cine-loop tracking. **That strategy is a declared plan,
not an implemented one.** `train_ultrasound.py` prints the declared stages and
then states plainly that no weights are fit; it evaluates the assembled pipeline
on phantoms. There is no `--stage` flag, and earlier revisions of this document
described one that never existed.

Implementing it needs a real 2D transvaginal dataset with a manually labelled
subset, which this repository does not have. Note also that a test set carved
from USOVA3D volumes is **not** independent 2D evaluation, and the registry
prohibits claiming it is.

### Read the counts correctly

The three quantities are not interchangeable and the schema refuses to confuse
them:

| Acquisition | May report | May **not** report |
|:--|:--|:--|
| Single 2D frame | follicles per cross-section, ovary area | any per-ovary count, ovarian volume |
| Cine loop / multi-frame | estimated unique per-ovary count, dimensions | a *true* per-ovary count |
| 3D volume | true per-ovary count, ovarian volume | — |

**The metric to watch is `quality_gate_unsafe_acceptance_rate`. It must stay at
0.0.** It measures how often the gate accepted an image it should have refused —
during development it was 67%, because noise volumes were being segmented into a
confident blob and measured. Anything above zero means the pipeline is producing
measurements from images that cannot support them.

GPU: yes, for segmentation training. CPU fallbacks exist but are for tests.

---

## 7. Step 9 — dynamic hormonal state

```bash
python scripts/prepare_mcphases.py --config configs/data/mcphases.yaml
python scripts/train_temporal.py   --config configs/experiments/exp_dynamic_state.yaml
```

GRU over the previous 14–30 participant-days. Heads: hormone reconstruction,
cycle phase, next-day symptoms, masked reconstruction.

**Splits are grouped by participant, always.** Adjacent days are near-duplicates;
splitting them randomly produces a model that has memorized each participant and
a metric that means nothing. `tests/unit/test_temporal_splits.py` fails if this
regresses — if you ever find yourself editing that test, stop.

**Run the missing-modality ablations.** The script reports performance with no
wearable, no CGM, no symptoms, and sparse hormones. A model that collapses
without CGM is a CGM model wearing a hormone model's name, and the degradation
table is what tells you which one you built.

GPU: helpful, not required. Small model, small dataset.

---

## 8. Cluster execution

### Resource shape

| Step | Hardware | Time | Parallel over |
|:--|:--|:--|:--|
| 3 static | CPU, 4 cores | minutes | seeds |
| 4 domains | CPU, 4 cores | minutes | — |
| 5 stability | CPU, 16+ cores | hours | seeds × bootstraps × K |
| 6–7 speech/docs | CPU, 2 cores | seconds | — |
| 8 ultrasound | **GPU** | hours | stages are sequential |
| 9 temporal | GPU or CPU | ~an hour | folds |

Steps 5, 8 and 9 are the only ones worth a scheduler. Steps 3, 4, 6 and 7 finish
faster than the queue wait.

### Submitting

Templates live in [`slurm/`](slurm/). Edit `slurm/prism.env` once, then:

```bash
sbatch slurm/train_static.sbatch
sbatch slurm/stability_sweep.sbatch      # array job over seeds
sbatch slurm/train_ultrasound.sbatch     # GPU
sbatch slurm/train_temporal.sbatch
```

Every template writes into `$PRISM_ARTIFACT_ROOT/<job>_<jobid>/` so concurrent
array tasks never collide on an output directory.

### Reproducibility on a cluster

Each run records its git commit, resolved config, environment, and seed. Two
cluster-specific rules:

1. **Never edit a config in place while jobs are queued.** Copy it, edit the
   copy, submit that. A queued job reads the config at *start* time, not at
   submit time — an in-place edit silently changes what a pending job runs.
2. **Set `PYTHONHASHSEED=0`** (the templates do). Python's hash randomization
   affects set and dict iteration order, which can perturb tie-breaking.

---

## 9. What to do when a result looks good

Be suspicious in this order:

1. **Leakage.** Was preprocessing fitted inside the fold? Did participant days
   straddle the split? Both have tests; run `make test` before believing a jump.
2. **Circular evaluation.** Was the evaluation set written by the same process
   that built the model? This has already happened twice in this repository —
   once in speech extraction, once in cluster stability.
3. **The baseline.** Does it beat majority-class and the single-feature rule?
4. **Calibration.** Discrimination without calibration is not a probability.
5. **Stability.** Does it survive a different seed?

Then write it down honestly, including what it does not show:

```bash
python scripts/build_model_card.py --experiment-dir artifacts/experiments/<ID>
```

The generated card never overwrites the human-reviewed `MODEL_CARD.md` — a
generated file cannot review its own claims.
