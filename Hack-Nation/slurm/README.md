# Slurm templates

Cluster submission for the steps that actually warrant a scheduler.

## First run on a new cluster

```bash
# 1. Clone and build the environment on the cluster (login node)
git clone https://github.com/AngelaNing1/Hack-Nation.git
cd Hack-Nation && make install

# 2. Point prism.env at your cluster
$EDITOR slurm/prism.env      # partitions, account, data root, module names

# 3. Verify BEFORE burning an allocation
sbatch slurm/smoke.sbatch
```

`smoke.sbatch` uses only synthetic fixtures and touches no real data. It exists
so a real job does not die at import time three hours into a queue.

## Templates

| Template | Step | Hardware | Array |
|:--|:--|:--|:--|
| `smoke.sbatch` | — | CPU ×4 | no |
| `train_static.sbatch` | 3 | CPU ×4 | no |
| `stability_sweep.sbatch` | 5 | CPU ×16 | **yes** — one seed per task |
| `train_ultrasound.sbatch` | 8 | **GPU** | no — stages are sequential |
| `train_temporal.sbatch` | 9 | GPU | **yes** — one fold per task |

Steps 4, 6 and 7 have no template on purpose: they finish faster than the queue
wait. Run them on a login node or locally.

## Why some are arrays and some are not

`stability_sweep` and `train_temporal` parallelize over independent units —
seeds and CV folds — so an array is a straight win.

`train_ultrasound` **cannot** be an array. Stage 2 fine-tunes stage 1's
checkpoint and stage 3 tracks using stage 2's segmenter. Running them
concurrently would fine-tune from a checkpoint that does not exist yet. To run a
single stage:

```bash
sbatch --export=ALL,STAGE=pretrain_2d slurm/train_ultrasound.sbatch
```

## Things that will bite you

**Editing a config while jobs are queued.** A pending job reads its config at
*start* time, not submit time. Editing in place silently changes what a queued
job runs — and the artifact will record the config it read, so the two disagree
with no error anywhere. Copy the config, edit the copy, submit that.

**A dirty working tree.** Every template records `git_commit`, but a dirty tree
means that hash does not describe what ran. `prism_record_provenance` warns;
heed it, because the warning is in the job log and nobody reads job logs of
successful runs.

**BLAS oversubscription.** `prism.env` pins `OMP_NUM_THREADS` to
`SLURM_CPUS_PER_TASK`. Without it, numpy spawns a thread per physical core
regardless of your allocation, and a 16-core job on a 128-core node will thrash.

**Output collisions.** Every template writes to a directory keyed by job and
array-task id. Do not "simplify" this — two array tasks writing one directory
produces silently interleaved artifacts.

**Credentialed data.** `train_temporal.sbatch` fails loudly if
`$PRISM_DATA_ROOT/mcphases` is missing rather than training on an empty frame.
Check your DUA covers running automated jobs against it.

## The one metric that gates the ultrasound job

`train_ultrasound.sbatch` exits non-zero if
`quality_gate_unsafe_acceptance_rate > 0`. That metric counts images accepted
for quantitative measurement that could not support one. It was 67% during
development, when noise volumes were being segmented into a confident blob and
measured. A job that produces measurements is not a job that succeeded.

## Container (optional)

If your cluster prefers containers to modules, build once:

```bash
apptainer build prism.sif container/prism.def
```

then uncomment the `PRISM_RUN` line in `prism.env`.
