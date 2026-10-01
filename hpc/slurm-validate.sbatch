#!/bin/bash
# ---------------------------------------------------------------------------
# Validate the references against high precision: one array task per shard.
#
# Runs after the merge (submit.sh chains it with --dependency=afterok), because
# a reference is validated from the merged store -- each result's exact points,
# disputes, worst inputs and bands, and every earlier counterexample. Each task
# writes its own write-once validation record into that store under its own
# validation id, so tasks never touch each other's files and no second merge
# is needed.
#
# The shards split *reference units* -- distinct reference identities -- not
# cells: an anvl cell and its JAX twin share a reference and are validated
# once, from both cells' inputs. A shard that fails or times out leaves its
# references unvalidated, which excludes nothing: the export still works, and
# is merely more conservative. Re-run just this step with
# ./submit.sh --validate-only.
# ---------------------------------------------------------------------------
#SBATCH --job-name=anvl-validate
#SBATCH --partition=compute            # CHANGE ME (or let submit.sh set it)
#SBATCH --array=1-32
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/validate-%A_%a.out
#SBATCH --error=logs/validate-%A_%a.err

set -euo pipefail
source "${CONFIG:-${SLURM_SUBMIT_DIR}/config.sh}"

mkdir -p "${SWEEP_ROOT}/tmp" "${SWEEP_ROOT}/home"

# Bound to its CPU like the sweep tasks (see slurm-sweep.sbatch); MPFR is
# single-threaded, so one core is all a task uses.
SRUN=(srun --cpus-per-task="${SLURM_CPUS_PER_TASK:-1}" --cpu-bind=cores)

echo "== validation shard ${SLURM_ARRAY_TASK_ID}/${VALIDATE_SHARDS} on $(hostname)"
"${SRUN[@]}" "${SINGULARITY}" exec --cleanenv \
  --bind "${SWEEP_ROOT}:/sweeps" \
  --env "NV_SWEEP_STORE=/sweeps/store" \
  --env "HOME=/sweeps/home" \
  --env "TMPDIR=/sweeps/tmp" \
  --env "OMP_NUM_THREADS=1" \
  --env "OPENBLAS_NUM_THREADS=1" \
  "${SIF}" \
  anvl-sweep validate-refs \
    --backends "${BACKENDS}" \
    ${FILTER:+--filter "${FILTER}"} \
    --shard "${SLURM_ARRAY_TASK_ID}" \
    --shards "${VALIDATE_SHARDS}"

echo "validation shard ${SLURM_ARRAY_TASK_ID}/${VALIDATE_SHARDS} done"
