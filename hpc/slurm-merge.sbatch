#!/bin/bash
# ---------------------------------------------------------------------------
# Fold a finished array's shards into the analysis store. Submitted by
# submit.sh with --dependency=afterok:<array job>, so it cannot run on a
# partial array: status counts cells against the declared grid, and a
# half-merged run reads as "never run" for everything still missing.
#
# Merging is a file copy -- the store is write-once, one Parquet file per
# (run, cell) -- so this is minutes, not hours.
# ---------------------------------------------------------------------------
#SBATCH --job-name=anvl-merge
#SBATCH --partition=compute            # CHANGE ME (or let submit.sh set it)
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/merge-%j.out
#SBATCH --error=logs/merge-%j.err

set -euo pipefail
source "${CONFIG:-${SLURM_SUBMIT_DIR}/config.sh}"

: "${ARRAY_JOB_ID:?set ARRAY_JOB_ID to the job id of the sweep array}"

"${SINGULARITY}" exec --cleanenv \
  --bind "${SWEEP_ROOT}:/sweeps" \
  --env "NV_SWEEP_STORE=/sweeps/store" \
  --env "NV_SWEEP_DEVICE=cpu" \
  --env "HOME=/sweeps/home" \
  --env "TMPDIR=/sweeps/tmp" \
  "${SIF}" \
  anvl-sweep merge --from "/sweeps/parts/${ARRAY_JOB_ID}"

"${SINGULARITY}" exec --cleanenv \
  --bind "${SWEEP_ROOT}:/sweeps" \
  --env "NV_SWEEP_STORE=/sweeps/store" \
  --env "HOME=/sweeps/home" \
  --env "TMPDIR=/sweeps/tmp" \
  "${SIF}" \
  anvl-sweep status
