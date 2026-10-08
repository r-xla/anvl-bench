#!/bin/bash
# ---------------------------------------------------------------------------
# After the workers: assemble any cell no worker finished assembling, then fold
# the queue's staging store into the analysis store. Submitted by submit.sh
# with --dependency=afterany:<worker array>, so it runs however the workers
# ended -- a worker killed at its time limit still leaves every finished part
# and cell behind, and the queue status this prints says what is missing.
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

: "${QUEUE_NAME:?set QUEUE_NAME to a queue under \$SWEEP_ROOT/queues}"

# finalise also prints the queue's status: parts done, stale, errored, left
"${SINGULARITY}" exec --cleanenv \
  --bind "${SWEEP_ROOT}:/sweeps" \
  --env "NV_SWEEP_STORE=/sweeps/parts/${QUEUE_NAME}" \
  --env "NV_SWEEP_DEVICE=cpu" \
  --home "${SWEEP_ROOT}/home" \
  --env "TMPDIR=/sweeps/tmp" \
  "${SIF}" \
  anvl-sweep finalise --queue "/sweeps/queues/${QUEUE_NAME}" --stale-minutes "${STALE_MINUTES}"

"${SINGULARITY}" exec --cleanenv \
  --bind "${SWEEP_ROOT}:/sweeps" \
  --env "NV_SWEEP_STORE=/sweeps/store" \
  --env "NV_SWEEP_DEVICE=cpu" \
  --home "${SWEEP_ROOT}/home" \
  --env "TMPDIR=/sweeps/tmp" \
  "${SIF}" \
  anvl-sweep merge --from "/sweeps/parts/${QUEUE_NAME}"

"${SINGULARITY}" exec --cleanenv \
  --bind "${SWEEP_ROOT}:/sweeps" \
  --env "NV_SWEEP_STORE=/sweeps/store" \
  --home "${SWEEP_ROOT}/home" \
  --env "TMPDIR=/sweeps/tmp" \
  "${SIF}" \
  anvl-sweep status
