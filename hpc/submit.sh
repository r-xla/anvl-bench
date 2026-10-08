#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Plan a queue, then submit the three steps, each waiting for the one before:
#
#   0. anvl-sweep plan        here, in the container: cut the grid into parts,
#                             into $SWEEP_ROOT/queues/<name>
#   1. slurm-work.sbatch      the worker pool, sweeping parts into
#                             $SWEEP_ROOT/parts/<name>
#   2. slurm-merge.sbatch     assemble what is left, fold into $SWEEP_ROOT/store
#   3. slurm-validate.sbatch  the validation array, checking the references
#                             against MPFR and recording it in the store
#
# then ./check.sh and ./export.sh by hand.
#
#   ./submit.sh                  plan a new queue and submit all three
#   ./submit.sh --dry-run        print the plan; write and submit nothing
#   ./submit.sh --resume <name>  more workers on an existing queue -- after a
#                                time limit, a crash, or to add capacity --
#                                then merge and validation again
#   ./submit.sh --validate-only  just step 3, against the store as it is now
#                                (e.g. after fixing a reference or a truth)
#
# Run this on the cluster, from the directory holding these scripts.
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
CONFIG="$(pwd)/config.sh"
source "${CONFIG}"

require_sif

LOGS="${SWEEP_ROOT}/logs"
mkdir -p "${LOGS}" "${SWEEP_ROOT}/queues" "${SWEEP_ROOT}/parts" "${SWEEP_ROOT}/store" \
         "${SWEEP_ROOT}/tmp" "${SWEEP_ROOT}/home" "${SWEEP_ROOT}/export"

# anvl-sweep inside the container, on this node, with the staging store of
# queue $1 (or none).
sweep() {
  local q="$1"
  shift
  "${SINGULARITY}" exec --cleanenv \
    --bind "${SWEEP_ROOT}:/sweeps" \
    --env "NV_SWEEP_STORE=/sweeps/parts/${q:-none}" \
    --home "${SWEEP_ROOT}/home" \
    --env "TMPDIR=/sweeps/tmp" \
    "${SIF}" \
    anvl-sweep "$@"
}

PLAN_ARGS=(--depth "${DEPTH}" --backends "${BACKENDS}"
  --unit-minutes "${UNIT_MINUTES}" --max-parts "${MAX_PARTS}")
if [[ -n "${FILTER}" ]]; then PLAN_ARGS+=(--filter "${FILTER}"); fi
if [[ -n "${COSTS}" ]]; then PLAN_ARGS+=(--costs "${COSTS}"); fi

# Check the plan before committing a pool to it: how many cells, how many
# parts, which cells were never measured. Seconds, and it runs nothing.
if [[ "${1:-}" == "--dry-run" ]]; then
  sweep "" plan --dry-run "${PLAN_ARGS[@]}"
  exit 0
fi

COMMON=(--partition="${PARTITION}")
if [[ -n "${ACCOUNT}" ]]; then COMMON+=(--account="${ACCOUNT}"); fi
if [[ -n "${QOS}"     ]]; then COMMON+=(--qos="${QOS}"); fi

submit_validate() {
  sbatch --parsable "${COMMON[@]}" "$@" \
    --export="ALL,CONFIG=${CONFIG}" \
    --array="1-${VALIDATE_SHARDS}" \
    --cpus-per-task=1 \
    --mem="${VALIDATE_MEM}" \
    --time="${VALIDATE_WALLTIME}" \
    --output="${LOGS}/validate-%A_%a.out" \
    --error="${LOGS}/validate-%A_%a.err" \
    slurm-validate.sbatch
}

if [[ "${1:-}" == "--validate-only" ]]; then
  VALIDATE_ID=$(submit_validate)
  echo "validation array: ${VALIDATE_ID}  (${VALIDATE_SHARDS} shards, against ${SWEEP_ROOT}/store)"
  echo "when it has finished:  ./check.sh, then ./export.sh"
  exit 0
fi

if [[ "${1:-}" == "--resume" ]]; then
  QUEUE_NAME="${2:?usage: ./submit.sh --resume <queue name>, one of: $(ls "${SWEEP_ROOT}/queues" | tr '\n' ' ')}"
  QDIR="${SWEEP_ROOT}/queues/${QUEUE_NAME}"
  [[ -f "${QDIR}/plan.rds" ]] || { echo "no queue at ${QDIR}" >&2; exit 1; }
  # Claims left by workers that are gone would otherwise hold their parts for
  # STALE_MINUTES. Released only once no worker of this queue is still queued
  # or running: a live worker's part would be swept twice.
  live=""
  if [[ -s "${QDIR}/arrays" ]]; then
    live=$(squeue -h -j "$(paste -sd, "${QDIR}/arrays")" -o %i 2>/dev/null || true)
  fi
  if [[ -n "${live}" ]]; then
    echo "workers of ${QUEUE_NAME} are still queued or running; adding more without releasing their claims"
  else
    sweep "${QUEUE_NAME}" queue --queue "/sweeps/queues/${QUEUE_NAME}" --release
  fi
else
  QUEUE_NAME="$(date +%Y%m%dT%H%M%S)"
  QDIR="${SWEEP_ROOT}/queues/${QUEUE_NAME}"
  sweep "${QUEUE_NAME}" plan --queue "/sweeps/queues/${QUEUE_NAME}" "${PLAN_ARGS[@]}"
fi

ARRAY_ID=$(sbatch --parsable "${COMMON[@]}" \
  --export="ALL,CONFIG=${CONFIG},QUEUE_NAME=${QUEUE_NAME}" \
  --array="1-${WORKERS}${WORKER_LIMIT:+%${WORKER_LIMIT}}" \
  --cpus-per-task="${CPUS_PER_TASK}" \
  --mem="${MEM_PER_TASK}" \
  --time="${WALLTIME}" \
  --output="${LOGS}/work-%A_%a.out" \
  --error="${LOGS}/work-%A_%a.err" \
  slurm-work.sbatch)
echo "${ARRAY_ID}" >> "${QDIR}/arrays"
echo "workers:     ${ARRAY_ID}  (${WORKERS} on queue ${QUEUE_NAME}, depth=${DEPTH}, backends=${BACKENDS})"

MERGE_ID=$(sbatch --parsable "${COMMON[@]}" \
  --dependency="afterany:${ARRAY_ID}" \
  --time="${MERGE_WALLTIME}" \
  --output="${LOGS}/merge-%j.out" \
  --error="${LOGS}/merge-%j.err" \
  --export="ALL,CONFIG=${CONFIG},QUEUE_NAME=${QUEUE_NAME}" \
  slurm-merge.sbatch)
echo "merge job:   ${MERGE_ID}  (afterany:${ARRAY_ID})"

VALIDATE_ID=$(submit_validate --dependency="afterok:${MERGE_ID}")
echo "validation:  ${VALIDATE_ID}  (${VALIDATE_SHARDS} shards, afterok:${MERGE_ID})"
echo
echo "queue:         ${QDIR}"
echo "staging store: ${SWEEP_ROOT}/parts/${QUEUE_NAME}"
echo "merged store:  ${SWEEP_ROOT}/store"
echo "logs:          ${LOGS}"
echo
echo "progress:      ./check.sh ${QUEUE_NAME}"
echo "when the validation array has finished:  ./check.sh, then ./export.sh"
