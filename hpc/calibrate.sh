#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Decide CPUS_PER_TASK from a measurement rather than a guess.
#
#   ./calibrate.sh submit [depth] [cpus]   run ONE shard; default quick, 4 cpus
#   ./calibrate.sh report <jobid>          CPU efficiency of that job
#
# The run writes into its own parts/<jobid> directory like any other, and merge
# takes an explicit ARRAY_JOB_ID, so a calibration run can never contaminate
# the analysis store. Delete it afterwards if you like; it costs nothing to
# leave, and `full` results supersede `quick` ones per cell anyway.
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
CONFIG="$(pwd)/config.sh"
source "${CONFIG}"

cmd="${1:-submit}"

case "${cmd}" in
submit)
  DEPTH_CAL="${2:-quick}"
  CPUS_CAL="${3:-${CPUS_PER_TASK}}"
  require_sif
  LOGS="${SWEEP_ROOT}/logs"
  mkdir -p "${LOGS}" "${SWEEP_ROOT}"/{parts,tmp,home}

  OPTS=(--partition="${PARTITION}")
  if [[ -n "${ACCOUNT}" ]]; then OPTS+=(--account="${ACCOUNT}"); fi
  if [[ -n "${QOS}"     ]]; then OPTS+=(--qos="${QOS}"); fi

  JOB=$(sbatch --parsable "${OPTS[@]}" \
    --job-name=anvl-calib \
    --array=1-1 \
    --cpus-per-task="${CPUS_CAL}" \
    --mem="${MEM_PER_TASK}" \
    --time=02:00:00 \
    --output="${LOGS}/calib-%A_%a.out" \
    --error="${LOGS}/calib-%A_%a.err" \
    --export="ALL,CONFIG=${CONFIG},DEPTH_OVERRIDE=${DEPTH_CAL},JOBS_OVERRIDE=1" \
    slurm-sweep.sbatch)

  echo "calibration job ${JOB}: shard 1/${SHARDS}, depth=${DEPTH_CAL}, cpus=${CPUS_CAL}"
  echo
  echo "  watch:   squeue -j ${JOB}"
  echo "  log:     ${LOGS}/calib-${JOB}_1.out"
  echo "  when it has FINISHED:  ./calibrate.sh report ${JOB}"
  ;;

report)
  JOB="${2:?usage: ./calibrate.sh report <jobid>}"
  state=$(sacct -j "${JOB}" --noheader -P -X --format=State | head -n 1 || true)
  echo "state: ${state:-unknown}"
  if [[ "${state}" == RUNNING* || "${state}" == PENDING* ]]; then
    echo
    echo "Still going -- TotalCPU is only final once the job has ended."
    echo "For a live look instead:  sstat -j ${JOB}.batch --format=AveCPU,MaxRSS"
    exit 0
  fi

  echo
  sacct -j "${JOB}" --format=JobID%20,State,Elapsed,TotalCPU,AllocCPUS,MaxRSS

  if command -v seff >/dev/null 2>&1; then
    echo; seff "${JOB}"; exit 0
  fi

  # No seff here, so compute it: efficiency = TotalCPU / (Elapsed x AllocCPUS).
  # TotalCPU comes formatted as [DD-]HH:MM:SS[.mmm] or MM:SS.mmm.
  #
  # -X: the allocation-level row, whose TotalCPU aggregates every step. Reading
  # a named step instead is a trap -- under srun the .batch step only launches
  # srun and books ~0 CPU, so keying on it reports 0% for a job that ran flat
  # out in step .0/.1.
  sacct -j "${JOB}" --noheader -P -X --format=JobID,TotalCPU,ElapsedRaw,AllocCPUS \
  | awk -F'|' '
      function secs(t,   d, p, n) {
        d = 0
        if (t ~ /-/) { split(t, p, "-"); d = p[1]; t = p[2] }
        n = split(t, p, ":")
        if (n == 3) return d*86400 + p[1]*3600 + p[2]*60 + p[3]
        if (n == 2) return d*86400 + p[1]*60 + p[2]
        return t + 0
      }
      $3 > 0 && $4 > 0 {
        used = secs($2); alloc = $3 * $4
        printf "\ncpu efficiency: %.0f%%  (%.0f cpu-seconds used of %.0f allocated, %d cores)\n", 100*used/alloc, used, alloc, $4
        printf "cores actually busy on average: %.2f\n", used/$3
      }'
  ;;

*) echo "usage: $0 {submit [depth] [cpus] | report <jobid>}" >&2; exit 2 ;;
esac
