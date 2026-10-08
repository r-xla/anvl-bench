#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Measure before committing a pool: what cells cost, and what a worker's cores
# are actually doing.
#
#   ./calibrate.sh submit [depth] [cpus] [workers]
#        sweep FILTER (or CALIBRATE_FILTER) at smoke depth by default, on 8
#        workers with CPUS_PER_TASK cpus each by default, through its own queue
#   ./calibrate.sh report <jobid>
#        that job's CPU efficiency, and the store to add to COSTS
#
# The costs are what let ./submit.sh cut each cell into parts of UNIT_MINUTES;
# a cell nothing has measured is cut into MAX_PARTS blindly. A smoke sweep is
# 1/8192 of a full one, so measuring the whole grid this way is cheap and
# needs no earlier campaign. Scaled up, a smoke time overstates the cost
# (compiling is a larger share of a short sweep), which only makes the parts
# smaller. To re-measure only what is new or changed since the last release:
#
#   CALIBRATE_FILTER="spec=nv_dbinom|nv_pbinom|nv_qbinom" ./calibrate.sh submit
#
# and list its store in COSTS *after* the release ZIP, so that it overrides it.
#
# The run goes into its own queue and staging store (calib-<time>), never into
# the analysis store, so a calibration can never contaminate a campaign.
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
CONFIG="$(pwd)/config.sh"
source "${CONFIG}"

cmd="${1:-submit}"

case "${cmd}" in
submit)
  DEPTH_CAL="${2:-smoke}"
  CPUS_CAL="${3:-${CPUS_PER_TASK}}"
  WORKERS_CAL="${4:-8}"
  FILTER_CAL="${CALIBRATE_FILTER:-${FILTER}}"
  require_sif
  LOGS="${SWEEP_ROOT}/logs"
  QUEUE_NAME="calib-$(date +%Y%m%dT%H%M%S)"
  mkdir -p "${LOGS}" "${SWEEP_ROOT}"/{queues,parts,tmp,home}

  "${SINGULARITY}" exec --cleanenv \
    --bind "${SWEEP_ROOT}:/sweeps" \
    --home "${SWEEP_ROOT}/home" \
    --env "TMPDIR=/sweeps/tmp" \
    "${SIF}" \
    anvl-sweep plan --queue "/sweeps/queues/${QUEUE_NAME}" --depth "${DEPTH_CAL}" \
      --backends "${BACKENDS}" ${FILTER_CAL:+--filter "${FILTER_CAL}"} --max-parts 1

  OPTS=(--partition="${PARTITION}")
  if [[ -n "${ACCOUNT}" ]]; then OPTS+=(--account="${ACCOUNT}"); fi
  if [[ -n "${QOS}"     ]]; then OPTS+=(--qos="${QOS}"); fi

  JOB=$(sbatch --parsable "${OPTS[@]}" \
    --job-name=anvl-calib \
    --array="1-${WORKERS_CAL}" \
    --cpus-per-task="${CPUS_CAL}" \
    --mem="${MEM_PER_TASK}" \
    --time="${CALIBRATE_WALLTIME:-08:00:00}" \
    --output="${LOGS}/calib-%A_%a.out" \
    --error="${LOGS}/calib-%A_%a.err" \
    --export="ALL,CONFIG=${CONFIG},QUEUE_NAME=${QUEUE_NAME},WALLTIME_OVERRIDE=${CALIBRATE_WALLTIME:-08:00:00}" \
    slurm-work.sbatch)
  echo "${JOB}" > "${SWEEP_ROOT}/queues/${QUEUE_NAME}/arrays"

  echo "calibration job ${JOB}: queue ${QUEUE_NAME}, depth=${DEPTH_CAL}, cpus=${CPUS_CAL}, workers=${WORKERS_CAL}"
  echo "  filter: ${FILTER_CAL:-(everything)}"
  echo
  echo "  watch:   squeue -j ${JOB}"
  echo "  log:     ${LOGS}/calib-${JOB}_1.out"
  echo "  when it has FINISHED:  ./calibrate.sh report ${JOB}"
  ;;

report)
  JOB="${2:?usage: ./calibrate.sh report <jobid>}"
  # every task's state, counted: "8 COMPLETED", say
  state=$(sacct -j "${JOB}" --noheader -P -X --format=State 2>/dev/null | sort | uniq -c \
    | awk '{printf "%s%d %s", (NR > 1 ? ", " : ""), $1, $2}' || true)
  echo "state: ${state:-unknown}"
  if [[ "${state}" == *RUNNING* || "${state}" == *PENDING* ]]; then
    echo
    echo "Still going -- TotalCPU is only final once the job has ended."
    echo "For a live look instead:  sstat -j ${JOB}.batch --format=AveCPU,MaxRSS"
    exit 0
  fi

  echo
  sacct -j "${JOB}" --format=JobID%20,State,Elapsed,TotalCPU,AllocCPUS,MaxRSS

  q=$(grep -lx "${JOB}" "${SWEEP_ROOT}"/queues/calib-*/arrays 2>/dev/null | head -n 1 || true)
  if [[ -n "${q}" ]]; then
    name=$(basename "$(dirname "${q}")")
    echo
    echo "costs: /sweeps/parts/${name}"
    echo "  add it to COSTS in config.sh for the next ./submit.sh -- last, so that it"
    echo "  overrides any earlier source for the cells it measured"
  fi

  # Efficiency = CPU time used / (elapsed x cores allocated), over every task
  # of the job together, and each task's peak memory.
  #
  # CPU time and memory are read from the job *steps* (.batch, .extern, and
  # the srun steps .0, .1, ...) and added up per task, never from the
  # allocation row. Sites differ on that row: some roll the steps' TotalCPU up
  # into it and some leave it at 00:00:00 (sacct -X), which read as 0% for
  # workers that ran flat out. Elapsed time and cores do come from the
  # allocation row. TotalCPU is [DD-]HH:MM:SS[.mmm] or MM:SS.mmm; MaxRSS is a
  # number with a K, M or G suffix.
  sacct -j "${JOB}" --noheader -P --format=JobID,TotalCPU,ElapsedRaw,AllocCPUS,MaxRSS \
  | awk -F'|' '
      function secs(t,   d, p, n) {
        d = 0
        if (t ~ /-/) { split(t, p, "-"); d = p[1]; t = p[2] }
        n = split(t, p, ":")
        if (n == 3) return d*86400 + p[1]*3600 + p[2]*60 + p[3]
        if (n == 2) return d*86400 + p[1]*60 + p[2]
        return t + 0
      }
      function mb(r,   v, u) {
        v = r + 0; u = substr(r, length(r))
        if (u == "K") return v / 1024
        if (u == "G") return v * 1024
        if (u == "M") return v
        return v / 1048576
      }
      {
        task = $1; sub(/\..*/, "", task)
        if ($1 == task) { elapsed[task] = $3; cpus[task] = $4; next }
        used[task] += secs($2)
        if (mb($5) > peak[task]) peak[task] = mb($5)
      }
      END {
        for (t in elapsed) {
          n++; u += used[t]; a += elapsed[t] * cpus[t]; e += elapsed[t]
          # insertion sort: a calibration has a handful of tasks
          for (i = n; i > 1 && pk[i - 1] > peak[t]; i--) pk[i] = pk[i - 1]
          pk[i] = peak[t]
        }
        if (n == 0 || a == 0) { print "\nno finished tasks to measure"; exit }
        printf "\ncpu efficiency: %.0f%%  (%.0f cpu-seconds used of %.0f allocated, %d task(s))\n", 100*u/a, u, a, n
        printf "cores actually busy per task on average: %.2f\n", u/e
        printf "peak memory per task: median %.0f MB, max %.0f MB (MEM_PER_TASK must exceed the max)\n", pk[int((n + 1) / 2)], pk[n]
      }'
  ;;

*) echo "usage: $0 {submit [depth] [cpus] [workers] | report <jobid>}" >&2; exit 2 ;;
esac
