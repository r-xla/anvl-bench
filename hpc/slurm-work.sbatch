#!/bin/bash
# ---------------------------------------------------------------------------
# One worker of the pool: take parts of cells from the queue and sweep them
# until none are left, or until none would finish inside this job's time.
#
# Submit via ./submit.sh, which names the queue (QUEUE_NAME), fills the queue
# name, walltime and array size from config.sh, and overrides the directives
# below on the command line.
#
# Workers do not divide the grid between them in advance: each claims the next
# open part as it finishes the last (mkdir in the queue directory is the
# lock), so a worker that draws cheap parts simply takes more of them. Every
# part is written once under its own name, and the worker that completes a
# cell's last part assembles the cell into the staging store.
# ---------------------------------------------------------------------------
#SBATCH --job-name=anvl-work
#SBATCH --partition=compute            # CHANGE ME (or let submit.sh set it)
#SBATCH --array=1-64
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=24:00:00
#SBATCH --output=logs/work-%A_%a.out
#SBATCH --error=logs/work-%A_%a.err

set -euo pipefail

# submit.sh passes CONFIG through --export; a hand-submitted job falls back to
# the directory it was submitted from. ($0 is the spool copy under Slurm, so it
# is no use for locating anything.)
source "${CONFIG:-${SLURM_SUBMIT_DIR}/config.sh}"
: "${QUEUE_NAME:?set QUEUE_NAME to a queue under \$SWEEP_ROOT/queues}"

# calibrate.sh submits with a walltime of its own; config.sh's is the default.
WALLTIME="${WALLTIME_OVERRIDE:-${WALLTIME}}"

mkdir -p "${SWEEP_ROOT}/parts/${QUEUE_NAME}" "${SWEEP_ROOT}/tmp" "${SWEEP_ROOT}/home"

# ---- the deadline ---------------------------------------------------------
# The worker claims only parts it expects to finish before this, so the job
# ends on its own rather than being killed mid-part. Slurm's own end time when
# it will tell us; otherwise start + walltime. Five minutes are kept back for
# writing the last part and leaving.
end=$(squeue -h -j "${SLURM_JOB_ID}" -o %e 2>/dev/null || true)
until=""
if [[ -n "${end}" && "${end}" != *Unknown* ]]; then
  until=$(date -d "${end}" +%s 2>/dev/null || true)
fi
if [[ -z "${until}" ]]; then
  wt="${WALLTIME}"
  [[ "${wt}" == *-* ]] || wt="0-${wt}"
  IFS=- read -r d hms <<<"${wt}"
  IFS=: read -r h m s <<<"${hms}"
  until=$(($(date +%s) + 10#${d} * 86400 + 10#${h} * 3600 + 10#${m} * 60 + 10#${s}))
fi
until=$((until - 300))

# ---- CPU binding ----------------------------------------------------------
# What actually holds this task to its allocation is CPU *affinity*, not the
# thread-count environment variables below: XLA parallelises each kernel over
# an intra-op thread pool it sizes from the schedulable-CPU count
# (sched_getaffinity), and nothing in pjrt or anvl exposes a knob for it. When
# Slurm binds the task to N CPUs, the pool sizes to N and the kernel enforces
# it; measured on an unbound machine, one anvl process on one jitted kernel
# spreads over ~3.5 cores and 29 threads, so this matters.
#
# srun requests that binding explicitly rather than inheriting whatever the
# batch step happened to get, which depends on the site's TaskPlugin config.
SRUN=(srun --cpus-per-task="${SLURM_CPUS_PER_TASK:-${CPUS_PER_TASK}}" --cpu-bind=cores)

# Recorded per worker because it is the only way to find out whether the above
# is actually true at this site -- and it has to be read from inside the
# container, which is what is bound.
echo "== worker ${SLURM_ARRAY_TASK_ID:-1}/${WORKERS} of queue ${QUEUE_NAME} on $(hostname), until $(date -d "@${until}" 2>/dev/null || echo "${until}")"
"${SRUN[@]}" "${SINGULARITY}" exec --cleanenv --bind "${SWEEP_ROOT}:/sweeps" "${SIF}" \
  sh -c 'grep Cpus_allowed_list /proc/self/status; nproc' || true

# HOME and TMPDIR are pointed at the bind mount deliberately: R, reticulate and
# XLA all write caches under HOME, and a quota'd home directory on a shared
# filesystem is the classic way for worker 37 of 40 to die at hour six. HOME
# has to be set with --home: Singularity refuses `--env HOME=...` with only a
# warning in the .err file and leaves HOME as your real home directory.
#
# The SLURM_* job identity is passed in by name because --cleanenv strips it:
# the worker writes it into every claim, so a dead worker's parts can be traced
# to its array task.
#
# The thread-count variables are pinned to 1, not to CPUS_PER_TASK: they reach
# R's BLAS, which this workload barely touches, and a BLAS pool sized to the
# whole allocation would just contend with XLA's pool for the same cores.
# The worker runs in a loop. Its memory creeps up over its life by a few MB a
# cell (no cause found), so past WORKER_MAX_MEMORY MB it stops after the part
# it is on and exits with status 3, and a fresh R process -- which starts clean
# -- takes over in this same job: same allocation, no return to the queue. Any
# other exit ends the job: the queue is empty, nothing fits before the
# deadline, or something failed. Each R process is one step in sacct.
restarts=0
while true; do
  set +e
  "${SRUN[@]}" "${SINGULARITY}" exec --cleanenv \
    --bind "${SWEEP_ROOT}:/sweeps" \
    --env "NV_SWEEP_STORE=/sweeps/parts/${QUEUE_NAME}" \
    --env "NV_SWEEP_DEVICE=cpu" \
    --home "${SWEEP_ROOT}/home" \
    --env "TMPDIR=/sweeps/tmp" \
    --env "OMP_NUM_THREADS=1" \
    --env "OPENBLAS_NUM_THREADS=1" \
    --env "XLA_FLAGS=--xla_force_host_platform_device_count=1" \
    --env "SLURM_JOB_ID=${SLURM_JOB_ID}" \
    --env "SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID:-}" \
    --env "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-}" \
    "${SIF}" \
    anvl-sweep work \
      --queue "/sweeps/queues/${QUEUE_NAME}" \
      --until "${until}" \
      --stale-minutes "${STALE_MINUTES}" \
      ${WORKER_MAX_MEMORY:+--max-memory "${WORKER_MAX_MEMORY}"}
  rc=$?
  set -e
  # a restart with under ten minutes left would only start up and leave
  if ((rc == 3 && $(date +%s) < until - 600)); then
    restarts=$((restarts + 1))
    echo "== restart ${restarts}: a fresh worker process takes over (memory passed ${WORKER_MAX_MEMORY} MB)"
    continue
  fi
  break
done
echo "== worker ${SLURM_ARRAY_TASK_ID:-1} finished after ${restarts} restart(s)"
if ((rc == 3)); then rc=0; fi
exit "${rc}"
