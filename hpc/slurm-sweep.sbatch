#!/bin/bash
# ---------------------------------------------------------------------------
# The sweep itself: one array task per shard of the grid.
#
# Submit via ./submit.sh, which fills the queue name, walltime and array size
# from config.sh and overrides the directives below on the command line. The
# directives are here so that a bare `sbatch slurm-sweep.sbatch` also works
# once you have edited them.
#
# The grid is canonically ordered by cell_id, so "shard i of N" selects the
# same cells on every machine with no coordination, and every cell writes its
# own Parquet file -- no locking, no shared handle, nothing to serialise.
# ---------------------------------------------------------------------------
#SBATCH --job-name=anvl-sweep
#SBATCH --partition=compute            # CHANGE ME (or let submit.sh set it)
#SBATCH --array=1-40
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=logs/sweep-%A_%a.out
#SBATCH --error=logs/sweep-%A_%a.err

set -euo pipefail

# submit.sh passes CONFIG through --export; a hand-submitted job falls back to
# the directory it was submitted from. ($0 is the spool copy under Slurm, so it
# is no use for locating anything.)
source "${CONFIG:-${SLURM_SUBMIT_DIR}/config.sh}"

# calibrate.sh sets these to run one shard at a shallower depth without anyone
# having to edit config.sh. Unset in a normal submission.
DEPTH="${DEPTH_OVERRIDE:-${DEPTH}}"
JOBS="${JOBS_OVERRIDE:-${JOBS}}"

# Where this array's shards land. Per array job id, so two submissions can
# never mix, and so a half-finished array is never merged by accident.
PARTS="${SWEEP_ROOT}/parts/${SLURM_ARRAY_JOB_ID}"
mkdir -p "${PARTS}" "${SWEEP_ROOT}/tmp" "${SWEEP_ROOT}/home"

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

# Recorded per shard because it is the only way to find out whether the above
# is actually true at this site -- and it has to be read from inside the
# container, which is what is bound.
echo "== shard ${SLURM_ARRAY_TASK_ID}/${SHARDS} on $(hostname)"
"${SRUN[@]}" "${SINGULARITY}" exec --cleanenv --bind "${SWEEP_ROOT}:/sweeps" "${SIF}" \
  sh -c 'grep Cpus_allowed_list /proc/self/status; nproc' || true

# HOME and TMPDIR are pointed at the bind mount deliberately: R, reticulate and
# XLA all write caches under HOME, and a quota'd home directory on a shared
# filesystem is the classic way for task 37 of 40 to die at hour six.
#
# The thread-count variables are pinned to 1, not to CPUS_PER_TASK: they reach
# R's BLAS, which this workload barely touches, and a BLAS pool sized to the
# whole allocation would just contend with XLA's pool for the same cores.
"${SRUN[@]}" "${SINGULARITY}" exec --cleanenv \
  --bind "${SWEEP_ROOT}:/sweeps" \
  --env "NV_SWEEP_STORE=/sweeps/parts/${SLURM_ARRAY_JOB_ID}" \
  --env "NV_SWEEP_DEVICE=cpu" \
  --env "HOME=/sweeps/home" \
  --env "TMPDIR=/sweeps/tmp" \
  --env "OMP_NUM_THREADS=1" \
  --env "OPENBLAS_NUM_THREADS=1" \
  --env "XLA_FLAGS=--xla_force_host_platform_device_count=1" \
  "${SIF}" \
  anvl-sweep run \
    --depth "${DEPTH}" \
    --backends "${BACKENDS}" \
    ${FILTER:+--filter "${FILTER}"} \
    --jobs "${JOBS}" \
    --quiet \
    --shard "${SLURM_ARRAY_TASK_ID}" \
    --shards "${SHARDS}"

echo "shard ${SLURM_ARRAY_TASK_ID}/${SHARDS} done -> ${PARTS}"
