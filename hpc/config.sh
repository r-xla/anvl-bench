# ---------------------------------------------------------------------------
# The single place to edit. Sourced by build.sh, submit.sh and export.sh, and
# by the job scripts when they run under Slurm.
#
# Everything marked CHANGE ME is a placeholder that will not work as written.
# ---------------------------------------------------------------------------

# ---- image ----------------------------------------------------------------
IMAGE_NAME="anvl-sweeps"
# A LITERAL, never $(date): config.sh is sourced by every script, so a computed
# date tag changes the .sif path at midnight and the image built yesterday
# stops being found. Bump it by hand when you build a new image, or override it
# for one command:  IMAGE_TAG=v0.5.1 ./build.sh all
IMAGE_TAG="${IMAGE_TAG:-v0.5.1}"
DOCKER_PLATFORM="linux/amd64"        # the cluster's arch, not the Mac's
ANVL_REF="binom-dist"                # branch of louisaslett/anvl to build; it
                                     # must carry the harness's work queue
ANVL_REPO="https://github.com/louisaslett/anvl.git"

# Local scratch for the exported tarball, on this Mac.
DIST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/dist"

# ---- cluster --------------------------------------------------------------
REMOTE_HOST="cqlx43@hpc"
REMOTE_DIR="/home/cqlx43/anvl-sweeps"       # (where the .tar and .sif live)
SIF="${REMOTE_DIR}/${IMAGE_NAME}-${IMAGE_TAG}.sif"

# Bind-mounted working space, on the *cluster* filesystem. Everything the run
# writes goes here -- nothing is ever written inside the container. Put this on
# scratch or project space, not in a quota'd home directory: the full grid's
# store is tens of GB before export.
#
# It appears inside the container as /sweeps.
SWEEP_ROOT="/nobackup/cqlx43/anvl-sweeps"

SINGULARITY="singularity"            # or "apptainer"

# ---- Slurm ----------------------------------------------------------------
PARTITION="shared"                   # (queue name)
ACCOUNT=""                           # CHANGE ME or leave empty to omit
QOS=""                               # optional; leave empty to omit

# The sweep is a pool of WORKERS long-lived workers taking parts of cells from
# a queue until none are left (see README, "How the work is divided"). Each
# worker stops claiming parts it cannot finish inside WALLTIME, so a shorter
# walltime costs nothing but more, shorter jobs -- pick what your partition
# schedules quickly.
WORKERS=256                          # array tasks, one worker each
WORKER_LIMIT=""                      # optional: at most this many at once (%K)
CPUS_PER_TASK=1
MEM_PER_TASK="4G"                    # a worker compiles many cells in its life
WALLTIME="48:00:00"                  # per worker
MERGE_WALLTIME="04:00:00"

# How cells are cut into parts. A cell is cut into parts of about UNIT_MINUTES
# of compute, from what it cost before (COSTS); a cell never measured is cut
# into MAX_PARTS. A claim whose heartbeat is older than STALE_MINUTES belongs to
# a dead worker and is taken over: keep it above the slowest single chunk.
UNIT_MINUTES=15
MAX_PARTS=256
STALE_MINUTES=30
# What cells cost before, to size their parts: paths INSIDE the container,
# comma-separated, IN ORDER OF PRIORITY -- a later source overrides earlier
# ones for the cells it measured. The container sees SWEEP_ROOT as /sweeps,
# so $SWEEP_ROOT/parts/calib-<time> on the cluster is written here as
# /sweeps/parts/calib-<time>; ./calibrate.sh report prints the line to paste.
# Costs only decide how finely cells are cut; they never change a result. The
# usual list:
#   1. optionally, the last release ZIP, copied into $SWEEP_ROOT/costs/
#   2. the store ./calibrate.sh prints: a smoke sweep of the whole grid, or of
#      just the functions that are new or changed since that release
# Empty = no costs: every cell is cut into MAX_PARTS, which works but is
# coarse. See README, "Sizing the pool".
COSTS="/sweeps/parts/calib-20261008T175501"

# The validation array (slurm-validate.sbatch): after the merge, each task
# checks a share of the references against 256-bit MPFR. Units are distinct
# reference identities, so an anvl cell and its JAX twin cost one. ~250 units
# for the whole grid; at smoke depth they averaged ~4 s each, up to a few
# minutes for the log-scale qnorm ones. Not yet timed at full depth -- the
# sample counts barely grow with depth, so expect similar. Each task reads the
# merged store's tables whole, hence the memory.
VALIDATE_SHARDS=32
VALIDATE_WALLTIME="04:00:00"
VALIDATE_MEM="8G"

# ---- what to sweep --------------------------------------------------------
DEPTH="full"                         # smoke | quick | full
BACKENDS="anvl,jax"                  # "anvl" alone halves the grid
FILTER=""                            # e.g. "spec=nv_qnorm"; empty = everything

# ---- export ---------------------------------------------------------------
# Becomes the artifact id and the zip name in anvl-bench's release layout.
# Must match the platform_key the run recorded: <os>-<arch>-<device>.
ARTIFACT_ID="linux-x86_64-cpu"

# ---- guard ----------------------------------------------------------------
# Called before anything is submitted. Without it a wrong IMAGE_TAG queues the
# whole array against a path that does not exist and every task fails at
# launch, minutes later and once per worker.
require_sif() {
  if [[ ! -f "${SIF}" ]]; then
    echo "no image at ${SIF}" >&2
    echo "IMAGE_TAG is '${IMAGE_TAG}'. Images present in ${REMOTE_DIR}:" >&2
    ls -1 "${REMOTE_DIR}"/*.sif 2>/dev/null >&2 || echo "  (none)" >&2
    echo "Set IMAGE_TAG in config.sh to match, or build one with build.sh." >&2
    exit 1
  fi
}

