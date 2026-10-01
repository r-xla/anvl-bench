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
# for one command:  IMAGE_TAG=20260927 ./build.sh all
IMAGE_TAG="${IMAGE_TAG:-20260927}"
DOCKER_PLATFORM="linux/amd64"        # the cluster's arch, not the Mac's
ANVL_REF="sweep-benchmarks"          # branch of louisaslett/anvl to build
ANVL_REPO="https://github.com/louisaslett/anvl.git"

# Local scratch for the exported tarball, on this Mac.
DIST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/dist"

# ---- cluster --------------------------------------------------------------
REMOTE_HOST="user@host"                   # CHANGE ME
REMOTE_DIR="/home/user/anvl-sweeps"       # CHANGE ME (where the .tar and .sif live)
SIF="${REMOTE_DIR}/${IMAGE_NAME}-${IMAGE_TAG}.sif"

# Bind-mounted working space, on the *cluster* filesystem. Everything the run
# writes goes here -- nothing is ever written inside the container. Put this on
# scratch or project space, not in a quota'd home directory: the full grid's
# store is tens of GB before export.
#
# It appears inside the container as /sweeps.
SWEEP_ROOT="/nobackup/user/anvl-sweeps"  # CHANGE ME 

SINGULARITY="singularity"            # or "apptainer"

# ---- Slurm ----------------------------------------------------------------
PARTITION="shared"                   # (queue name)
ACCOUNT=""                           # CHANGE ME or leave empty to omit
QOS=""                               # optional; leave empty to omit

SHARDS=320                            # array tasks; see README on sizing
CPUS_PER_TASK=1
MEM_PER_TASK="2G"
WALLTIME="72:00:00"                  # per array task
MERGE_WALLTIME="04:00:00"

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
JOBS=1                               # forked workers *within* one array task;
                                     # keep at 1 unless CPUS_PER_TASK > 1 and
                                     # you have measured that it helps

# ---- export ---------------------------------------------------------------
# Becomes the artifact id and the zip name in anvl-bench's release layout.
# Must match the platform_key the run recorded: <os>-<arch>-<device>.
ARTIFACT_ID="linux-x86_64-cpu"

# ---- guard ----------------------------------------------------------------
# Called before anything is submitted. Without it a wrong IMAGE_TAG queues the
# whole array against a path that does not exist and every task fails at
# launch, minutes later and 320 times over.
require_sif() {
  if [[ ! -f "${SIF}" ]]; then
    echo "no image at ${SIF}" >&2
    echo "IMAGE_TAG is '${IMAGE_TAG}'. Images present in ${REMOTE_DIR}:" >&2
    ls -1 "${REMOTE_DIR}"/*.sif 2>/dev/null >&2 || echo "  (none)" >&2
    echo "Set IMAGE_TAG in config.sh to match, or build one with build.sh." >&2
    exit 1
  fi
}
