#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Build the sweep image for the cluster's architecture, and get it there.
#
#   ./build.sh build     docker buildx, linux/amd64
#   ./build.sh save      docker save -> dist/<image>-<tag>.tar.gz
#   ./build.sh ship      scp the tarball to the cluster
#   ./build.sh sif       build the .sif over ssh on the cluster
#   ./build.sh all       all four, in order
#   ./build.sh connect   just open (or check) the connection the remote steps use
#   ./build.sh disconnect  close it
#
# The remote steps share one ssh connection of their own, opened in the
# foreground -- with your terminal attached, so an MFA prompt is shown -- the
# first time a step needs it, and reused by every later step for up to
# REMOTE_PERSIST, so each step can still be run on its own with at most one
# MFA. It is kept apart from any ControlMaster in your ~/.ssh/config: the sif
# step feeds its script on stdin, and a connection that has to authenticate
# with stdin taken cannot show you the MFA prompt, so it waits for ever.
#
# The three steps after `build` are stubs in the sense that the *host and paths*
# they act on are placeholders in config.sh -- the commands themselves are real
# and will work once REMOTE_HOST / REMOTE_DIR are yours.
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./config.sh

IMAGE="${IMAGE_NAME}:${IMAGE_TAG}"

# ---- the connection the remote steps share --------------------------------
SSH_SOCK="${HOME}/.ssh/anvl-build-%C"
REMOTE_PERSIST="${REMOTE_PERSIST:-2h}"
SSH_OPTS=(
  -o ControlPath="${SSH_SOCK}"
  -o ConnectTimeout=30
  -o ServerAliveInterval=30
  -o ServerAliveCountMax=4
)

remote_up() {
  if ssh "${SSH_OPTS[@]}" -O check "${REMOTE_HOST}" 2>/dev/null; then
    return 0
  fi
  echo "==> connecting to ${REMOTE_HOST} (authenticate / approve MFA if asked)"
  # -f: go to the background only *after* authenticating, so the prompt is
  # shown here; -N: no remote command, this connection only carries others.
  ssh "${SSH_OPTS[@]}" -o ControlMaster=yes -o ControlPersist="${REMOTE_PERSIST}" -fN "${REMOTE_HOST}"
  ssh "${SSH_OPTS[@]}" -O check "${REMOTE_HOST}"
}

# Every remote command goes through the shared connection, never a new one:
# if it has gone, fail now rather than wait on an authentication nobody sees.
rssh() { ssh "${SSH_OPTS[@]}" -o ControlMaster=no -o BatchMode=yes "${REMOTE_HOST}" "$@"; }
rscp() { scp "${SSH_OPTS[@]}" -o ControlMaster=no -o BatchMode=yes "$@"; }

echo "image: ${IMAGE}   sif: ${SIF}"
TARBALL="${DIST_DIR}/${IMAGE_NAME}-${IMAGE_TAG}.tar"

step_build() {
  # NOTE: on an Apple-silicon Mac this builds linux/amd64 under QEMU emulation.
  # It works, but pjrt's C++ compile is the slow part and emulation makes it
  # slower still -- budget half an hour or more, and do not be alarmed by it.
  # If that becomes tiresome, the alternatives are (a) push to a registry from
  # a native amd64 runner and pull on the cluster, or (b) build the .sif
  # directly on the cluster with `apptainer build --fakeroot ... docker://...`.
  echo "==> docker buildx build (${DOCKER_PLATFORM})"
  docker buildx build \
    --platform "${DOCKER_PLATFORM}" \
    --build-arg "ANVL_REPO=${ANVL_REPO}" \
    --build-arg "ANVL_REF=${ANVL_REF}" \
    --tag "${IMAGE}" \
    --load \
    .
  echo "==> built ${IMAGE}"
}

step_save() {
  echo "==> docker save -> ${TARBALL}.gz"
  mkdir -p "${DIST_DIR}"
  docker save "${IMAGE}" -o "${TARBALL}"
  gzip -f "${TARBALL}"
  ls -lh "${TARBALL}.gz"
}

step_ship() {
  remote_up
  echo "==> scp to ${REMOTE_HOST}:${REMOTE_DIR}"
  rssh "mkdir -p '${REMOTE_DIR}'"
  rscp "${TARBALL}.gz" "${REMOTE_HOST}:${REMOTE_DIR}/"
  # Also send the job scripts, so the cluster side is self-contained.
  rscp config.sh submit.sh calibrate.sh export.sh slurm-sweep.sbatch slurm-merge.sbatch slurm-validate.sbatch \
      "${REMOTE_HOST}:${REMOTE_DIR}/"
}

step_sif() {
  remote_up
  echo "==> building ${SIF} on ${REMOTE_HOST}"
  # Singularity needs no root for docker-archive://, so this runs fine on a
  # login node -- but it is disk- and metadata-hungry. Its scratch is the login
  # node's local /tmp, not NFS home: unpacking hundreds of thousands of small
  # files onto NFS is where a build spends its time. Some sites forbid heavy
  # work on the login node entirely; if yours does, wrap this in a small
  # interactive job instead.
  # The script below is an *unquoted* heredoc, so this shell expands it before
  # sending it: \$ defers a variable to the cluster, and it must contain no
  # backquotes or \$( ) that are meant for the cluster -- not even in a comment,
  # where a backquoted word is still run here. (A comment reading "piped into
  # <backquote>head<backquote>" once ran head locally, waiting on the terminal
  # for ever, before ssh was even started.) Comments belong out here.
  #
  # The list check writes to a file rather than piping into head: R dies on
  # SIGPIPE when head closes the pipe early, and the "ignoring SIGPIPE signal"
  # error looks like a broken image when it is only a closed pipe.
  rssh bash -s <<REMOTE
set -euo pipefail
say() { echo "    [\$(date +%H:%M:%S) \$(hostname -s)] \$*"; }
say "connected; working in ${REMOTE_DIR}"
cd '${REMOTE_DIR}'
export SINGULARITY_TMPDIR="/tmp/\${USER}-anvl-sif/tmp"
export SINGULARITY_CACHEDIR="/tmp/\${USER}-anvl-sif/cache"
mkdir -p "\$SINGULARITY_TMPDIR" "\$SINGULARITY_CACHEDIR"
trap 'rm -rf "/tmp/\${USER}-anvl-sif"' EXIT
say "unzipping ${IMAGE_NAME}-${IMAGE_TAG}.tar.gz"
gunzip -kf '${IMAGE_NAME}-${IMAGE_TAG}.tar.gz'
say "building the .sif (scratch in /tmp; a few minutes)"
${SINGULARITY} build --force '${SIF}' \
  'docker-archive://${REMOTE_DIR}/${IMAGE_NAME}-${IMAGE_TAG}.tar'
rm -f '${REMOTE_DIR}/${IMAGE_NAME}-${IMAGE_TAG}.tar'
say "built; checking it runs"
${SINGULARITY} exec '${SIF}' anvl-sweep list > '${REMOTE_DIR}/.list-check.txt'
head -n 5 '${REMOTE_DIR}/.list-check.txt'
echo '--- plugin, as pinned in the image ---'
${SINGULARITY} exec '${SIF}' sh -c 'ls -l "\$PJRT_PLUGIN_PATH_CPU"'
say "done"
REMOTE
}

case "${1:-all}" in
  build) step_build ;;
  save)  step_save ;;
  ship)  step_ship ;;
  sif)   step_sif ;;
  all)   step_build; step_save; step_ship; step_sif ;;
  connect) remote_up ;;
  disconnect) ssh "${SSH_OPTS[@]}" -O exit "${REMOTE_HOST}" ;;
  *) echo "usage: $0 {build|save|ship|sif|all|connect|disconnect}" >&2; exit 2 ;;
esac
