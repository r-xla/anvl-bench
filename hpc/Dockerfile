# ---------------------------------------------------------------------------
# anvl distribution-API sweep harness, for an HPC cluster via Singularity.
#
# Builds the whole r-xla stack from GitHub -- the same order as _setup.sh in
# the sibling-checkout root -- plus the sweep harness that lives in the
# benchmarks/ directory of the anvl fork, and a JAX virtualenv for the
# `--backends anvl,jax` comparison.
#
# The checkouts are laid out as siblings under /opt/r-xla, because the harness
# records each one's git SHA in every run's provenance by looking at
# <anvl>/../<pkg>. Flatten that layout and the provenance columns go NA.
#
#   docker buildx build --platform linux/amd64 -t anvl-sweeps:latest .
#
# See build.sh for the full build -> save -> ship -> .sif workflow.
# ---------------------------------------------------------------------------
ARG R_VERSION=4.6.1
FROM rocker/r-ver:${R_VERSION}

# Which sources to build. Override any of these with --build-arg.
ARG ANVL_REPO=https://github.com/louisaslett/anvl.git
ARG ANVL_REF=sweep-benchmarks
ARG PJRT_REF=main
ARG STABLEHLO_REF=main
ARG TENGEN_REF=main
ARG XLAMISC_REF=main

ENV R_XLA_ROOT=/opt/r-xla
ENV DEBIAN_FRONTEND=noninteractive

# pjrt compiles against protobuf and needs C++20; the rest is what R packages
# in this stack want at build time.
RUN apt-get update && apt-get install -y --no-install-recommends \
      git ca-certificates \
      libprotobuf-dev protobuf-compiler \
      libcurl4-openssl-dev libssl-dev libxml2-dev zlib1g-dev \
      python3 python3-venv python3-dev \
      unzip zip \
      libgmp-dev libmpfr-dev \
  && rm -rf /var/lib/apt/lists/*

# Under Singularity the container runs as *your* uid while /opt/r-xla is owned
# by root, which makes git refuse to report a HEAD ("dubious ownership") and
# silently empties the provenance SHAs. This is the fix.
RUN git config --system --add safe.directory '*'

# ---- sources --------------------------------------------------------------
WORKDIR ${R_XLA_ROOT}
RUN git clone --depth 1 -b ${TENGEN_REF}    https://github.com/r-xla/tengen.git    tengen    \
 && git clone --depth 1 -b ${PJRT_REF}      https://github.com/r-xla/pjrt.git      pjrt      \
 && git clone --depth 1 -b ${XLAMISC_REF}   https://github.com/r-xla/xlamisc.git   xlamisc   \
 && git clone --depth 1 -b ${STABLEHLO_REF} https://github.com/r-xla/stablehlo.git stablehlo \
 && git clone --depth 1 -b ${ANVL_REF}      ${ANVL_REPO}                           anvl

# ---- R dependencies -------------------------------------------------------
# Read straight out of the five DESCRIPTIONs rather than being listed here, so
# a new Imports: upstream does not need an edit to this file. The ecosystem
# packages themselves are excluded -- they are installed from the clones below,
# never resolved from GitHub, which is the whole point of not using
# pak/devtools here (they would follow anvl's Remotes: and pull main over the
# pinned clones).
RUN Rscript -e 'own <- c("anvl","pjrt","stablehlo","tengen","xlamisc"); \
      dirs <- file.path(Sys.getenv("R_XLA_ROOT"), own); \
      d <- unlist(lapply(dirs, function(p) { f <- file.path(p, "DESCRIPTION"); \
            dcf <- read.dcf(f); flds <- intersect(c("Depends","Imports","LinkingTo"), colnames(dcf)); \
            unlist(strsplit(dcf[1L, flds], ",")) })); \
      d <- trimws(sub("\\(.*", "", d)); \
      d <- setdiff(unique(d[nzchar(d)]), c("R", own, rownames(installed.packages()))); \
      message("installing: ", paste(d, collapse = ", ")); \
      if (length(d)) install.packages(d); \
      miss <- d[!d %in% rownames(installed.packages())]; \
      if (length(miss)) stop("failed to install: ", paste(miss, collapse = ", "))'

# The harness itself: nanoparquet + cli always, reticulate for --backends jax,
# DBI + duckdb for query.R's sw_sql(), and Rmpfr for `validate-refs`, which
# checks the stable and gradient references against high precision (it needs
# the GMP and MPFR libraries installed above; nothing else in the harness
# uses it, so a sweep never loads it).
RUN install2.r --error --skipinstalled nanoparquet cli reticulate DBI duckdb Rmpfr

# ---- the stack, in dependency order (as _setup.sh) ------------------------
RUN R CMD INSTALL ${R_XLA_ROOT}/tengen \
 && R CMD INSTALL ${R_XLA_ROOT}/pjrt \
 && R CMD INSTALL ${R_XLA_ROOT}/xlamisc \
 && R CMD INSTALL ${R_XLA_ROOT}/stablehlo \
 && R CMD INSTALL ${R_XLA_ROOT}/anvl

# ---- runtime system libraries for the R packages --------------------------
# P3M's Linux "binaries" link against system libraries that the base image does
# not carry -- fs needs libuv, and R CMD INSTALL does not notice, so the first
# symptom is a dyn.load() failure at the point of first use (which was the PJRT
# plugin download, several layers later).
#
# This list is not guesswork: it is what
#   pak::pkg_sysreqs(<the packages installed above>)
# reports for ubuntu-24.04. Re-run that after adding a package here.
#
# It sits after the installs rather than beside the build toolchain because
# these are runtime dependencies of what was just installed, and the check
# below is what ties the two together. (It also means adding a library here
# does not invalidate the cached layer holding the stack's compile.)
RUN apt-get update && apt-get install -y --no-install-recommends \
      cmake make libuv1-dev libpng-dev python3 xz-utils \
  && rm -rf /var/lib/apt/lists/*

# Fail here, loudly, rather than at first use. LD_LIBRARY_PATH must include
# R's own lib or every package reports `libR.so => not found` -- R resolves
# that one from R_HOME at load time, so it is a false positive and would make
# this check useless.
RUN set -eu; \
    export LD_LIBRARY_PATH="$(R RHOME)/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"; \
    bad=0; \
    for so in $(find /usr/local/lib/R/site-library -name '*.so'); do \
      if ldd "$so" 2>/dev/null | grep -q "not found"; then \
        echo "unresolved shared library in $so:"; ldd "$so" | grep "not found"; bad=1; \
      fi; \
    done; \
    if [ "$bad" != 0 ]; then \
      echo "Add the missing system package(s) to the apt layer above; pak::pkg_sysreqs() names them." >&2; \
      exit 1; \
    fi; \
    echo "no unresolved shared libraries"

# ---- the PJRT CPU plugin, baked in ----------------------------------------
# Downloaded at build time, not first run: a compute node may have no outbound
# network, and 40 array tasks racing to populate one cache directory is a
# corruption waiting to happen. Copied to a fixed path under /opt and pinned
# with PJRT_PLUGIN_PATH_CPU, because the default cache location is derived from
# HOME and the container's HOME is yours, not the build's.
ENV PJRT_INSTALL=1
RUN Rscript -e 'pjrt::install_pjrt(cuda = FALSE); \
      p <- pjrt:::plugin_path("cpu"); \
      stopifnot(length(p) == 1L, file.exists(p)); \
      dir.create("/opt/r-xla/pjrt-plugins", recursive = TRUE, showWarnings = FALSE); \
      stopifnot(file.copy(p, "/opt/r-xla/pjrt-plugins/pjrt_plugin_cpu.so"))'
ENV PJRT_PLUGIN_PATH_CPU=/opt/r-xla/pjrt-plugins/pjrt_plugin_cpu.so

# ---- JAX, for --backends anvl,jax -----------------------------------------
# The harness falls back to ../../../py-benchmarks/.venv when RETICULATE_PYTHON
# is unset; that checkout is not in this image, so the pin is what wires it up.
# jax_enable_x64 is set by the harness itself (sweeps/_normal.R), not here.
RUN python3 -m venv /opt/py \
 && /opt/py/bin/pip install --no-cache-dir --upgrade pip \
 && /opt/py/bin/pip install --no-cache-dir "jax[cpu]>=0.10.2" "numpy>=2.0" "scipy>=1.14"
ENV RETICULATE_PYTHON=/opt/py/bin/python
ENV JAX_PLATFORMS=cpu

# ---- runtime --------------------------------------------------------------
# NV_SWEEP_STORE is deliberately NOT set here: it must point at bind-mounted
# space outside the container (see config.sh / SWEEP_ROOT). Left unset, the
# harness would write into the image's read-only R_user_dir cache and fail.
ENV NV_SWEEP_DEVICE=cpu

RUN printf '%s\n' \
      '#!/bin/sh' \
      '# Thin wrapper so job scripts do not carry the path to run.R.' \
      'exec Rscript /opt/r-xla/anvl/benchmarks/api-distributions/run.R "$@"' \
      > /usr/local/bin/anvl-sweep \
 && chmod +x /usr/local/bin/anvl-sweep

WORKDIR /opt/r-xla/anvl/benchmarks/api-distributions

# A build that cannot run the engine is not worth shipping. ~30 s, and with
# Rmpfr present it includes the validator's own checks. The store is
# a throwaway: a real one belongs on bind-mounted space, and baking result files
# into the image would make them look like a run that happened on the cluster.
RUN NV_SWEEP_STORE=/tmp/selftest-store anvl-sweep selftest \
 && rm -rf /tmp/selftest-store

CMD ["anvl-sweep", "status"]
