# `hpc/` — running the full sweep on a cluster under Singularity

This directory builds the CPU image, runs the sweep through Slurm, and packages
results for publication. Run its scripts from `hpc/`. `config.sh` selects the
source repository/ref, image, cluster paths and scheduler resources.

Documentation ownership:

- [Harness execution guide](../../anvl-sweeps/benchmarks/api-distributions/HPC.md):
  shard assignment, run identity, merging and completeness requirements.
- [Harness README](../../anvl-sweeps/benchmarks/api-distributions/README.md):
  scoring, validation, result selection and exported tables.
- [Site publication guide](../README.md#how-the-results-get-here): release assets
  and website deployment.

Consult the `config.sh` variables `ANVL_REPO` / `ANVL_REF` for the code built into your image.

**Before a production submission:** read the harness's run-identity limitation.
The supplied cell-sharded workflow can separate value/gradient evidence, and
Slurm success alone does not establish successful sweep coverage. The checks
below expose these limitations; the scripts do not yet enforce completeness.

```
config.sh            the only file you must edit
Dockerfile           the whole r-xla stack + the harness + JAX, from GitHub
build.sh             build (amd64) -> save -> scp -> .sif
submit.sh            submit sweep -> merge -> validation, each waiting on the last
calibrate.sh         one shard at quick depth, then its CPU efficiency
slurm-sweep.sbatch     1. one array task per shard of the grid
slurm-merge.sbatch     2. fold the shards into the analysis store
slurm-validate.sbatch  3. one array task per shard of the references, checked
                          against MPFR and recorded in the store
export.sh            merged store -> anvl-bench release ZIP
```

## The workflow

```bash
# 1. on the Mac — edit config.sh first, including a new IMAGE_TAG
./build.sh all            # or: build / save / ship / sif, one at a time
                          # (ship and sif open one ssh connection in the
                          #  foreground -- MFA shows here -- and share it;
                          #  ./build.sh disconnect closes it)

# 2. on the cluster, in REMOTE_DIR: check the image before trusting it
source config.sh && mkdir -p "$SWEEP_ROOT/home" "$SWEEP_ROOT/tmp"
"${SINGULARITY}" exec --cleanenv --bind "$SWEEP_ROOT:/sweeps" \
  --env NV_SWEEP_STORE=/sweeps/selftest-store \
  --env HOME=/sweeps/home --env TMPDIR=/sweeps/tmp \
  "$SIF" anvl-sweep selftest
# Keep this separate from the analysis store; remove it when no longer needed.

# 3. still on the cluster
./calibrate.sh submit     # one shard at quick depth, to size CPUS_PER_TASK
./calibrate.sh report <jobid>
./submit.sh --dry-run     # what shard 1 would take; runs nothing
./submit.sh               # sweep array -> merge -> validation array

# 4. inspect logs, coverage and validation status (see below), then export
./export.sh               # -> $SWEEP_ROOT/dist/linux-x86_64-cpu.zip
```

The self-test in step 2 is the same one the image build runs, but on the
cluster's CPUs and filesystem, which is what the sweep will use: its rounding
and `-0` checks exist to catch a platform difference before a many-hour run
rather than after. Expect `N/N assertions passed` and exit status 0, and **no**
"skip the comparator checks" line — that line would mean Rmpfr is missing from
the image, and step 3's validation would fail. It points at a throwaway store,
so its runs never mix into the real one. If your site discourages even half a
minute of work on the login node, prefix the `singularity` line with
`srun --partition="$PARTITION" --time=00:10:00 --mem=2G`.

Attach the ZIP to a release and deploy it using the
[site publication guide](../README.md#how-the-results-get-here). Release upload
and deployment instructions are maintained there.

## Where things are written

The scripts direct results and user cache/temp paths to the bind mount. `SWEEP_ROOT` in `config.sh` is a
path on the cluster filesystem, bind-mounted at `/sweeps`, and it is the one
path you must get right:

```
$SWEEP_ROOT/
  parts/<array job id>/   one shared staging store per array; tasks write
                          separate files by table, run and cell
  store/                  the analysis store, merged into after the array; the
                          validation array writes its records here too
  export/                 the Parquet tables + manifest.json
  dist/<id>.zip           the release asset
  logs/                   Slurm stdout/stderr
  home/  tmp/             HOME and TMPDIR for the container
```

Put it on scratch or project space. `HOME` and `TMPDIR` are redirected there on
purpose: R, reticulate and XLA all write caches under `HOME`, and a quota'd home
directory is the classic way for task 37 of 40 to die at hour six.

Only the **staging directory** is keyed by array job ID. All submissions merge
into the same `store/`, so versions, depths and runs accumulate there. Set
`SWEEP_ROOT` to a fresh campaign directory for a publication at a new software
version or platform. Preserve the image and configuration used for that campaign;
job scripts source `config.sh` when they run, so do not edit it while jobs are
pending or running.

`afterok` waits for successful process exits. The harness can record cell errors
and still exit successfully, so it can advance through merge and validation
with incomplete successful coverage. See the checks below.

## The validation step

`slurm-validate.sbatch` runs after merge and writes reference-validation records
into the analysis store. Its partitioning and interpretation are defined in the
[harness guide](../../anvl-sweeps/benchmarks/api-distributions/HPC.md).

Use `./submit.sh --validate-only` to retry validation against the current store,
for example after an interrupted task. Completed failed comparisons are recorded
as failures; interrupted tasks may write no new records, and previous records
remain in force. Changing a reference requires rebuilding the image and rerunning
the affected sweeps so that their stored reference identities match. Changes to
MPFR truths or validation methods follow the harness's identity rules.

`export.sh` warns when no validation records were exported. Presence of a
`validations.parquet` file does not establish that all selected references passed.

## Choices the image makes, and why

- **The clones are siblings under `/opt/r-xla`.** The harness records each
  ecosystem package's git SHA per run by looking at `<anvl>/../<pkg>`. A
  flatter layout builds fine and silently produces NA provenance.
- **`R CMD INSTALL` per package, never `pak`/`devtools`.** Those follow anvl's
  `Remotes:` and pull the siblings from GitHub main over the pinned clones,
  which can produce an incompatible stack. CRAN dependencies
  are read out of the five `DESCRIPTION`s at build time rather than listed in
  the Dockerfile, so a new `Imports:` upstream needs no edit here.
- **Git ownership configuration.** The image configures `safe.directory` for
  explicit Git commands under the runtime user. Harness provenance reads `.git`
  files directly and does not depend on this setting.

- **The PJRT CPU plugin is downloaded at build time** and pinned with
  `PJRT_PLUGIN_PATH_CPU`, not left to first use: a compute node may have no
  outbound network, the default cache path is derived from `HOME` (which is
  yours, not the build's), and 40 tasks racing to populate one cache directory
  is a corruption waiting to happen.
- **System libraries are listed in the Dockerfile and checked.** The list was
  informed by `pak::pkg_sysreqs()`; it is not regenerated automatically. P3M's
  Linux binaries link libraries the base image lacks (`fs` needs libuv), and
  `R CMD INSTALL` does not notice — the first symptom is a `dyn.load()` failure
  at first use, layers later. The build scans every installed `.so` with `ldd`
  and fails on any unresolved soname, with `LD_LIBRARY_PATH` set to R's own lib
  so that `libR.so` does not report as missing for every package.
- **JAX is a venv at `/opt/py`, pinned with `RETICULATE_PYTHON`.** Without the
  pin the harness looks for `../../../py-benchmarks/.venv`, which is not in the
  image. `jax_enable_x64` is set by the harness itself, not here.
- **`--cleanenv` on every `singularity exec`.** It strips the *host's*
  environment, not the image's, so the pins above survive and a stray
  `R_LIBS_USER` or `PJRT_*` on the login node cannot leak into a run.
- **Rmpfr, with the GMP and MPFR libraries,** for the validation step. Nothing
  in an ordinary sweep needs it; selftest also uses it. It is included so a
  compute node can run validation without installing dependencies.
- **The build runs `anvl-sweep selftest`.** A sweep that silently sweeps
  nothing looks exactly like a sweep that found nothing, and at full depth that
  is an expensive way to find out. With Rmpfr present it includes the
  validator's own checks.

## Cores per task

The scripts request CPU affinity with `srun --cpu-bind=cores` and log
`Cpus_allowed_list` and `nproc` from inside the container. Check these logs on
your cluster; allocation and binding depend on the site's configuration.
`OMP_NUM_THREADS=1` and `OPENBLAS_NUM_THREADS=1` constrain libraries that honour
those settings, but should not be treated as a general limit on XLA threads.

Earlier measurements of an unbound `nv_qnorm` process reported about 3.5 cores
and 29 threads, and about 2.3 cores with a legacy Eigen flag disabled. Their
runtime version was not recorded here. Use `calibrate.sh` on the current image
and target partition to choose `CPUS_PER_TASK`; retain image identity, affinity,
runtime and peak memory alongside the measurements.

## Sizing the array

Query the grid **inside the image** before choosing `SHARDS`:

```bash
source config.sh
"${SINGULARITY}" exec --cleanenv "$SIF" anvl-sweep list --backends "$BACKENDS"
```

Include the configured filter when estimating a filtered campaign. The local
harness on 2026-09-29 has 256 combined cells (160 anvl, 96 JAX); the current
`SHARDS=320` would leave 64 tasks empty for that grid. The image may contain a
different revision. Anvl/JAX twins are separate cells and are not guaranteed to
share a task or node.

Use the [harness assignment rule](../../anvl-sweeps/benchmarks/api-distributions/HPC.md#assigning-work)
to inspect the actual shard contents. Increasing shards trades shorter jobs
against startup and compilation overhead. Calibrate representative functions,
precisions and gradient cells, rather than scaling one unusually cheap cell.
`calibrate.sh` samples one shard, not the whole workload.

Measure peak memory as well as elapsed time. Chunked enumeration does not bound
compilation, retained regions, parallel workers or analysis-table memory.
Merge, validation and export can have different resource needs from sweeps.
Run export in an allocation if its measured cost or site policy requires it.

## Checking completion and recovering work

1. Inspect scheduler state and sweep logs. Every intended cell should report
   `OK`; inspect `ERROR` entries and the final per-task counts. A zero exit code
   alone is insufficient.
2. Inspect the merged store with `anvl-sweep status --backends "$BACKENDS"`
   under the same container bindings and `NV_SWEEP_STORE=/sweeps/store` used by
   `slurm-merge.sbatch`. Compare successful cells at the intended depth with the
   selected grid. In a reused store, restrict the audit to this submission's
   run IDs so older results do not conceal missing work.
3. Inspect validation logs and the selected references' statuses. Distinguish
   failed comparisons, missing records and previous validation records.
4. Before publishing, inspect the export manifest's platforms, versions, SHAs,
   depths and coverage. `export.sh` warns about a platform-name mismatch but
   does not reject it or certify a coherent snapshot.

For an interrupted sweep, retain its staging data and identify missing or
failed work before resubmission. The current scripts do not implement automatic
resume or a completeness gate. Changing `FILTER` changes row-based shard
assignment; preserve the original configuration when retrying shard numbers.
If making a custom grouped invocation to preserve value/gradient evidence,
keep both kinds in the same invocation as described in the harness guide.

Export recreates `$SWEEP_ROOT/export` and replaces the ZIP named by
`ARTIFACT_ID`; archive a previous deliverable first if it needs to be retained.

## Monitoring the job

```
squeue --me
sacct -j <jobid> -X --format=JobID,JobName,State,Elapsed,TotalCPU
```

After the run, keep a record of runtime:

```
sacct -j <jobid> -X --format=JobID,JobName,State,Elapsed,TotalCPU > ~/anvl-sweeps/job-<jobid>-timing.txt
```

## If you later want the CUDA column

The supplied image and scripts select CPU execution, including `JAX_PLATFORMS=cpu`
and the CPU device label. A GPU workflow needs a GPU-capable PJRT plugin and JAX
installation, explicit runtime device selection, compatible host drivers, GPU
scheduler resources and container GPU passthrough. Verify the actual device for
both backends on a compute node before sweeping. Set `NV_SWEEP_DEVICE=cuda` only
after configuring CUDA execution; that variable labels provenance and does not
select a device. Keep GPU results in a separate campaign store.
