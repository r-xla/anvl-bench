# `hpc/` — running the full sweep on a cluster under Singularity

This directory builds the CPU image, runs the sweep through Slurm, and packages
results for publication. Run its scripts from `hpc/`. `config.sh` selects the
source repository/ref, image, cluster paths and scheduler resources.

Documentation ownership:

- [Harness execution guide](https://github.com/r-xla/anvl/blob/main/benchmarks/api-distributions/HPC.md):
  the work queue, run identity, merging and completeness requirements.
- [Harness README](https://github.com/r-xla/anvl/blob/main/benchmarks/api-distributions/README.md):
  scoring, validation, result selection and exported tables.
- [Site publication guide](../README.md#how-the-results-get-here): release assets
  and website deployment.

Consult the `config.sh` variables `ANVL_REPO` / `ANVL_REF` for the code built into your image.

**Before a production submission:** Slurm success alone does not establish
successful sweep coverage. `check.sh` exposes what is missing; the scripts do
not enforce completeness.

```
config.sh            the only file you must edit
Dockerfile           the whole r-xla stack + the harness + JAX, from GitHub
build.sh             build (amd64) -> save -> scp -> .sif
submit.sh            plan a queue, then workers -> merge -> validation, each
                     waiting on the last; --resume adds workers to a queue
calibrate.sh         measure new cells cheaply (smoke depth), and the CPU
                     efficiency of a worker
check.sh             logs, queue, coverage and validation status, before export
slurm-work.sbatch      1. the worker pool: each task takes parts of cells from
                          the queue until none are left
slurm-merge.sbatch     2. assemble what is left, fold into the analysis store
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
  --home "$SWEEP_ROOT/home" --env TMPDIR=/sweeps/tmp \
  "$SIF" anvl-sweep selftest
# Keep this separate from the analysis store; remove it when no longer needed.

# 3. still on the cluster
./calibrate.sh submit     # every cell at smoke depth: costs, and CPU use
./calibrate.sh report <jobid>  # prints the store to put in COSTS
./submit.sh --dry-run     # the plan: cells, parts, where each cost came from
./submit.sh               # plan -> workers -> merge -> validation array

# 4. check logs, queue, coverage and validation status, then export
./check.sh                # exits 0 when nothing needs attention
./submit.sh --resume <queue>   # only if check.sh says parts are left
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
  queues/<queue>/         one per ./submit.sh (named for when it was planned):
                          the plan, the claims, every finished part's state,
                          and `arrays`, the worker pools submitted for it
  parts/<queue>/          that queue's staging store; workers write separate
                          files by table, run and cell
  store/                  the analysis store, merged into after the workers;
                          the validation array writes its records here too
  costs/                  earlier release ZIPs, for COSTS (your copies)
  export/                 the Parquet tables + manifest.json
  dist/<id>.zip           the release asset
  logs/                   Slurm stdout/stderr
  home/  tmp/             HOME and TMPDIR for the container
```

A finished part's state is kilobytes to a few hundred, so a queue of tens of
thousands of parts is a few GB at most. Delete `queues/<queue>` once its cells
are merged and checked; `parts/<queue>` once they are exported.

Put it on scratch or project space. `HOME` and `TMPDIR` are redirected there on
purpose: R, reticulate and XLA all write caches under `HOME`, and a quota'd home
directory is the classic way for task 37 of 40 to die at hour six. The scripts
set `HOME` with `--home "$SWEEP_ROOT/home"`, which mounts that directory at its
own path inside the container. `--env HOME=...` does not work: Singularity
ignores it, prints `Overriding HOME environment variable with SINGULARITYENV_HOME
is not permitted` to stderr, and leaves `HOME` as your real home directory.

Only the **queue and staging directory** are per submission. All submissions
merge into the same `store/`, so versions, depths and runs accumulate there. Set
`SWEEP_ROOT` to a fresh campaign directory for a publication at a new software
version or platform. Preserve the image and configuration used for that campaign;
job scripts source `config.sh` when they run, so do not edit it while jobs are
pending or running.

The merge waits on the workers with `afterany`, so it runs however they ended:
it assembles and merges every cell that is complete and reports the rest.
Validation waits on the merge with `afterok`. The harness can record cell
errors and still exit successfully, so a campaign can advance through merge
and validation with incomplete coverage. See the checks below.

## The validation step

`slurm-validate.sbatch` runs after merge and writes reference-validation records
into the analysis store. Its partitioning and interpretation are defined in the
[harness guide](https://github.com/r-xla/anvl/blob/main/benchmarks/api-distributions/HPC.md).

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

## How the work is divided

Cells differ in cost by orders of magnitude. In the last full run one took 31
minutes and another 8 hours, and a binomial quantile cell takes about a
hundred times a normal one. A job per cell therefore ends when its slowest
cell does, and most of the allocation waits for it.

So the work is a queue instead (the [harness guide](https://github.com/r-xla/anvl/blob/main/benchmarks/api-distributions/HPC.md#the-work-queue)
has the details):

- `submit.sh` first **plans** the campaign, on the login node, in the
  container: every cell of `FILTER` is cut into *parts* -- contiguous stretches
  of the cell's 2048 chunks, both signs -- of about `UNIT_MINUTES` each, sized
  from what the cell cost before (`COSTS`, see [Sizing the pool](#sizing-the-pool)).
  A cell never measured is cut into `MAX_PARTS`. Parts are queued biggest
  first.
- `WORKERS` array tasks then each run one long-lived **worker**, which claims
  the next part, sweeps it, saves its state, and claims again until the queue
  is empty. A worker stays on one cell while it has parts left, so its
  functions are compiled once. The worker that finishes a cell's last part
  assembles the cell -- identical, table for table, to the cell swept whole --
  and writes it to the staging store.
- A worker knows its job's end time and takes only parts it expects to finish
  before it, judged from what the cell's finished parts cost. It leaves rather
  than being killed mid-part.
- Every worker writes under the plan's one run ID, so a gradient cell and its
  value twin are always in the same run, wherever each was swept.

Once a worker holds an allocation it keeps it until the queue is empty or its
walltime is spent. A cluster that fills up after you submit slows only the
workers still pending, not the ones already running.

## Sizing the pool

A plan needs to know roughly what each cell costs, to cut it into parts of
about `UNIT_MINUTES`. Costs decide only how finely cells are cut -- a cell
assembled from 3 parts is identical to one assembled from 64 -- so a missing
or stale cost wastes some time and never changes a result. `COSTS` lists where
they come from, **in order of priority**: a later source overrides earlier
ones for the cells it measured.

**No earlier campaign is needed.** Measure the whole grid at smoke depth --
1/8192 of a full sweep -- and use that:

```bash
./calibrate.sh submit             # FILTER at smoke depth, on 8 workers
./calibrate.sh report <jobid>     # prints e.g. /sweeps/parts/calib-20261008T161544
# config.sh:  COSTS="/sweeps/parts/calib-20261008T161544"
```

Scaled up, a smoke time overstates a cell's cost, since compiling is a larger
share of a short sweep; an overstated cell is merely cut into more parts.

**With an earlier release**, its ZIP measured the cells it swept at full
depth, which is better still -- for the code it was swept with. Copy it into
`$SWEEP_ROOT/costs/`, list it first, and calibrate only what is new or has
changed since, listed after it so it overrides the release's figures for
those cells:

```bash
mkdir -p "$SWEEP_ROOT/costs" && cp linux-x86_64-cpu.zip "$SWEEP_ROOT/costs/"
CALIBRATE_FILTER="spec=nv_dbinom|nv_pbinom|nv_qbinom" ./calibrate.sh submit
# config.sh:  COSTS="/sweeps/costs/linux-x86_64-cpu.zip,/sweeps/parts/calib-..."
```

Costs are matched by cell, not by code, so the plan cannot tell that a
function changed since its cost was measured: re-calibrating it is up to you.
A stale cost that understates a cell gives it long parts, and a worker may be
killed at its time limit mid-part; the part is swept again by the next worker
or `--resume`, so only time is lost. A cell no source measured is cut into
`MAX_PARTS` and queued first.

Check the plan before submitting it:

```bash
./submit.sh --dry-run
```

It reports how many cells each source supplied, the parts, the estimated
core-hours of the measured cells, and every cell nothing measured.

Then choose:

- `WORKERS`: the pool. The campaign takes roughly *total core-hours / WORKERS*
  once the pool is running, but never less than the largest part. More workers
  than parts is waste.
- `WALLTIME`: per worker. Workers stop claiming parts they cannot finish, so a
  short walltime wastes little; pick what your partition schedules quickly.
  Parts that did not fit are left for `./submit.sh --resume`.
- `UNIT_MINUTES`: the part size for measured cells. Smaller parts balance the
  end of the campaign better and cost a few seconds each in overhead.
- `MAX_PARTS`: how finely an unmeasured cell is cut. At full depth a cell has
  2048 chunks, so up to 2048.
- `MEM_PER_TASK`: a worker compiles many cells over its life. Measure peak
  memory with `./calibrate.sh report`.

## Cores per task

The scripts request CPU affinity with `srun --cpu-bind=cores` and log
`Cpus_allowed_list` and `nproc` from inside the container. Check these logs on
your cluster; allocation and binding depend on the site's configuration.
`OMP_NUM_THREADS=1` and `OPENBLAS_NUM_THREADS=1` constrain libraries that honour
those settings, but should not be treated as a general limit on XLA threads.

Earlier measurements of an unbound `nv_qnorm` process reported about 3.5 cores
and 29 threads, and about 2.3 cores with a legacy Eigen flag disabled. Their
runtime version was not recorded here. Use `./calibrate.sh submit <depth>
<cpus>` and `./calibrate.sh report` on the current image and target partition
to choose `CPUS_PER_TASK`; retain image identity, affinity, runtime and peak
memory alongside the measurements.

## Checking completion and recovering work

`./check.sh` runs checks 1–4 below, and prints
one summary that ends in either "nothing failed" or a count of problems:

```bash
./check.sh                          # newest queue and validation array
./check.sh <queue> [<val id>]       # specific ones
CHECK_VERBOSE=1 ./check.sh          # also print the full queue and status
```

It takes the queue's worker arrays from `queues/<queue>/arrays` and the
validation array from the log file names in `$SWEEP_ROOT/logs`. It ignores
calibration queues (`calib-*`), and after a `--validate-only` retry it uses the
newer validation array. It lists every worker that died, every part that
errored, every cell not yet assembled, every reference that failed, and the
store's coverage row for `ARTIFACT_ID` at `DEPTH`. It exits 1 if it finds any
problem. Troubles of an earlier worker pool are only warnings: a `--resume`
is what they are for, and the queue section says whether it worked. Use the
check to decide whether to export: `export.sh` does not run it or depend on
it. It does not read the export manifest (check 5), which only exists after
`export.sh`.

The checks, which you can also run by hand:

1. Inspect scheduler state and worker logs. Every worker should end in
   `worker done:`, with the reason it stopped; inspect `ERROR` lines. A zero
   exit code alone is insufficient.
2. Inspect the queue with `anvl-sweep queue --queue /sweeps/queues/<queue>`
   under the same container bindings: how many cells are assembled, how many
   parts are done, running, stale, errored or not started, and the core-hours
   spent and expected.
3. Inspect the merged store with `anvl-sweep status --backends "$BACKENDS"`
   under the same container bindings and `NV_SWEEP_STORE=/sweeps/store` used by
   `slurm-merge.sbatch`. It reports, per platform and depth, how many declared
   cells succeeded on their newest attempt, lists every cell whose newest
   attempt errored, and tallies reference statuses. In a reused store, a cell
   this submission never reached still counts as swept from an earlier run at
   the same depth; a fresh `SWEEP_ROOT` per campaign avoids that.
4. Inspect validation logs and the selected references' statuses. Distinguish
   failed comparisons, missing records and previous validation records.
5. Before publishing, inspect the export manifest's platforms, versions, SHAs,
   depths and coverage counts (`n_cells_declared`, `n_cells_errored`,
   `n_cells_not_run`). The export warns when any cell errored or was never run,
   and the site shows those cells as such; `export.sh` also warns about a
   platform-name mismatch. None of these warnings rejects the export or
   certifies a coherent snapshot.

**Recovering.** If parts are left -- workers hit their walltime, were
pre-empted or crashed -- add workers to the same queue:

```bash
./submit.sh --resume <queue>
```

Every finished part is kept, so nothing is swept twice. If no worker of the
queue is still queued or running, the claims of unfinished parts are released
first, so the new workers start on them at once. Otherwise the new workers
join the old ones, and a dead worker's claim is taken over once its heartbeat
is `STALE_MINUTES` old. The merge and validation steps run again afterwards.

A part that errored makes its cell an errored cell, as an ordinary sweep
would. Errors are usually deterministic, so a retry needs a fixed image and a
new queue. To retry a transient error, delete that part's files from
`queues/<queue>/done/` and its cell's directory from `queues/<queue>/final/`,
then resume. Do not edit `config.sh`'s `DEPTH`, `BACKENDS` or `FILTER` to
resume: the queue's plan fixes what is swept, and `--resume` ignores them.

Export recreates `$SWEEP_ROOT/export` and replaces the ZIP named by
`ARTIFACT_ID`; archive a previous deliverable first if it needs to be retained.

## Monitoring the job

```
squeue --me
sacct -j <jobid> -X --format=JobID,JobName,State,Elapsed,TotalCPU
./check.sh <queue>        # or just the queue's progress:
"${SINGULARITY}" exec --cleanenv --bind "$SWEEP_ROOT:/sweeps" --home "$SWEEP_ROOT/home" \
  "$SIF" anvl-sweep queue --queue /sweeps/queues/<queue>
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
