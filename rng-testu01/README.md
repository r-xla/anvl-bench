# `rng-testu01` — statistical test batteries for anvl's RNG

This harness runs anvl's random stream through the batteries that Salmon et
al. (2011), *Parallel Random Numbers: As Easy as 1, 2, 3* (SC '11), used to
validate ThreeFry and Philox: TestU01's **SmallCrush**, **Crush** and
**BigCrush**. It can also run [PractRand](https://pracrand.sourceforge.net/),
which reads the same bit stream.

anvl draws its bits with `prim_sample_bits()`: ThreeFry-2x32 with 20 rounds
(the generator and round count of the paper's `threefry2x32` and of JAX). The
global random state is a key k, the seed passed to `nv_set_seed()`, and a
counter c, which starts at 0. Every call of `prim_sample_bits()`, a *sampling
call*, splits the random state: it draws from the fresh key
k' = ThreeFry_k(c), hashing it with the counters 0, 1, 2, ..., and advances c
by one. Value t of a draw comes from the hash of the counter t: a `ui32` value
is the xor of the hash's two words, a `ui64` value its first word above its
second, as in JAX. See the "Random Numbers" vignette for the details.

So the stream a user sees is made of many keys, each used for the few values
of one sampling call. `gen.R` writes exactly that stream: the values of
consecutive sampling calls, `--per-call` values each.

## Files

| File       | What it does                                                                 |
|------------|------------------------------------------------------------------------------|
| `build.sh` | Downloads and builds TestU01 (and PractRand with `--practrand`) into `_deps/`, compiles the harness `testu01` |
| `testu01.c`| Reads raw 32-bit words (or doubles, `-d`) from stdin and runs one battery    |
| `gen.R`    | Writes anvl's stream of consecutive sampling calls to stdout, from one seed or several interleaved |
| `run.sh`   | Pipes `gen.R` into a battery and files the report under `results/`          |

`_deps/`, the binaries and `results/` are gitignored.

`build.sh` downloads the batteries from their authors and checks each archive
against a SHA-256 written in the script:

- TestU01 1.2.3 by Pierre L'Ecuyer and Richard Simard, from
  <https://github.com/umontreal-simul/TestU01-2009> at commit
  `57e98bf33880daedc930739c81e39b89a8d24dba`. Apache License 2.0.
- PractRand pre-0.95 by Chris Doty-Humphrey, from
  <https://downloads.sourceforge.net/project/pracrand/PractRand-pre0.95.zip>.
  Public domain, as its [SourceForge project page](https://sourceforge.net/projects/pracrand/)
  states; the archive itself contains no license file.

## Quick start

Run everything where R, anvl's dependencies and a pjrt build that the anvl
checkout works with are installed. `gen.R` loads anvl with
`pkgload::load_all()` from the `anvl` checkout next to this repository, or the
one `ANVL_PATH` names (set `ANVL_INSTALLED=1` to use the installed package).

```bash
cd rng-testu01
./build.sh                       # TestU01 + harness, ~20 s (needs network)
./build.sh --practrand           # also PractRand's RNG_test, ~1 min

./run.sh smallcrush              # seed 42, ~20 s
./run.sh crush --seed 7          # hours, see below
./run.sh bigcrush                # many hours
```

`run.sh` writes `results/<battery>-<options>-<time>.txt` (the battery's
report), `.tsv` (every statistic's p-value) and `.log` (generator throughput,
the number of words the battery consumed, wall time and exit status). The exit
status is 1 if a p-value is outside [1e-10, 1 - 1e-10], TestU01's threshold for
a clear failure.

The pieces also work on their own:

```bash
Rscript gen.R --seed 42 | ./testu01 -n "anvl seed 42" smallcrush
Rscript gen.R --seed 42 --mode unif | ./testu01 -d smallcrush
Rscript gen.R --seed 42 | ./RNG_test stdin32 -tlmax 1GB
```

To run Crush or BigCrush past the end of a session, detach it from the shell,
in the environment that has R and anvl:

```bash
nohup setsid ./run.sh bigcrush --seed 42 </dev/null >/dev/null 2>&1 &
```

and check on it with `cat results/bigcrush-*.log`. The `.txt`
report is written when the battery ends; with `TESTU01_OPTS=-v` it instead
fills in test by test with TestU01's detailed output, which shows how far the
battery has got.

## What to run

### Options

`gen.R` (passed through by `run.sh`):

| Option         | Default      | Meaning                                                                     |
|----------------|--------------|-----------------------------------------------------------------------------|
| `--seed S`     | 42           | Seed for `nv_set_seed()`                                                    |
| `--dtype D`    | `ui32`       | `ui32` or `ui64` draws                                                      |
| `--mode M`     | `bits`       | `bits`: raw words. `unif`: `nv_runif(dtype = "f64")` as doubles (`testu01 -d`) |
| `--per-call P` | 1            | Values drawn per sampling call                                              |
| `--streams K`  | 1            | Interleave the streams seeded S, S + 1, ..., S + K - 1                      |
| `--chunk C`    | 1            | Numbers taken from one stream before moving on to the next                  |
| `--calls M`    | 2^20 / K     | Sampling calls per jitted call (an `nv_scan()`), per stream                 |
| `--n N`        | unlimited    | Stop after N numbers                                                        |

`testu01` options, passed by `run.sh` through `TESTU01_OPTS`: `-r` reverses
the bits of every word, `-v` prints TestU01's detailed output.
`PRACTRAND_MAX` sets PractRand's `-tlmax` (default `1GB`).

### Variants worth running

- **Values per sampling call** (`--per-call 1`, `2`, `4`, ...). With one
  value per call, each key contributes only its first hash; with more,
  consecutive hashes of one key.
- **Several seeds.** One battery run is one sample: rerun with other `--seed`s.
- **Reversed bits** (`TESTU01_OPTS=-r`). Most TestU01 tests turn each word
  into a uniform in [0, 1) and look at its leading bits, so the low bits of a
  word are tested less. Reversing the bits tests them as strongly.
- **Interleaved streams** (`--streams K --chunk 1`). The stream that TestU01 sees
  takes one word from each of K generators seeded with consecutive integers in
  turn. Consecutive seeds are keys that differ in a few low bits.
- **Uniform doubles** (`--mode unif`). Tests `nv_runif()` rather than the bits
  under it. TestU01 reads the leading 32 bits of each double, so this mostly
  checks the conversion to `[0, 1)`.

Not covered yet: deep chains of forks (`nv_while()` loops nested in each
other), which need a way to draw from a random state other than the global one.

## The batteries

The word counts are those the harness reported for SmallCrush and BigCrush,
and that of the TestU01 user guide (L'Ecuyer & Simard, 2007) for Crush. Times
are for this harness on one core of the dev machine for SmallCrush, and rough
for Crush. The BigCrush time is that of a 4-core job on the `lrz-cpu` nodes of
the LRZ AI Systems (Xeon Gold 6148), where the generator sets the pace (see
below).

| Battery    | Tests / statistics | 32-bit words consumed   | Time            |
|------------|--------------------|-------------------------|-----------------|
| SmallCrush | 10 / 15            | 2.27e8 (~2^27.8, ~0.9 GB) | ~20 s         |
| Crush      | 96 / 144           | ~2^35 (~140 GB)         | ~1 h            |
| BigCrush   | 106 / 160          | 3.57e11 (~2^38.4, ~1.4 TB) | ~11 h        |
| PractRand  | grows by doubling  | `-tlmax` bytes          | ~12 s per GB    |

- **SmallCrush** is a quick screen: birthday spacings, collisions, gaps,
  poker, coupon collector, max-of-t, weight distribution, binary matrix rank,
  Hamming independence and random walks. Weak generators (e.g. many LCGs)
  already fail it.
- **Crush** adds many more parameterisations of these tests and others
  (serial tests on overlapping tuples, close pairs, Fourier, linear complexity,
  autocorrelation, ...). It separates generators with long-range or linear
  structure (Mersenne Twister fails its linear-complexity tests).
- **BigCrush** runs the tests of Crush on longer sequences and more
  parameterisations; it is the most stringent of the three.
- **PractRand** is a later battery (not in the paper) that keeps testing ever
  longer prefixes of the stream (256 MB, 512 MB, 1 GB, ...), so it is a cheap
  way to look further into one stream than BigCrush does.

anvl's generator throughput (draw plus read-back into R) is about 250 MB/s on
the dev machine, so there it feeds Crush in about 10 minutes and BigCrush in
about 95 minutes, less than TestU01 itself needs. On the `lrz-cpu` nodes it is
37-40 MB/s, and the generator sets the pace: BigCrush takes about 11 h there,
of which TestU01 itself uses 5.5 h of CPU time, and PractRand about 27 s per GB.

### Reading a report

TestU01 lists only the statistics whose p-value lies outside
[0.001, 0.999]; "All tests were passed" means none did. With 144-160
statistics, a p-value of 1e-4 or so turns up by chance in about one run in
ten: the user guide calls p in [1e-10, 1e-4] *suspicious* and asks for a
rerun, and p outside [1e-10, 1 - 1e-10] a *clear failure*. A suspicious value
that does not recur with another seed is noise; one that recurs, or
a clear failure, is a defect. The same reading applies to PractRand's
"unusual" (noise) and "suspicious" / "FAIL" evaluations.

## How the paper used them

Salmon et al. call a generator **Crush-resistant** if it passes SmallCrush,
Crush and BigCrush, and for each of their counter-based generators report the
fewest rounds at which it is. For ThreeFry-2x32 that is 13 rounds; the
recommended 20 rounds, which anvl uses, leave a safety margin of 7. They also
ran the batteries on more than the plain incrementing-counter stream: keys and
counters varied in unusual patterns (e.g. only some words or bits changing),
and many keys. This harness covers the parts reachable through anvl's API:

- many keys derived from consecutive counters of one key, each used for a
  few values: the default, with `--per-call`;
- many related root keys: `--streams`, consecutive seeds being keys that
  differ in their low bits;
- the low bits of each word: `TESTU01_OPTS=-r`.

The paper's unusual counter patterns would need a C reimplementation of the
hash, which tests the algorithm rather than anvl.

## Results

Results for the stream described above, from anvl's `feat/rng-split-review`
(`8311ca1`, with pjrt `00a228f`), on the `lrz-cpu` nodes of the LRZ AI Systems
on 2026-11-09 and 2026-11-10. Every run passed.

SmallCrush:

| Run                                       | Smallest / largest p-value |
|-------------------------------------------|----------------------------|
| `--per-call 1`                            | 0.13 / 0.98                |
| `--per-call 1 --seed 1`                   | 0.015 / 0.99               |
| `--per-call 1`, `-r`                      | 0.013 / 0.94               |
| `--per-call 2`                            | 0.034 / 0.91               |
| `--per-call 2`, `-r`                      | 0.15 / 0.99                |
| `--per-call 4`                            | 0.0038 / 0.93              |
| `--per-call 3 --dtype ui64`               | 0.16 / 0.93                |
| `--per-call 1 --streams 64`               | 0.027 / 0.99               |
| `--per-call 1 --mode unif`                | 0.22 / 0.94                |

BigCrush passed all 160 statistics on the stream of one value per sampling
call, the case in which each key is used for the fewest bits:

| Run                    | Smallest / largest p-value | Time   |
|------------------------|----------------------------|--------|
| `--per-call 1`         | 0.020 / 0.990              | 10.8 h |
| `--per-call 1`, `-r`   | 0.0067 / 0.995             | 10.9 h |

PractRand rated nothing suspicious or worse. Each "unusual" evaluation came from
a different test and did not recur at the next length:

| Run                         | Tested up to | Time   | "Unusual" evaluations                                          |
|-----------------------------|--------------|--------|----------------------------------------------------------------|
| `--per-call 1`              | 1 TB         | 7.7 h  | BCFN at 512 GB (p = 1 - 8.8e-4), [Low1/32]mod3n at 1 TB (p = 1 - 2.4e-4) |
| `--per-call 1 --streams 16` | 64 GB        | 37 min | [Low1/32]BCFN at 256 MB (p = 1 - 3.0e-4)                        |

Crush has not been run on this stream. It passed all 144 statistics with
`--per-call 1` (smallest / largest p-value 0.0049 / 0.998) and with reversed
bits (0.019 / 0.996) on 2026-10-15, on the stream of `feat/rng-split` at the
time, which made two `ui32` values of each hash, its low and its high word,
where the current stream xors them into one.
