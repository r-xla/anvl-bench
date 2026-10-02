#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Is the campaign complete enough to export? Step 4 of the workflow, in one go:
#
#   1. Slurm         every sweep and validation task COMPLETED
#   2. sweep logs    every shard reached its "done:" line; no cell ERRORed
#   3. validation    every shard reached its final line; no reference FAILed
#   4. the store     anvl-sweep status: coverage at $DEPTH, errors, references
#
#   ./check.sh                        newest sweep and validation arrays
#   ./check.sh <sweep id> [<val id>]  specific ones
#
# Array ids are read off the log names ($SWEEP_ROOT/logs/sweep-<id>_<n>.out),
# newest first, so a calibration run (calib-*) is never mistaken for the sweep
# and a ./submit.sh --validate-only retry is picked up over the original.
#
# Prints a summary and exits 0 when nothing needs attention, 1 otherwise. It
# reports; it does not resubmit anything or gate export.sh. Run it on the login
# node, from this directory -- step 4 is the same seconds-long status call the
# merge job makes.
# ---------------------------------------------------------------------------
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./config.sh

require_sif

LOGS="${SWEEP_ROOT}/logs"
problems=0
warnings=0
fail() { echo "  FAIL  $*"; problems=$((problems + 1)); }
warn() { echo "  warn  $*"; warnings=$((warnings + 1)); }
ok()   { echo "  ok    $*"; }
section() { printf '\n== %s\n' "$1"; }

# Newest array id among logs named <prefix>-<id>_<task>.out.
newest_id() {
  ls "${LOGS}"/"$1"-*_*.out 2>/dev/null \
    | sed -E "s|.*/$1-([0-9]+)_[0-9]+\.out$|\1|" | sort -n | uniq | tail -n 1
}

A="${1:-$(newest_id sweep)}"
V="${2:-$(newest_id validate)}"
if [[ -z "${A}" ]]; then
  echo "no sweep logs in ${LOGS}; pass the sweep array id explicitly" >&2
  exit 2
fi
echo "sweep array ${A}, validation array ${V:-(none)}, depth ${DEPTH}, logs ${LOGS}"

# Shard numbers that do not end in the given line (missing log included).
unfinished() { # <prefix> <id> <n> <regex>
  local i
  for ((i = 1; i <= $3; i++)); do
    grep -qE "$4" "${LOGS}/$1-$2_${i}.out" 2>/dev/null || printf '%s ' "${i}"
  done
}

# ---- 1. Slurm -------------------------------------------------------------
section "Slurm task states"
if command -v sacct >/dev/null 2>&1; then
  ids="${A}${V:+,${V}}"
  bad=$(sacct -j "${ids}" -X --noheader -P --format=JobID,JobName,State,ExitCode 2>/dev/null \
    | awk -F'|' '$3 != "COMPLETED"')
  if [[ -z "${bad}" ]]; then
    ok "every task of ${ids} COMPLETED"
  else
    fail "tasks not COMPLETED (JobID|JobName|State|ExitCode):"
    echo "${bad}" | sed 's/^/          /'
  fi
else
  warn "sacct not available; skipping scheduler states"
fi

# ---- 2. sweep logs --------------------------------------------------------
section "Sweep logs (${SHARDS} shards)"
missing=$(unfinished sweep "${A}" "${SHARDS}" '^done:')
if [[ -z "${missing}" ]]; then
  ok "all ${SHARDS} shards reached 'done:'"
else
  fail "shards with no 'done:' line (killed, timed out or crashed): ${missing}"
fi

read -r n_ok n_err < <(cat "${LOGS}"/sweep-"${A}"_*.out 2>/dev/null \
  | awk '/^done:/ {ok += $2; err += $4} END {print ok + 0, err + 0}')
empty=$(grep -l '^done: 0 ok, 0 error' "${LOGS}"/sweep-"${A}"_*.out 2>/dev/null | wc -l | tr -d " ")
echo "        ${n_ok} cells ok, ${n_err} errored; ${empty} shard(s) had no cells"
# Cell ids are <spec>/<backend>/..., so this shows whether every backend ran.
echo "        ok by backend: $(cat "${LOGS}"/sweep-"${A}"_*.out 2>/dev/null \
  | awk '/^\[ *[0-9]+\/ *[0-9]+\] OK / {split($NF, p, "/"); n[p[2]]++}
         END {for (b in n) printf "%s %d  ", b, n[b]}')"
if ((n_err > 0)); then
  fail "cells that errored:"
  grep -H '\] ERROR ' "${LOGS}"/sweep-"${A}"_*.out | sed "s|^${LOGS}/|          |"
fi

# The same line in every .err (a container warning, say) is shown once, counted.
errfiles=$(find "${LOGS}" -name "sweep-${A}_*.err" -size +0 2>/dev/null)
if [[ -n "${errfiles}" ]]; then
  warn "$(echo "${errfiles}" | wc -l | tr -d " ") non-empty .err file(s); distinct lines, most frequent first:"
  echo "${errfiles}" | xargs cat | sort | uniq -c | sort -rn | head -n 10 \
    | awk '{n = $1; $1 = ""; printf "          %6d x %s\n", n, substr($0, 2, 100)}'
fi

# ---- 3. validation logs ---------------------------------------------------
section "Validation logs (${VALIDATE_SHARDS} shards)"
if [[ -z "${V}" ]]; then
  fail "no validation logs found; run ./submit.sh --validate-only"
else
  # A shard writes its records only at the very end, so no final line means it
  # recorded nothing. "nothing to validate" is a shard that had no work.
  missing=$(unfinished validate "${V}" "${VALIDATE_SHARDS}" '^validation |^nothing to validate')
  if [[ -z "${missing}" ]]; then
    ok "all ${VALIDATE_SHARDS} shards finished"
  else
    fail "shards with no final line (recorded nothing): ${missing}"
  fi
  read -r n_pass n_fail < <(cat "${LOGS}"/validate-"${V}"_*.out 2>/dev/null \
    | awk '/^validation / {p += $3; f += $5} END {print p + 0, f + 0}')
  echo "        ${n_pass} references passed, ${n_fail} failed"
  if ((n_fail > 0)); then
    fail "failed references:"
    grep -H '\] FAIL ' "${LOGS}"/validate-"${V}"_*.out | sed "s|^${LOGS}/|          |"
  fi
fi

# ---- 4. the store ---------------------------------------------------------
section "Store status (anvl-sweep status)"
mkdir -p "${SWEEP_ROOT}/home" "${SWEEP_ROOT}/tmp"
STATUS=$("${SINGULARITY}" exec --cleanenv \
  --bind "${SWEEP_ROOT}:/sweeps" \
  --env "NV_SWEEP_STORE=/sweeps/store" \
  --home "${SWEEP_ROOT}/home" \
  --env "TMPDIR=/sweeps/tmp" \
  "${SIF}" \
  anvl-sweep status --backends "${BACKENDS}" ${FILTER:+--filter "${FILTER}"} 2>&1)
rc=$?
if ((rc != 0)); then
  fail "anvl-sweep status exited ${rc}:"
  echo "${STATUS}" | tail -n 20 | sed 's/^/          /'
else
  # The coverage row for this platform at this depth, e.g.
  #   "    full    256 of  256 cells   (complete)"
  row=$(echo "${STATUS}" | awk -v pk="${ARTIFACT_ID}" -v d="${DEPTH}" '
    /^  [^ ]/ { cur = $1 }
    cur == pk && $1 == d && /cells/ { print; exit }')
  if [[ -z "${row}" ]]; then
    fail "no '${DEPTH}' coverage row for platform ${ARTIFACT_ID} (wrong ARTIFACT_ID, or nothing merged?)"
  elif [[ "${row}" == *"(complete)"* && "${row}" != *errored* ]]; then
    ok "${ARTIFACT_ID}: $(echo "${row}" | sed 's/^ *//')"
  else
    fail "${ARTIFACT_ID}: $(echo "${row}" | sed 's/^ *//')"
  fi

  # Older harnesses print no ERRORS section; section 2 covers errors from the
  # logs either way.
  errs=$(echo "${STATUS}" | grep -o '^ERRORS ([0-9]*)')
  if [[ -z "${errs}" ]]; then
    echo "        (this harness's status lists no errors; see the sweep logs above)"
  elif [[ "${errs}" == "ERRORS (0)" ]]; then
    ok "no cell's newest attempt errored"
  else
    fail "${errs} -- see the full status below"
  fi

  for kind in stable gradient; do
    line=$(echo "${STATUS}" | grep -E "^  ${kind} references" | sed -E 's/^ +//; s/ +/ /g')
    if [[ "${line}" == *failed* || "${line}" == *"no identity"* ]]; then
      fail "${line}"
    elif [[ "${line}" == *"not validated"* ]]; then
      warn "${line} (unvalidated references exclude nothing from the export)"
    else
      ok "${line}"
    fi
  done
fi

# ---- summary --------------------------------------------------------------
section "Summary"
if ((problems == 0)); then
  echo "  nothing failed (${warnings} warning(s)). Next: ./export.sh, then read its"
  echo "  warnings and the coverage counts in ${SWEEP_ROOT}/export/manifest.json."
else
  echo "  ${problems} problem(s), ${warnings} warning(s) -- resolve before ./export.sh."
  echo "  See 'Checking completion and recovering work' in README.md."
fi
echo "  Note: the store is shared by every submission in this SWEEP_ROOT, so"
echo "  section 4 also counts cells swept by earlier runs at the same depth."

if ((problems > 0)) || [[ "${CHECK_VERBOSE:-}" == 1 ]]; then
  section "Full anvl-sweep status"
  echo "${STATUS:-}"
fi

((problems == 0))
