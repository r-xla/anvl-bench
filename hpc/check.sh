#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Is the campaign complete enough to export? Step 4 of the workflow, in one go:
#
#   1. Slurm         every worker and validation task COMPLETED
#   2. worker logs   every worker reached its "worker done:" line; no part
#                    ERRORed and every cell assembled
#   3. the queue     every part done, every cell assembled (anvl-sweep queue)
#   4. validation    every shard reached its final line; no reference FAILed
#   5. the store     anvl-sweep status: coverage at $DEPTH, errors, references
#
#   ./check.sh                        newest queue and validation array
#   ./check.sh <queue> [<val id>]     specific ones
#
# A queue's worker arrays are listed in $SWEEP_ROOT/queues/<queue>/arrays --
# one per ./submit.sh or ./submit.sh --resume -- and calibration queues
# (calib-*) are never taken for the campaign. The validation array is read off
# the log names, newest first, so a ./submit.sh --validate-only retry is picked
# up over the original.
#
# Prints a summary and exits 0 when nothing needs attention, 1 otherwise. It
# reports; it does not resubmit anything or gate export.sh. Run it on the login
# node, from this directory -- steps 3 and 5 are seconds-long calls in the
# container.
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

# Shard numbers that do not end in the given line (missing log included).
unfinished() { # <prefix> <id> <n> <regex>
  local i
  for ((i = 1; i <= $3; i++)); do
    grep -qE "$4" "${LOGS}/$1-$2_${i}.out" 2>/dev/null || printf '%s ' "${i}"
  done
}

Q="${1:-$(ls "${SWEEP_ROOT}/queues" 2>/dev/null | grep -v '^calib-' | sort | tail -n 1)}"
V="${2:-$(newest_id validate)}"
QDIR="${SWEEP_ROOT}/queues/${Q}"
if [[ -z "${Q}" || ! -f "${QDIR}/plan.rds" ]]; then
  echo "no queue '${Q}' in ${SWEEP_ROOT}/queues; pass its name explicitly" >&2
  exit 2
fi
ARRAYS=$(paste -sd, "${QDIR}/arrays" 2>/dev/null)
# Only the newest pool's troubles fail the check: an earlier one's killed
# workers are what a --resume was for, and section 3 says whether it worked.
LAST=$(tail -n 1 "${QDIR}/arrays" 2>/dev/null)
older() { [[ "$1" != "${LAST}"* ]]; }
echo "queue ${Q}, worker arrays ${ARRAYS:-(none)}, validation array ${V:-(none)}, depth ${DEPTH}, logs ${LOGS}"

# ---- 1. Slurm -------------------------------------------------------------
section "Slurm task states"
if command -v sacct >/dev/null 2>&1; then
  ids="${ARRAYS}${V:+${ARRAYS:+,}${V}}"
  bad=$(sacct -j "${ids}" -X --noheader -P --format=JobID,JobName,State,ExitCode 2>/dev/null \
    | awk -F'|' '$3 != "COMPLETED"')
  old=$(echo "${bad}" | while IFS= read -r l; do [[ -n "${l}" ]] && older "${l}" && echo "${l}"; done)
  new=$(echo "${bad}" | while IFS= read -r l; do [[ -n "${l}" ]] && ! older "${l}" && echo "${l}"; done)
  if [[ -z "${bad}" ]]; then
    ok "every task of ${ids} COMPLETED"
  fi
  # A worker stops claiming before its time limit, so TIMEOUT means a part
  # took far longer than expected; its claim goes stale and is taken over.
  if [[ -n "${new}" ]]; then
    fail "tasks not COMPLETED (JobID|JobName|State|ExitCode):"
    echo "${new}" | sed 's/^/          /'
  fi
  if [[ -n "${old}" ]]; then
    warn "tasks of earlier worker pools not COMPLETED (see section 3 for what is still missing):"
    echo "${old}" | sed 's/^/          /'
  fi
else
  warn "sacct not available; skipping scheduler states"
fi

# ---- 2. worker logs -------------------------------------------------------
section "Worker logs"
wlogs=()
for a in ${ARRAYS//,/ }; do
  for f in "${LOGS}"/work-"${a}"_*.out; do [[ -e "${f}" ]] && wlogs+=("${f}"); done
done
if ((${#wlogs[@]} == 0)); then
  fail "no worker logs for arrays ${ARRAYS:-(none)}"
else
  unfinished=$(grep -L '^worker done:' "${wlogs[@]}" | sed -E "s|^${LOGS}/work-||; s|\.out$||")
  if [[ -z "${unfinished}" ]]; then
    ok "all ${#wlogs[@]} workers reached 'worker done:'"
  fi
  new=$(for w in ${unfinished}; do older "${w}" || printf '%s ' "${w}"; done)
  old=$(for w in ${unfinished}; do older "${w}" && printf '%s ' "${w}"; done)
  if [[ -n "${new}" ]]; then
    fail "workers with no 'worker done:' line (killed, timed out or crashed): ${new}"
  fi
  if [[ -n "${old}" ]]; then
    warn "workers of earlier pools with no 'worker done:' line: ${old}"
  fi
  read -r n_ok n_err < <(cat "${wlogs[@]}" \
    | awk '/^worker done:/ {ok += $3; err += $6} END {print ok + 0, err + 0}')
  echo "        ${n_ok} parts ok, ${n_err} errored"
  echo "        why workers stopped: $(cat "${wlogs[@]}" | sed -n 's/^worker done:.*stopped because //p' \
    | sort | uniq -c | awk '{n = $1; $1 = ""; printf "%s x%d; ", substr($0, 2), n}')"
  if ((n_err > 0)); then
    fail "parts that errored:"
    grep -h -A1 '^\[[0-9]*\] ERROR' "${wlogs[@]}" | grep -v '^--$' | sed 's/^/          /'
  fi
  if grep -q '^  could not assemble' "${wlogs[@]}"; then
    fail "cells that could not be assembled (./submit.sh --resume ${Q} retries them in its merge):"
    grep -h '^  could not assemble' "${wlogs[@]}" | sed 's/^  /          /'
  fi
  if grep -q 'WARNING: its parts ran on' "${wlogs[@]}"; then
    warn "$(cat "${wlogs[@]}" | grep -c 'WARNING: its parts ran on') cell(s) whose parts ran on different CPU models:"
    grep -h -B1 'WARNING: its parts ran on' "${wlogs[@]}" | grep -v '^--$' | head -n 10 | sed 's/^ */          /'
  fi
  errfiles=$(for f in "${wlogs[@]}"; do e="${f%.out}.err"; [[ -s "${e}" ]] && echo "${e}"; done)
  if [[ -n "${errfiles}" ]]; then
    warn "$(echo "${errfiles}" | wc -l | tr -d " ") non-empty .err file(s); distinct lines, most frequent first:"
    echo "${errfiles}" | xargs cat | sort | uniq -c | sort -rn | head -n 10 \
      | awk '{n = $1; $1 = ""; printf "          %6d x %s\n", n, substr($0, 2, 100)}'
  fi
fi

# ---- 3. the queue ---------------------------------------------------------
section "Queue (anvl-sweep queue)"
mkdir -p "${SWEEP_ROOT}/home" "${SWEEP_ROOT}/tmp"
QSTATUS=$("${SINGULARITY}" exec --cleanenv \
  --bind "${SWEEP_ROOT}:/sweeps" \
  --home "${SWEEP_ROOT}/home" \
  --env "TMPDIR=/sweeps/tmp" \
  "${SIF}" \
  anvl-sweep queue --queue "/sweeps/queues/${Q}" --stale-minutes "${STALE_MINUTES}" 2>&1)
rc=$?
if ((rc != 0)); then
  fail "anvl-sweep queue exited ${rc}:"
  echo "${QSTATUS}" | tail -n 20 | sed 's/^/          /'
else
  # "  cells    668 in all: 668 assembled, 0 ready to assemble, 0 still being swept"
  cells=$(echo "${QSTATUS}" | grep -E '^  cells ')
  parts=$(echo "${QSTATUS}" | grep -E '^  parts ')
  read -r n_cells n_asm < <(echo "${cells}" | awk '{print $2, $5}')
  if [[ -n "${n_cells}" && "${n_cells}" == "${n_asm}" ]]; then
    ok "every one of ${n_cells} cells assembled"
  else
    fail "$(echo "${cells}" | sed 's/^ *//')"
    echo "          $(echo "${parts}" | sed 's/^ *//')"
    echo "          more workers: ./submit.sh --resume ${Q}"
  fi
  if [[ "${parts}" == *"(0 errored)"* ]]; then
    ok "no part errored"
  else
    fail "$(echo "${parts}" | sed 's/^ *//')"
  fi
fi

# ---- 4. validation logs ---------------------------------------------------
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

# ---- 5. the store ---------------------------------------------------------
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
echo "  section 5 also counts cells swept by earlier runs at the same depth."

if ((problems > 0)) || [[ "${CHECK_VERBOSE:-}" == 1 ]]; then
  section "Full anvl-sweep queue"
  echo "${QSTATUS:-}"
  section "Full anvl-sweep status"
  echo "${STATUS:-}"
fi

((problems == 0))
