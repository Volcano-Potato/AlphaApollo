#!/usr/bin/env bash
# Run the three Task C arms across both phases.
#
#   export OPENAI_API_KEY=...            # never stored in a config; every model block reads it
#   nohup ./scripts/run_experiments.sh > run.log 2>&1 &
#   disown
#
# The nohup/disown is not decoration. Two diagnostic runs were lost mid-flight to an agent
# session tearing down its background tasks -- not to anything on this machine (the process
# holds 491MB; the box has 24GB). A multi-hour run must belong to the system, not to whatever
# shell started it.
#
# Safe to re-run after an interruption: every phase resumes from its own progress marker at
# batch granularity, having first dropped the partial rows the aborted batch left behind (see
# alphaapollo/core/harness/resume.py). Re-running a completed phase is a no-op that costs one
# process start.
#
# Usage:
#   ./scripts/run_experiments.sh                 # adaptation (parallel), then held-out
#   ./scripts/run_experiments.sh adapt           # adaptation only
#   ./scripts/run_experiments.sh heldout         # held-out only (needs adaptation finished)
#   PARALLEL=0 ./scripts/run_experiments.sh      # one arm at a time
#   ARMS="baseline evo" ./scripts/run_experiments.sh

set -uo pipefail
cd "$(dirname "$0")/.."

: "${ARMS:=baseline raw evo}"
: "${PARALLEL:=1}"
: "${CONFIG_DIR:=examples/configs}"
: "${OUT:=outputs/harness}"
PHASES="${1:-all}"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "FATAL: OPENAI_API_KEY is not set. Export it; it is deliberately absent from every config." >&2
  exit 1
fi

# The local SOCKS proxy refused ~25% of connections to DashScope, which surfaced as lost
# problems rather than as an error. Direct is both correct and faster (0.38s vs 1.00s).
export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost,::1},dashscope.aliyuncs.com"
export no_proxy="$NO_PROXY"

PY="${PY:-python}"
mkdir -p "$OUT"

log() { echo "[$(date '+%F %T')] $*"; }

run_one() {   # run_one <phase> <arm>
  local phase=$1 arm=$2
  local cfg="$CONFIG_DIR/harness_${phase}_${arm}.yaml"
  local logfile="$OUT/${phase}-${arm}.log"
  log "START $phase/$arm  -> $logfile"
  if "$PY" -m alphaapollo.workflows.evo --config "$cfg" >>"$logfile" 2>&1; then
    log "DONE  $phase/$arm"
  else
    log "FAIL  $phase/$arm (exit $?) -- see $logfile; re-run this script to resume"
    return 1
  fi
}

announce_concurrency() {
  # The number that actually matters is arms x max_workers, and it is nowhere in any single
  # config -- one says 4, the script decides there are three of them. Printing it makes the
  # multiplication visible at launch instead of at the first rate-limit storm, which shows up
  # as lost problems rather than as an error.
  local n_arms workers
  n_arms=$(echo "$ARMS" | wc -w | tr -d ' ')
  workers=$(grep -E "^\s+max_workers:" "$CONFIG_DIR/harness_base.yaml" | grep -oE "[0-9]+" | head -1)
  if [[ "$PARALLEL" == "1" ]]; then
    log "concurrency: $n_arms arms x $workers workers = $((n_arms * workers)) in flight (DashScope: 12 safe, 16 throttles)"
  else
    log "concurrency: serial, $workers workers in flight"
  fi
}

run_phase() {   # run_phase <phase>
  local phase=$1 rc=0
  announce_concurrency
  if [[ "$PARALLEL" == "1" ]]; then
    # The arms are independent: no shared state, no shared output directory. Three processes
    # is ~1.5GB, and the real ceiling is the provider's concurrency limit, not this machine.
    local pids=()
    for arm in $ARMS; do run_one "$phase" "$arm" & pids+=($!); done
    for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
  else
    for arm in $ARMS; do run_one "$phase" "$arm" || rc=1; done
  fi
  return $rc
}

# A held-out run evaluates the *final frozen* artifact. It gets a copy, never the adaptation
# run's own directory: pointing it at the original would let the evaluation append to the thing
# it is evaluating. The copy is refreshed on every held-out launch so it always reflects the
# adaptation run as it currently stands.
freeze_state() {
  for arm in $ARMS; do
    local src dst
    case "$arm" in
      evo) src="$OUT/adapt-evo/store";  dst="$OUT/heldout-evo/store" ;;
      raw) src="$OUT/adapt-raw/pool";   dst="$OUT/heldout-raw/pool"  ;;
      *)   continue ;;                  # baseline has nothing to carry across
    esac
    if [[ ! -d "$src" ]]; then
      echo "FATAL: $src does not exist -- run the adaptation phase for '$arm' first." >&2
      return 1
    fi
    rm -rf "$dst"
    mkdir -p "$(dirname "$dst")"
    cp -R "$src" "$dst"
    log "froze $src -> $dst"
  done
}

# The held-out phase must not inherit the adaptation phase's resume marker: it is a different
# run over a different stream, and `resume.plan_resume` would reject the fingerprint anyway.
# Clearing it here makes a re-frozen state produce a clean re-evaluation rather than an error.
clear_heldout_progress() {
  for arm in $ARMS; do rm -f "$OUT/heldout-${arm}/progress.json"; done
}

rc=0
case "$PHASES" in
  adapt)   run_phase adapt || rc=1 ;;
  heldout) freeze_state && clear_heldout_progress && { run_phase heldout || rc=1; } || rc=1 ;;
  all)
    run_phase adapt || rc=1
    if [[ $rc -eq 0 ]]; then
      freeze_state && clear_heldout_progress && { run_phase heldout || rc=1; } || rc=1
    else
      log "SKIP held-out: the adaptation phase did not finish cleanly"
    fi
    ;;
  *) echo "unknown phase '$PHASES' (expected: adapt | heldout | all)" >&2; exit 2 ;;
esac

# Derived artifacts: harness.md / summary.json / evolution.jsonl. Cheap, and regenerable from
# the store at any time, so a failure here is reported but does not change the run's exit code.
if [[ "$PHASES" != "adapt" && $rc -eq 0 ]]; then
  for arm in $ARMS; do
    case "$arm" in
      evo) store="$OUT/adapt-evo/store" ;;
      *)   continue ;;
    esac
    "$PY" -m alphaapollo.core.harness.export --store_root "$store" \
        --out_dir "$OUT/export-$arm" >"$OUT/export-$arm.log" 2>&1 \
        && log "exported $OUT/export-$arm" || log "WARN export failed for $arm"
  done
fi

log "exit $rc"
exit $rc
