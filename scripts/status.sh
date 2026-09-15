#!/usr/bin/env bash
# One-command status for a running (or finished) experiment.
#
#   ./scripts/status.sh            # once
#   ./scripts/status.sh -w         # refresh every 30s
#
# Reads only on-disk artifacts, never the terminal the run was launched from -- so it works
# whether the run is in tmux, under nohup, or already finished, and it is safe to call while
# the run is writing.

set -uo pipefail
cd "$(dirname "$0")/.."
: "${OUT:=outputs/harness}"

summarise() {
  printf '\n%s  [%s]\n' "════════════════════════════════" "$(date '+%F %T')"

  # Counted by piping to wc, not with `pgrep -c`: BSD/macOS pgrep has no -c flag, and the usage
  # error it prints instead is easy to read past -- leaving the count blank while the table below
  # still renders, which looks like "no processes" rather than like a broken command.
  local alive
  alive=$(pgrep -f "alphaapollo.workflows.evo" 2>/dev/null | wc -l | tr -d ' ')
  if [ "${alive:-0}" -gt 0 ]; then
    printf 'processes : %s alive\n' "$alive"
  else
    printf 'processes : none running\n'
  fi

  printf '\n%-18s %6s %7s %7s %6s %8s %s\n' RUN DONE PASS@1 FINAL ERRORS BATCHES UPDATED
  for phase in adapt heldout; do
    for arm in baseline raw evo; do
      d="$OUT/$phase-$arm"
      m="$d/metrics.jsonl"
      [ -f "$m" ] || continue
      python - "$m" "$d" "$phase-$arm" <<'PY'
import json, sys, time
from pathlib import Path
path, run_dir, name = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
rows = []
for line in path.read_text(encoding="utf-8").splitlines():
    try:
        r = json.loads(line)
    except json.JSONDecodeError:
        continue          # a torn last line is normal while the run is writing
    if "adapt/pass_final" in r:
        rows.append(r)
ok = [r for r in rows if not r.get("adapt/error")]
n = len(ok) or 1
r0 = 100 * sum(r.get("adapt/pass1_round0", 0) for r in ok) / n
fin = 100 * sum(r.get("adapt/pass_final", 0) for r in ok) / n
err = sum(1 for r in rows if r.get("adapt/error"))
prog = run_dir / "progress.json"
batches = json.loads(prog.read_text())["completed_batches"] if prog.exists() else 0
age = time.time() - path.stat().st_mtime
age_s = f"{age/60:.0f}m ago" if age > 90 else f"{age:.0f}s ago"
print(f"{name:<18} {len(rows):>6} {r0:>6.1f}% {fin:>6.1f}% {err:>6} {batches:>8} {age_s}")
PY
    done
  done

  # A stalled run looks identical to a slow one in the table above; the log's last line is what
  # separates them.
  printf '\n'
  for f in "$OUT"/*.log; do
    [ -f "$f" ] || continue
    last=$(grep -vE "^wandb:|^\s*$" "$f" 2>/dev/null | tail -1 | cut -c1-110)
    [ -n "$last" ] && printf '%-18s %s\n' "$(basename "$f" .log)" "$last"
  done
}

if [ "${1:-}" = "-w" ]; then
  while true; do clear; summarise; sleep 30; done
else
  summarise
fi
