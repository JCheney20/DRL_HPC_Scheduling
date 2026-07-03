#!/usr/bin/env bash
# Copy analysis outputs and every algorithm's final model off scratch into safe
# home storage. Idempotent (rsync). Run ONCE at the end, after select_best has
# decided the winner (best_algorithm.json) — the model copy is gated on it.
# Usage: archive_results.sh <trace_name> <archive_dir>
set -euo pipefail

TRACE="${1:?usage: archive_results.sh <trace_name> <archive_dir>}"
ARCHIVE="${2:?usage: archive_results.sh <trace_name> <archive_dir>}"

echo "Archiving result/${TRACE}/ -> ${ARCHIVE}/${TRACE}/ ..."
mkdir -p "${ARCHIVE}/${TRACE}/result"
rsync -a "result/${TRACE}/" "${ARCHIVE}/${TRACE}/result/"
rsync -a logs/run_log.csv logs/baseline_run_log.csv "${ARCHIVE}/${TRACE}/" 2>/dev/null || true

# Every algorithm's final model — but only once select_best has decided the
# winner (best_algorithm.json), i.e. the pipeline has finished.
python - "${TRACE}" "${ARCHIVE}" <<'PY'
import csv, json, os, shutil, sys, pathlib
trace, arch = sys.argv[1], os.path.expanduser(sys.argv[2])
best = pathlib.Path(f"result/{trace}/best/best_algorithm.json")
if not best.exists():
    print("  (no best_algorithm.json yet — pipeline unfinished; skipping model copy)")
    sys.exit(0)
winner = json.loads(best.read_text()).get("treatment_id")
dst = pathlib.Path(arch) / trace / "models"
n = 0
with open("logs/run_log.csv") as f:
    for row in csv.DictReader(f):
        mp = row.get("model_path")
        if mp and os.path.exists(mp):
            out = dst / mp            # preserve relative path so per-seed/algo models don't collide
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(mp, out); n += 1
print(f"  copied {n} final model(s) across all algorithms (winner={winner})")
PY
echo "✓ Archived to ${ARCHIVE}/${TRACE}/"
