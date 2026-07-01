#!/usr/bin/env bash
# Copy analysis outputs (and the winning algo's final models) off scratch into
# safe home storage. Idempotent (rsync) — run it after eval to snapshot the
# aggregation/stats inputs early, and again at the end to grab stats + models.
# Usage: archive_results.sh <trace_name> <archive_dir>
set -euo pipefail

TRACE="${1:?usage: archive_results.sh <trace_name> <archive_dir>}"
ARCHIVE="${2:?usage: archive_results.sh <trace_name> <archive_dir>}"

echo "Archiving result/${TRACE}/ -> ${ARCHIVE}/${TRACE}/ ..."
mkdir -p "${ARCHIVE}/${TRACE}/result"
rsync -a "result/${TRACE}/" "${ARCHIVE}/${TRACE}/result/"
rsync -a logs/run_log.csv logs/baseline_run_log.csv "${ARCHIVE}/${TRACE}/" 2>/dev/null || true

# Winning algo's final models — only once select_best has produced best_algorithm.json.
python - "${TRACE}" "${ARCHIVE}" <<'PY'
import csv, json, os, shutil, sys, pathlib
trace, arch = sys.argv[1], os.path.expanduser(sys.argv[2])
best = pathlib.Path(f"result/{trace}/best/best_algorithm.json")
if not best.exists():
    print("  (no best_algorithm.json yet — skipping model copy; run again after select_best)")
    sys.exit(0)
winner = json.loads(best.read_text())["treatment_id"]
dst = pathlib.Path(arch) / trace / "models"; dst.mkdir(parents=True, exist_ok=True)
n = 0
with open("logs/run_log.csv") as f:
    for row in csv.DictReader(f):
        if row.get("treatment_id") == winner and os.path.exists(row["model_path"]):
            shutil.copy2(row["model_path"], dst); n += 1
print(f"  copied {n} final model(s) for winner={winner}")
PY
echo "✓ Archived to ${ARCHIVE}/${TRACE}/"
