#!/usr/bin/env bash
# Wait for the value grid to finish, then write the analysis out.
#
#   setsid nohup bash scripts/value_axis_report.sh > logs/report.log 2>&1 &
#
# Detached from whoever launched it, so the grid and its analysis complete
# without a session or an ssh connection staying alive. The grid is read twice:
# at the last checkpoint, and at the first, which is already past the point where
# behaviour converged. An axis that is the same direction at both budgets is one
# the extra updates only lengthened.

set -euo pipefail
export PATH="${HOME}/.local/bin:${PATH}"
cd "$(dirname "$0")/.."

BASE="${1:-/workspace/data/runs/novalue11.s1234}"
CHECKPOINT="${CHECKPOINT:-/workspace/data/runs/novalue11.s1234/local-files/cp_140206080}"
DEADLINE=$(( SECONDS + ${WAIT_SECONDS:-14400} ))

while ! grep -q GRID_COMPLETE "${BASE}/logs/grid.log" 2>/dev/null; do
    if grep -qE "Traceback" "${BASE}/logs/grid.log" 2>/dev/null; then
        echo "grid failed; not analysing a partial sweep"
        exit 1
    fi
    if [ "${SECONDS}" -gt "${DEADLINE}" ]; then
        echo "gave up waiting for the grid"
        exit 1
    fi
    sleep 60
done

mkdir -p "${BASE}/results"
for pair in "-1 full" "0 quarter"; do
    set -- ${pair}
    at=$1; name=$2
    echo "=== analysis at ${name} budget ==="
    uv run python experiments/014_value_axis_analysis.py \
        --base "${CHECKPOINT}" \
        --arms "${BASE}/runs" \
        --levels "${BASE}/levels/v050" \
        --at "${at}" \
        > "${BASE}/results/value-axis-${name}.txt" 2>&1 || echo "  ${name} budget failed, see the file"
    tail -3 "${BASE}/results/value-axis-${name}.txt"
done

echo ANALYSIS_COMPLETE
