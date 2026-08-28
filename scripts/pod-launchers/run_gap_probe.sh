#!/usr/bin/env bash
# Run the distance-gap probe on all three bcnv11 seeds, sequentially.
set -uo pipefail
cd /workspace/GoalMisgeneralisation
export PATH="$HOME/.local/bin:$PATH"
for s in 1 2 3; do
    uv run python scripts/probe_bc_distance.py /workspace/data/offline/runs/bcnv11.s$s \
        > results/gap-probe-bcnv11.s$s.txt 2> results/gap-probe-bcnv11.s$s.log \
        && echo "s$s ok" >> results/gap-probe.status \
        || echo "s$s FAILED" >> results/gap-probe.status
done
echo done >> results/gap-probe.status
