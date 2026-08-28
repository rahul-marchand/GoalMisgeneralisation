#!/usr/bin/env bash
set -uo pipefail
cd /workspace/GoalMisgeneralisation
export PATH="$HOME/.local/bin:$PATH"
for s in 1 2 3; do
    uv run python scripts/distance_field_bc.py /workspace/data/offline/runs/bcnv11.s$s \
        > results/distance-field-bcnv11.s$s.txt 2> results/distance-field-bcnv11.s$s.log \
        && echo "s$s ok" >> results/distance-field.status \
        || echo "s$s FAILED" >> results/distance-field.status
done
echo done >> results/distance-field.status
