#!/usr/bin/env bash
set -uo pipefail
cd /workspace/GoalMisgeneralisation
export PATH="$HOME/.local/bin:$PATH"
for s in 1 2 3; do
    uv run python scripts/steer_flip_bc.py /workspace/data/offline/runs/bcnv11.s$s \
        --flip figures/data/h1/flip/bcnv11.s$s.npz \
        > results/steer-flip-bcnv11.s$s.txt 2> results/steer-flip-bcnv11.s$s.log \
        && echo "s$s ok" >> results/steer-flip.status \
        || echo "s$s FAILED" >> results/steer-flip.status
done
echo done >> results/steer-flip.status
