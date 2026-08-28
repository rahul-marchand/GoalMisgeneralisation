#!/usr/bin/env bash
set -uo pipefail
s=$1
cd /workspace/GoalMisgeneralisation
export PATH="$HOME/.local/bin:$PATH"
uv run python scripts/steer_flip_bc.py /workspace/data/offline/runs/bcnv11.s$s \
    --flip figures/data/h1/flip/bcnv11.s$s.npz \
    > results/steer-flip-bcnv11.s$s.txt 2> results/steer-flip-bcnv11.s$s.log \
    && echo ok > results/steer-flip.s$s.status \
    || echo FAILED > results/steer-flip.s$s.status
