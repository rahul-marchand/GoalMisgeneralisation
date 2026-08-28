#!/usr/bin/env bash
set -uo pipefail
cd /workspace/GoalMisgeneralisation
export PATH="$HOME/.local/bin:$PATH"
D=/workspace/data/offline/demos/scaling
R=/workspace/data/offline/runs/scaling/sc11.d512l4.s1
uv run python scripts/steer_flip_bc.py $R --flip figures/data/scaling/flip.d512l4.npz \
    --probe-demos $D/train.rho100 --eval-demos $D/test.rho100 \
    > results/steer-flip-d512l4.txt 2> results/steer-flip-d512l4.log \
    && echo "steer ok" >> results/d512-chain.status || echo "steer FAILED" >> results/d512-chain.status
