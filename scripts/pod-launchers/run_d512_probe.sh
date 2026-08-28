#!/usr/bin/env bash
set -uo pipefail
cd /workspace/GoalMisgeneralisation
export PATH="$HOME/.local/bin:$PATH"
D=/workspace/data/offline/demos/scaling
R=/workspace/data/offline/runs/scaling/sc11.d512l4.s1
uv run python scripts/probe_bc_distance.py $R --probe-demos $D/train.rho100 --eval-demos $D/test.rho100 \
    --n-eval 20000 --choices figures/data/scaling/d512l4.npz \
    > results/gap-probe-d512l4.txt 2> results/gap-probe-d512l4.log \
    && echo "probe ok" >> results/d512-chain.status || echo "probe FAILED" >> results/d512-chain.status
uv run python scripts/probe_flip_bc.py $R --flip figures/data/scaling/flip.d512l4.npz \
    --probe-demos $D/train.rho100 --eval-demos $D/test.rho100 \
    > results/probe-flip-d512l4.txt 2> results/probe-flip-d512l4.log \
    && echo "probeflip ok" >> results/d512-chain.status || echo "probeflip FAILED" >> results/d512-chain.status
