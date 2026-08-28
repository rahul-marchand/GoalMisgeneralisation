#!/usr/bin/env bash
set -uo pipefail
cd /workspace/GoalMisgeneralisation
export PATH="$HOME/.local/bin:$PATH"
uv run python scripts/margin_swap_bc.py /workspace/data/offline/runs/bcnv11.s1 \
    --eval-demos /workspace/data/offline/demos/test.rho100 --choices figures/data/h1/bcnv11.s1.npz \
    > results/margin-swap-bcnv11.s1.txt 2> results/margin-swap-bcnv11.s1.log \
    && echo "bcnv11 ok" >> results/margin-swap.status || echo "bcnv11 FAILED" >> results/margin-swap.status
uv run python scripts/margin_swap_bc.py /workspace/data/offline/runs/scaling/sc11.d512l4.s1 \
    --eval-demos /workspace/data/offline/demos/scaling/test.rho100 --choices figures/data/scaling/d512l4.npz \
    > results/margin-swap-d512l4.txt 2> results/margin-swap-d512l4.log \
    && echo "d512l4 ok" >> results/margin-swap.status || echo "d512l4 FAILED" >> results/margin-swap.status
echo done >> results/margin-swap.status
