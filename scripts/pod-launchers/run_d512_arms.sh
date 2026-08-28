#!/usr/bin/env bash
# Decode a subset of d512l4's arms; $1 = "o0" or "o1" (one sweep per pod).
set -uo pipefail
sweep=$1
cd /workspace/GoalMisgeneralisation
export PATH="$HOME/.local/bin:$PATH"
D=/workspace/data/offline/demos/scaling
mkdir -p figures/data/scaling/arms.d512l4
for a in $(ls /workspace/data/offline/runs/scaling/sc11.d512l4.s1/arms/ | grep "^$sweep"); do
    out=figures/data/scaling/arms.d512l4/$a.npz
    [ -f "$out" ] && continue
    uv run python scripts/decode_h1.py /workspace/data/offline/runs/scaling/sc11.d512l4.s1/arms/$a 20000 $out \
        --demos $D/test.rho100 >> results/arms-d512l4.$sweep.log 2>&1 \
        || { echo "$a FAILED" >> results/arms-d512l4.status; }
done
echo "$sweep ok" >> results/arms-d512l4.status
