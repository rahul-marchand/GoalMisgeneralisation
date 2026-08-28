#!/usr/bin/env bash
set -uo pipefail
cd /workspace/GoalMisgeneralisation
export PATH="$HOME/.local/bin:$PATH"
mkdir -p figures/data/drc
for s in s1234 s5678 s9012; do
    uv run python scripts/decode_h1_drc.py /workspace/data/runs/novalue11.$s \
        --episodes 50000 --out figures/data/drc/novalue11.$s.npz \
        > results/decode-h1-drc.$s.txt 2> results/decode-h1-drc.$s.log \
        && echo "$s ok" >> results/decode-h1-drc.status \
        || echo "$s FAILED" >> results/decode-h1-drc.status
done
echo done >> results/decode-h1-drc.status
