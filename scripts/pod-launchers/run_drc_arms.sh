#!/usr/bin/env bash
set -uo pipefail
cd /workspace/GoalMisgeneralisation
export PATH="$HOME/.local/bin:$PATH"
mkdir -p figures/data/drc/arms.s1234
for a in $(ls /workspace/data/runs/novalue11.s1234/arms/ | grep "@400k$"); do
    cp_dir=$(ls -d /workspace/data/runs/novalue11.s1234/arms/$a/local-files/cp_* 2>/dev/null | sort | tail -1)
    [ -z "$cp_dir" ] && { echo "$a NO-CHECKPOINT" >> results/drc-arms.status; continue; }
    out=figures/data/drc/arms.s1234/$a.npz
    [ -f "$out" ] && continue
    uv run python scripts/decode_h1_drc.py "$cp_dir" --episodes 20000 --values 1.0 0.5 --out "$out" \
        >> results/drc-arms.log 2>&1 || echo "$a FAILED" >> results/drc-arms.status
done
echo done >> results/drc-arms.status
