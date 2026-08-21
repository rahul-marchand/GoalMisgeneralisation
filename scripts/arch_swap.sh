#!/usr/bin/env bash
# Train the architecture-swap agents: maze11 and clean11fv with the DRC replaced.
#
#   STEPS=150000000 bash scripts/arch_swap.sh resnet [DATA_DIR]
#   STEPS=150000000 bash scripts/arch_swap.sh vit    [DATA_DIR]
#
# One architecture per invocation, the proxy arm (rho=1.0) and then its control
# (rho=0.5), sequentially on one GPU, each inside the tmux session the caller
# started. Run names mirror the DRC agents they stand beside:
#
#   resnet11.s1234   resnet11clean.s1234      <- maze11.s1234 / clean11fv.s1234
#   vit11.s1234      vit11clean.s1234
#
# Everything but `--net` is what maze11 was launched with: 11x11, objectives
# worth (1.0, 0.5) with the value channel present, the 1M-level dataset
# `levels/values/1.00-0.50@1M` (the fingerprint maze11 trained on, under its
# current path), training on `train`, cleanba's evaluation on `valid` at rho
# 1.0/0.5/0.0 throughout. STEPS is deliberately not defaulted: profile first
# (PROFILE=1 runs 2M steps into a scratch directory and prints the SPS), then
# choose a budget that the learning curve, not the DRC's 150M, justifies.
#
# A run whose directory already holds a RUN COMPLETE marker is skipped, so the
# script can be re-run after an interruption and picks up the arm it lost.

set -euo pipefail
export PATH="${HOME}/.local/bin:${PATH}"
export PYTHONUNBUFFERED=1
cd "$(dirname "$0")/.."

NET="${1:?net: resnet or vit}"
DATA="${2:-/workspace/data}"
LEVELS="${LEVELS:-${DATA}/levels/values/1.00-0.50@1M}"
LOGS="${DATA}/logs/arch-swap"
mkdir -p "${LOGS}"

if [ "${PROFILE:-0}" = "1" ]; then
    STEPS="${STEPS:-2000000}"
    run="${DATA}/runs/profile-${NET}11.s1234"
    rm -rf "${run}"
    echo "profiling ${NET} for ${STEPS} steps into ${run}"
    uv run python experiments/001_maze_repro.py --net "${NET}" --correlation 1.0 \
        --min-size 11 --max-size 11 --levels "${LEVELS}" --total-timesteps "${STEPS}" \
        --run-dir "${run}" 2>&1 | tee "${LOGS}/profile-${NET}.log"
    uv run python - "${run}/metrics.csv" <<'PY'
import sys, pandas as pd
frame = pd.read_csv(sys.argv[1], index_col=0)
sps = frame["charts/0/SPS"].dropna() if "charts/0/SPS" in frame else None
print("SPS median over the run:", float(sps.median()) if sps is not None else "column missing", "  columns:", [c for c in frame.columns if "SPS" in c])
PY
    exit 0
fi

: "${STEPS:?set STEPS to the training budget chosen after profiling}"

for arm in "11 1.0" "11clean 0.5"; do
    set -- ${arm}
    name="${NET}$1.s1234"
    rho="$2"
    run="${DATA}/runs/${name}"
    if grep -qs "RUN COMPLETE" "${LOGS}/train-${name}.log"; then
        echo "${name}: already complete, skipping"
        continue
    fi
    echo "=== ${name}  (rho=${rho}, ${STEPS} steps)  $(date -u +%FT%TZ) ==="
    uv run python experiments/001_maze_repro.py --net "${NET}" --correlation "${rho}" \
        --min-size 11 --max-size 11 --levels "${LEVELS}" --total-timesteps "${STEPS}" \
        --run-dir "${run}" \
        --note "Stream arch-swap: maze11/clean11fv with the DRC(3,3) swapped for ${NET}; rho=${rho}; see RUNS.toml." \
        2>&1 | tee -a "${LOGS}/train-${name}.log"
done
echo "ALL DONE ${NET}"
