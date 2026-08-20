#!/usr/bin/env bash
# The whole value-axis pipeline for one swapped-in architecture, unattended.
#
#   bash scripts/arch_swap_value_axis.sh AGENT NET STEPS [DATA_DIR]
#   e.g.  bash scripts/arch_swap_value_axis.sh resnet11novalue.s1234 resnet 150000000
#         WAIT_LOG=train-vit11clean.s1234.log bash scripts/arch_swap_value_axis.sh vit11novalue.s1234 vitl 200000000
#
# Meant to sit in one tmux session on the pod and need nobody: the sessions
# that launch and watch it die every few hours, and the compute must not. Each
# stage skips whatever is already on disk (the campaign.sh idiom), so the
# script can be re-run after an interruption and picks up where it stopped.
#
#   1  train the hidden-value base (novalue11's setup, the network swapped),
#      or wait for a trainer that is already running it
#   2  002 at rho=1.0 on the test split: is the base competent (DRC ~95%)?
#   3  BASE.json, which the sweep driver reads for the checkpoint and values
#   4  the o0 and o1 sweeps (scripts/value_axis_sweep.py, 400k per arm, the
#      design grid), resumable arm by arm
#   5  014 on each sweep (+ leave-one-out on o0, + extrapolation on o1) and
#      015 across both, with the exact flags the DRC results were made with
#
# Results land in results/arch-swap-value-axis-<AGENT>-*.txt in the checkout
# this runs from, to be scp'd and committed. Logs under DATA/logs/arch-swap/.

set -uo pipefail
export PATH="${HOME}/.local/bin:${PATH}"
export PYTHONUNBUFFERED=1
cd "$(dirname "$0")/.."

AGENT="${1:?agent run name, e.g. resnet11novalue.s1234}"
NET="${2:?net: resnet, vit or vitl}"
STEPS="${3:?base training steps}"
DATA="${4:-/workspace/data}"
LEVELS="${LEVELS:-${DATA}/levels/values/1.00-0.50@1M}"        # the base's training levels
TEST_LEVELS="${TEST_LEVELS:-${DATA}/levels/values/1.00-0.50@500k}"  # what 014/015 roll out on (DRC's)
ARM_STEPS="${ARM_STEPS:-400000}"
CHECKPOINTS="${CHECKPOINTS:-4}"
LOGS="${DATA}/logs/arch-swap"
run="${DATA}/runs/${AGENT}"
train_log="${LOGS}/train-${AGENT}.log"
mkdir -p "${LOGS}" results

stamp() { date -u +%FT%TZ; }
say() { echo "[$(stamp)] $*"; }

# ---- 1  base agent -------------------------------------------------------
if ! grep -qs "RUN COMPLETE" "${train_log}"; then
    if pgrep -f -- "--run-dir ${run}( |$)" >/dev/null; then
        say "a trainer for ${AGENT} is already running; waiting for RUN COMPLETE in ${train_log}"
        until grep -qs "RUN COMPLETE" "${train_log}"; do sleep 60; done
    else
        if [ -d "${run}/local-files" ] && [ -n "$(ls -A "${run}/local-files" 2>/dev/null)" ]; then
            say "!! ${run} has checkpoints but no RUN COMPLETE and no trainer: a dead run. Decide by hand (resume or move it aside); not retraining over it."
            exit 1
        fi
        if [ -n "${WAIT_LOG:-}" ]; then
            say "waiting for RUN COMPLETE in ${LOGS}/${WAIT_LOG} before taking the GPU"
            until grep -qs "RUN COMPLETE" "${LOGS}/${WAIT_LOG}"; do sleep 60; done
        fi
        say "=== train ${AGENT} (--net ${NET}, hidden values, rho=1.0, ${STEPS} steps) ===" | tee -a "${train_log}"
        uv run python experiments/001_maze_repro.py --net "${NET}" --correlation 1.0 --hide-values \
            --min-size 11 --max-size 11 --levels "${LEVELS}" --total-timesteps "${STEPS}" \
            --run-dir "${run}" \
            --note "Stream arch-swap (redirect 2026-08-20): novalue11 with the DRC(3,3) swapped for ${NET}; hidden values, rho=1.0; the base for the ${NET} value-axis sweeps. See RUNS.toml." \
            2>&1 | tee -a "${train_log}"
        grep -qs "RUN COMPLETE" "${train_log}" || { say "!! training did not complete"; exit 1; }
    fi
fi
final="$(ls -d "${run}"/local-files/cp_* | sort -t_ -k2 -n | tail -1)"
say "base ${AGENT} complete; final checkpoint ${final##*/}"

# The evaluators do not need 90% of the card; the arms (013 = cleanba training)
# keep cleanba's defaults, so the evaluator settings are per command, not global.
EVAL_ENV=(env XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.3}")

done_file() { [ -s "$1" ] && ! grep -qs "Traceback" "$1"; }

# ---- 2  competence ---------------------------------------------------------
out="results/arch-swap-value-axis-${AGENT}-competence.txt"
if done_file "${out}"; then say "competence present"; else
    say "=== 002 competence at rho=1.0 ==="
    {
        echo "Hidden-value base ${AGENT}, final checkpoint ${final##*/}; 002 at rho=1.0 on the test split."
        echo "Measured $(stamp) by scripts/arch_swap_value_axis.sh. DRC novalue11 reaches ~95% chose_optimal here."
        echo
        "${EVAL_ENV[@]}" uv run python experiments/002_measure_proxy.py "${final}" --levels "${LEVELS}" --episodes 2048 --correlations 1.0 \
            --json "figures/data/${AGENT}.json"
    } > "${out}" 2>&1 || say "!! 002 failed, see ${out}"
fi

# ---- 3  BASE.json ----------------------------------------------------------
if [ -s "${run}/BASE.json" ]; then say "BASE.json present: $(tr -d '\n ' < "${run}/BASE.json")"; else
    n="$(ls -d "${run}"/local-files/cp_* | wc -l)"
    printf '{\n  "checkpoint": "local-files/%s",\n  "values": [1.0, 0.5],\n  "objectives": 2,\n  "steps": %s,\n  "checkpoints_saved": %s\n}\n' \
        "${final##*/}" "${STEPS}" "${n}" > "${run}/BASE.json"
    say "wrote ${run}/BASE.json -> ${final##*/}"
fi

# ---- 4  sweeps -------------------------------------------------------------
sweep_log="${LOGS}/sweep-${AGENT}.log"
for attempt in 1 2; do
    if grep -qs "SWEEP_COMPLETE" "${sweep_log}"; then break; fi
    say "=== sweep o0+o1 of ${AGENT}, ${ARM_STEPS} steps per arm (attempt ${attempt}) ==="
    uv run python scripts/value_axis_sweep.py --data "${DATA}" --agent "${AGENT}" --objectives 0 1 \
        --steps "${ARM_STEPS}" --checkpoints "${CHECKPOINTS}" >> "${sweep_log}" 2>&1 \
        || say "!! sweep exited non-zero, see ${sweep_log}"
done
grep -qs "SWEEP_COMPLETE" "${sweep_log}" || { say "!! sweep incomplete after two attempts; stopping before analysis"; exit 1; }
say "sweep complete: $(ls -d "${run}"/arms/o*@* 2>/dev/null | wc -l) arm directories"

# ---- 5  analysis -----------------------------------------------------------
ARMS="${run}/arms"
analyse() {  # outfile script args...
    local out="results/arch-swap-value-axis-${AGENT}-$1.txt"; shift
    if done_file "${out}"; then say "${out} present"; return; fi
    say "=== $1 -> ${out} ==="
    "${EVAL_ENV[@]}" uv run python "$@" > "${out}" 2>&1 || say "!! $1 failed, see ${out}"
}
analyse o0         experiments/014_value_axis_analysis.py --base "${final}" --arms "${ARMS}" --levels "${TEST_LEVELS}" --prefix o0 --base-value 1.0 --arm-steps "${ARM_STEPS}" --at -1
analyse o0-heldout experiments/014_value_axis_analysis.py --base "${final}" --arms "${ARMS}" --levels "${TEST_LEVELS}" --prefix o0 --base-value 1.0 --arm-steps "${ARM_STEPS}" --at -1 --leave-one-out
analyse o1         experiments/014_value_axis_analysis.py --base "${final}" --arms "${ARMS}" --levels "${TEST_LEVELS}" --prefix o1 --base-value 0.5 --arm-steps "${ARM_STEPS}" --at -1 --extrapolate 1.00 1.05 1.10 1.20 1.30 1.50
analyse value-or-gap experiments/015_value_or_gap.py --base "${final}" --arms "${ARMS}" --levels "${TEST_LEVELS}" --arm-steps "${ARM_STEPS}" --skip-behaviour
say "VALUE-AXIS DONE ${AGENT}"
