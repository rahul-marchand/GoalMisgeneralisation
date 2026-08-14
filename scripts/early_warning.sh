#!/usr/bin/env bash
# Does the representation move before the behaviour does?
#
#   bash scripts/early_warning.sh [DATA_DIR]
#
# The guiding hypothesis in CLAUDE.md is that an agent's internal plan
# representations predict goal misgeneralisation *before* it appears in
# behaviour. Nothing has tested it. Figure 6 of Experiment 1 shows the proxy gap
# emerging after competence at ~20M steps, but that compares behaviour against
# behaviour -- a return gap against a return curve -- and says nothing about
# whether anything internal moved first.
#
# The data to test it is already on disk. ``eval_at_steps`` saves every 195
# updates for the first 20M steps, which is roughly one checkpoint per million,
# and competence arrives at about 20M. So there are ~20 checkpoints sitting
# exactly in the window where the question lives, on every 11x11 agent, and this
# needs inference rather than training.
#
# Two agents, differing in one variable:
#
#   maze11     rho=1.0, colour perfectly predicts which objective is richer
#   clean11fv  rho=0.5, colour says nothing, everything else identical
#
# At each checkpoint, two measurements:
#
#   002  the behavioural gap -- choice quality at rho=1.0 against rho=0.0. This
#        is the thing that is supposed to lag.
#   003  the plan probe, fitted at rho=1.0 and scored at rho=0.0. If the probe's
#        read of where the agent is going degrades under the swept correlation
#        before the choices do, that is the claim.
#
# What would refute it: the two curves moving together, or the probe gap opening
# only after the behavioural one. Both are real outcomes and both are worth the
# afternoon this costs, since the alternative is a thesis resting on an
# untested premise.

set -euo pipefail
export PATH="${HOME}/.local/bin:${PATH}"

# JAX preallocates ~90% of the card by default. A training run holding that
# leaves nothing for an evaluation process, which then dies on
# `gpusolverDnCreate failed: cuSolver internal error` -- and dies per checkpoint,
# so the sweep completes while producing nothing. That is exactly how the first
# attempt lost 33 of its 42 sections: the failures were logged per checkpoint and
# the run still reported success at the end.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.15}"
export PYTHONUNBUFFERED=1

failures=0
cd "$(dirname "$0")/.."

DATA="${1:-/workspace/data}"
OUT="${OUT:-results/early-warning.txt}"
EPISODES="${EPISODES:-512}"
# Every checkpoint below this many steps, plus a few beyond it for contrast.
# Competence lands at ~20M and the dense saves stop at 20M, so this is the whole
# pre-competence window and the first two sparse points after it.
MAX_STEPS="${MAX_STEPS:-40000000}"

mkdir -p "$(dirname "${OUT}")"

AGENTS=("maze11.s1234" "clean11fv.s1234")
LEVELS="${DATA}/levels/values/1.00-0.50@1M"

{
    echo "Does an internal signature move before the behavioural gap does?"
    echo
    echo "maze11 is the proxy agent (rho=1.0), clean11fv the single-variable control"
    echo "(rho=0.5). Checkpoints up to ${MAX_STEPS} steps, which covers the dense saves"
    echo "before competence at ~20M."
    echo
} > "${OUT}"

for agent in "${AGENTS[@]}"; do
    run="${DATA}/runs/${agent}"
    if [ ! -d "${run}/local-files" ]; then
        echo "  ${agent}: not on the volume, skipping" | tee -a "${OUT}"
        continue
    fi
    for checkpoint in $(ls -d "${run}"/local-files/cp_* 2>/dev/null | sort -t_ -k2 -n); do
        steps="${checkpoint##*cp_}"
        [ "${steps}" -le "${MAX_STEPS}" ] || continue

        echo "=== ${agent} @ ${steps} steps ===" | tee -a "${OUT}"

        # Behaviour: the gap that Figure 6 already shows, per checkpoint.
        uv run python experiments/002_measure_proxy.py "${checkpoint}" \
            --levels "${LEVELS}" \
            --episodes "${EPISODES}" \
            --correlations 1.0 0.0 >> "${OUT}" 2>&1 || { echo "  002 FAILED" >> "${OUT}"; failures=$((failures + 1)); }

        # Representation: fit the plan probe where the proxy holds, score it
        # where it is reversed. A probe that stops reading the plan under the
        # swept correlation has noticed something the choices have not yet shown.
        uv run python experiments/003_probe_plan.py "${checkpoint}" \
            --levels "${LEVELS}" \
            --correlation 1.0 >> "${OUT}" 2>&1 || { echo "  003 FAILED" >> "${OUT}"; failures=$((failures + 1)); }
    done
done

echo
if [ "${failures}" -gt 0 ]; then
    echo "WARNING: ${failures} measurements failed. The sweep is incomplete -- do not"
    echo "read a trajectory off it. Check for cuSolver errors, which mean the GPU was"
    echo "already held by a training run."
fi
echo "Wrote ${OUT}"
echo "Plot the behavioural rho gap and the probe gap against steps on one axis."
echo "The claim is that the probe curve separates first; anything else refutes it."
