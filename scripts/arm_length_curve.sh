#!/usr/bin/env bash
# How the axis estimate's reliability depends on how long an arm trained.
#
#   bash scripts/arm_length_curve.sh [DATA_DIR]
#
# No training at all. ``013`` saves four evenly spaced checkpoints per arm, so
# every arm already on disk is four arms at four budgets, and the whole curve can
# be read out of what is there:
#
#   seed 1234's 3M arms   ->  750k, 1.5M, 2.25M, 3M
#   seed 5678's 750k arms ->  187k, 375k, 562k, 750k
#
# Eight points from 187k to 3M, which is the range worth knowing about before the
# campaign commits to an arm length for every later phase.
#
# The reason this matters is that the numbers so far point the wrong way. Seed
# 1234's 3M arms reached a split-half reliability of 0.14; its own 750k
# checkpoint reached 0.23; seed 5678's 750k arms reached 0.27-0.29. Longer arms
# have been *worse*, which would mean the first grid paid four times the GPU for
# a noisier estimate. If that holds across the whole curve, every later phase
# gets cheaper and cleaner at once, and the right arm length may be shorter than
# 750k.
#
# The fine-tune learning rate is constant unless --anneal-to is passed, so an
# intermediate checkpoint of a 3M arm really is comparable to a shorter arm
# rather than being a point on a decaying schedule.

set -euo pipefail
export PATH="${HOME}/.local/bin:${PATH}"
cd "$(dirname "$0")/.."

DATA="${1:-/workspace/data}"
OUT="${OUT:-results/arm-length-curve.txt}"
RESAMPLES="${RESAMPLES:-2000}"

mkdir -p "$(dirname "${OUT}")"

# agent tag : arms directory : levels at the base values
AGENTS=(
    "novalue11.s1234"
    "novalue11.s5678"
)

{
    echo "How reliability depends on arm length, from checkpoints already on disk."
    echo
    echo "Each arm saved four evenly spaced checkpoints, so --at 0..3 walks a quarter,"
    echo "a half, three quarters and all of whatever that sweep's arm budget was."
    echo
} > "${OUT}"

for agent in "${AGENTS[@]}"; do
    base_json="${DATA}/runs/${agent}/BASE.json"
    if [ ! -f "${base_json}" ]; then
        echo "  ${agent}: no BASE.json, skipping (run migrate_volume.py first)" | tee -a "${OUT}"
        continue
    fi
    checkpoint="${DATA}/runs/${agent}/$(uv run python -c "import json,sys; print(json.load(open(sys.argv[1]))['checkpoint'])" "${base_json}")"
    arms="${DATA}/runs/${agent}/arms"
    levels="${DATA}/levels/values/1.00-0.50@150k"
    [ -d "${levels}" ] || levels="${DATA}/levels/values/1.00-0.50@500k"

    for at in 0 1 2 3; do
        echo "=== ${agent}, checkpoint index ${at} ===" | tee -a "${OUT}"
        # --skip-behaviour: this asks about the weight estimate only, and the
        # behavioural pass is most of the runtime.
        uv run python experiments/015_value_or_gap.py \
            --base "${checkpoint}" \
            --arms "${arms}" \
            --levels "${levels}" \
            --at "${at}" \
            --resamples "${RESAMPLES}" \
            --skip-behaviour >> "${OUT}" 2>&1 || echo "  failed at index ${at}" >> "${OUT}"
    done
done

echo
echo "Wrote ${OUT}"
echo "Read off: reliability of axis_0 and axis_1 against arm length, and whether the"
echo "permutation p-value improves or worsens as arms get longer."
