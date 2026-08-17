#!/usr/bin/env bash
# The rest of the robustness campaign, in one resumable chain.
#
#   ARM_STEPS=200000 bash scripts/campaign.sh [DATA_DIR]
#
# Every stage skips whatever is already on disk, so this can be re-run after an
# interruption without repeating work — the same idiom as the sweep driver and
# the original three_objective.sh. That matters more than usual here: the chain
# is twelve hours long and a reclaimed pod should cost the stage it happened in,
# not the campaign.
#
# ARM_STEPS is not defaulted on purpose. The arm-length curve says reliability
# rises monotonically as arms get shorter across every point measured, and the
# curve never turned over, so there is no defensible default until the low end
# has been read. Pass the length the 400k sweep's eight checkpoints picked out.
#
# Order is by evidence per GPU-hour, so an interrupted chain has done the most
# valuable things first:
#
#   1  seed 5678's wide sweep      ~0.3 h   makes the headline claim n=2 at high leverage
#   2  base-checkpoint ladder      ~0.6 h   was meant to gate the 250M extension
#   3  third two-objective seed    ~4.8 h   takes every Experiment 2 claim to n=3
#   4  two three-objective seeds   ~5.5 h   takes the n=1 post-hoc hierarchy to n=3
#
# The maze11 bridge was removed on request. It fine-tuned the one agent that has
# a value channel, as a control on whether an agent that can *read* what an
# objective is worth bothers to compile it into its weights. Every value-axis
# result is and remains on four-channel agents.

set -euo pipefail
export PATH="${HOME}/.local/bin:${PATH}"
cd "$(dirname "$0")/.."

DATA="${1:-/workspace/data}"
: "${ARM_STEPS:?set ARM_STEPS to the arm length the low-end curve picked}"
CHECKPOINTS="${CHECKPOINTS:-4}"
LOGS="${DATA}/logs"
mkdir -p "${LOGS}"

# A stage that fails must not take the stages after it with it. The maze11
# bridge died on every attempt because 013 assumed every agent had no value
# channel, and `set -e` meant the third seed and both three-objective seeds --
# the bulk of the campaign -- never started at all.
FAILED_STAGES=""

sweep() {  # agent [extra args...]
    local agent="$1"; shift
    echo "=== sweep ${agent} at ${ARM_STEPS} steps ==="
    if ! uv run python scripts/value_axis_sweep.py --data "${DATA}" --agent "${agent}" \
        --steps "${ARM_STEPS}" --checkpoints "${CHECKPOINTS}" "$@" \
        >> "${LOGS}/sweep-${agent}.log" 2>&1; then
        echo "  !! sweep ${agent} FAILED -- see ${LOGS}/sweep-${agent}.log"
        FAILED_STAGES="${FAILED_STAGES} sweep:${agent}"
        return 0
    fi
}

train_base() {  # tag steps values... ; trains only if not already finished
    local tag="$1" steps="$2"; shift 2
    # Finished means a saved checkpoint, not a directory. cleanba creates
    # local-files as soon as it starts, so a run killed before its first save
    # leaves an empty one -- and checking for the directory declared it done,
    # skipped it on every retry, and left the sweep failing forty times on a base
    # that did not exist. Same mistake as judging an arm by a file existing.
    if [ -n "$(ls "${DATA}/runs/${tag}/local-files" 2>/dev/null | grep '^cp_')" ]; then
        echo "  ${tag} present"; return
    fi
    if [ -d "${DATA}/runs/${tag}" ]; then
        echo "  ${tag} has no saved checkpoint, restarting it"
        rm -rf "${DATA}/runs/${tag}"
    fi
    local n=$#; local levels="${DATA}/levels/values/$(uv run python -c "
from goalmisgen.volume import values_tag
print(values_tag([$(echo "$@" | tr ' ' ',')]))")@1M"
    if [ ! -d "${levels}" ]; then
        echo "=== levels for ${tag} ==="
        uv run python scripts/generate_levels.py --n-levels 1000000 --min-size 11 --max-size 11 \
            --valid-levels 50000 --test-levels 50000 --n-objectives "${n}" --objective-values "$@" \
            --out "${levels}" >> "${LOGS}/generate.log" 2>&1
    fi
    echo "=== train ${tag}, ${steps} steps ==="
    uv run python experiments/001_maze_repro.py \
        --levels "${levels}" --total-timesteps "${steps}" \
        --min-size 11 --max-size 11 --n-objectives "${n}" --objective-values "$@" \
        --hide-values --seed "${SEED:-1234}" \
        --run-dir "${DATA}/runs/${tag}" \
        --note "${NOTE:-Campaign base agent ${tag}.}" >> "${LOGS}/train-${tag}.log" 2>&1 || {
            echo "  !! training ${tag} FAILED -- see ${LOGS}/train-${tag}.log"
            FAILED_STAGES="${FAILED_STAGES} train:${tag}"; return 0; }
    uv run python - "${DATA}/runs/${tag}" <<'PY'
import json, sys
from pathlib import Path
agent = Path(sys.argv[1])
checkpoints = sorted((agent / "local-files").glob("cp_*"))
cfg = json.loads((checkpoints[-1] / "cfg.json").read_text())
c = cfg.get("cfg", cfg)
(agent / "BASE.json").write_text(json.dumps({
    "checkpoint": str(checkpoints[-1].relative_to(agent)),
    "values": list(c["train_env"]["objective_values"]),
    "objectives": c["train_env"]["n_objectives"],
    "steps": c["total_timesteps"],
    "checkpoints_saved": len(checkpoints),
}, indent=2) + "\n")
PY
}

point_at() {  # tag source_agent checkpoint_name -- an agent view of an earlier checkpoint
    local tag="$1" source="$2" checkpoint="$3"
    local agent="${DATA}/runs/${tag}"
    [ -e "${agent}/BASE.json" ] && { echo "  ${tag} present"; return; }
    mkdir -p "${agent}"
    ln -sfn "../${source}/local-files" "${agent}/local-files"
    uv run python - "${agent}" "${checkpoint}" <<'PY'
import json, sys
from pathlib import Path
agent, checkpoint = Path(sys.argv[1]), sys.argv[2]
cfg = json.loads((agent / "local-files" / checkpoint / "cfg.json").read_text())
c = cfg.get("cfg", cfg)
(agent / "BASE.json").write_text(json.dumps({
    "checkpoint": f"local-files/{checkpoint}",
    "values": list(c["train_env"]["objective_values"]),
    "objectives": c["train_env"]["n_objectives"],
    "steps": c["total_timesteps"],
    "checkpoints_saved": 1,
}, indent=2) + "\n")
PY
}

echo "############ 1. seed 5678's wide sweep ############"
sweep novalue11.s5678

echo "############ 2. base-checkpoint ladder ############"
# Arms fitted from earlier checkpoints of the same agent, to ask whether the
# axis direction is settled long before base training ends. This is what the
# 250M extension was meant to be gated on.
for cp in cp_070103040 cp_100146560; do
    if [ -d "${DATA}/runs/novalue11.s1234/local-files/${cp}" ]; then
        tag="novalue11.s1234.at${cp#cp_0}"
        point_at "${tag}" novalue11.s1234 "${cp}"
        sweep "${tag}" --objectives 1
    else
        echo "  ${cp} not saved, skipping that rung"
    fi
done

echo "############ 3. third two-objective seed ############"
SEED=9012 NOTE="A third seed of novalue11, identical but for the seed. Takes the one-knob, exchange-rate and channel-localisation claims to n=3, which is the smallest number that supports an interval." \
    train_base novalue11.s9012 150000000 1.0 0.5
sweep novalue11.s9012

echo "############ 4. two more three-objective seeds ############"
# The grid has to be the one being replicated, not the current default. The
# original threeobj_v2 swept objectives 0 and 1 at +/-0.2 and +/-0.4 and
# objective 2 at +/-0.15 and +/-0.3 -- narrower because 0.4 sits close to its
# neighbour. Those offsets cross rank boundaries on purpose, which is what makes
# a second dimension necessary and is why --allow-reorder is passed here and
# nowhere else.
for seed in 5678 9012; do
    SEED="${seed}" NOTE="A further seed of the uneven three-objective task, to test the hierarchical account out of sample against predictions registered before launch." \
        train_base "threeobj.uneven.s${seed}" 80000000 1.0 0.55 0.4
    sweep "threeobj.uneven.s${seed}" --objectives 0 1 --offsets 0.2 0.4 --allow-reorder
    sweep "threeobj.uneven.s${seed}" --objectives 2 --offsets 0.15 0.3 --allow-reorder
done

echo
if [ -n "${FAILED_STAGES}" ]; then
    echo "CAMPAIGN_INCOMPLETE -- failed:${FAILED_STAGES}"
    exit 1
fi
echo "CAMPAIGN_COMPLETE"
