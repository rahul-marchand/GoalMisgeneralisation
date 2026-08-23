#!/usr/bin/env bash
# The width/depth scaling campaign, as run on the pod. Registered in
# Preregistration-scaling.md; read that first, it is what these numbers mean.
#
#   bash scripts/scaling_pod.sh plan                 # what will be generated and trained, and how big
#   bash scripts/scaling_pod.sh levels               # the base level pool          (CPU, hours)
#   bash scripts/scaling_pod.sh demos                # expert routes over it        (CPU, hours)
#   bash scripts/scaling_pod.sh armlevels            # 24 pools at the arms' values (CPU)
#   bash scripts/scaling_pod.sh armdemos             # expert routes over those     (CPU)
#   bash scripts/scaling_pod.sh bases                # the nine shapes, queued      (GPU)
#   bash scripts/scaling_pod.sh calibrate SHAPE      # the arm learning rate for one cell
#   bash scripts/scaling_pod.sh arms SHAPE           # that cell's 26 arms
#   bash scripts/scaling_pod.sh analysis SHAPE       # 027, 028 and 029 on that cell
#   bash scripts/scaling_pod.sh chain                # all of the above, unattended and resumable
#
# Every stage skips what is already on the volume, so re-running resumes. The
# unit of "already done" is a `done.json` for a run and a `meta.json` for a
# dataset, the same rule the offline-BC campaign used.
#
# THE DATA IS SINGLE-EPOCH BY CONSTRUCTION. A base sees 30,720,000
# demonstrations in 30,000 steps of batch 1024, and the pool holds exactly
# 30,720,000 training levels; an arm sees 256,000 in 1,000 steps of batch 256
# against a pool of 310,000. Nothing here repeats a maze, so no result of this
# campaign can be explained by memorisation. Do not "save time" by shrinking a
# pool -- that is the one confound the design removes by construction.

set -uo pipefail
export PATH="${HOME}/.local/bin:${PATH}"
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.45}"

cd "$(dirname "$0")/.."

DATA="${DATA:-/workspace/data}"
LEVELS="${DATA}/levels/values"
DEMOS="${DATA}/offline/demos/scaling"
RUNS="${DATA}/offline/runs/scaling"
RESULTS="${DATA}/offline/results/scaling"
LOGS="${DATA}/logs/scaling"

# --- the registered protocol. Changing any of these forks the campaign. -------
BASE_STEPS="${BASE_STEPS:-30000}"
BASE_BATCH="${BASE_BATCH:-256}"
BASE_LR="${BASE_LR:-3e-4}"
BASE_WARMUP="${BASE_WARMUP:-500}"
FT_STEPS="${FT_STEPS:-1000}"
FT_BATCH="${FT_BATCH:-256}"
FT_WARMUP="${FT_WARMUP:-50}"
CALIBRATION_LRS="${CALIBRATION_LRS:-3e-5 1e-4 3e-4}"
# Set to a rate to skip calibration entirely and use that rate for every arm.
# The width/depth campaign calibrated per cell to remove the confound that one
# fixed rate is a different-sized step at different widths, and the calibrated
# rate then drove its headline metric at r = -0.856. A sweep at FIXED width has
# no such confound to remove, so the honest move is one rate, chosen once.
FIXED_ARM_LR="${FIXED_ARM_LR:-}"
ARM_OBJECTIVES="${ARM_OBJECTIVES:-0 1}"
ARM_OFFSETS="${ARM_OFFSETS:-}"          # empty = the registered six magnitudes
# Which slice of a cell's arms this worker takes. The arms of one cell are
# independent runs writing to disjoint directories, so a fleet can share them
# out; without this the largest cell's 26 arms queue behind its own base on one
# card and set the makespan for everything.
ARM_SHARD_INDEX="${ARM_SHARD_INDEX:-0}"
ARM_SHARD_COUNT="${ARM_SHARD_COUNT:-1}"
SEED="${SEED:-1}"
EVAL_LEVELS="${EVAL_LEVELS:-1024}"   # levels decoded per evaluation set, per checkpoint

TRAIN_LEVELS=$((BASE_STEPS * BASE_BATCH))          # 30,720,000 -- one epoch, exactly
VALID_LEVELS="${VALID_LEVELS:-400000}"             # the null arm fine-tunes here; also single-epoch
TEST_LEVELS="${TEST_LEVELS:-50000}"                # the shared held-out set 027/029 measure on
# Splits are assigned by hashing each maze, so the achieved train count is a
# binomial draw around the request rather than exact. The pool carries 2% of
# headroom so that a short draw still covers a single epoch; a run that saw the
# same demonstration twice would void the whole premise.
POOL_LEVELS=$(( (TRAIN_LEVELS + VALID_LEVELS + TEST_LEVELS) * 102 / 100 ))
ARM_TRAIN_LEVELS="${ARM_TRAIN_LEVELS:-310000}"     # > FT_STEPS * FT_BATCH = 256,000
ARM_HOLDOUT="${ARM_HOLDOUT:-5000}"
ARM_POOL_LEVELS=$((ARM_TRAIN_LEVELS + 2 * ARM_HOLDOUT))
# Pools are named by their exact level count rather than a rounded "31M". Two
# pools of the same values and different sizes must not collide on a name, and
# the rounded form does exactly that at the sizes this campaign uses.
BASE_POOL="${LEVELS}/1.00-0.50@${POOL_LEVELS}"
arm_pool() { echo "${LEVELS}/$1@${ARM_POOL_LEVELS}"; }

# --- the grid: three widths by three depths, d_model/n_heads fixed at 32 ------
# name         d_model  layers  heads
SHAPES="${SHAPES:-
d128l4    128   4   4
d256l4    256   4   8
d512l4    512   4  16
d128l8    128   8   4
d256l8    256   8   8
d512l8    512   8  16
d128l16   128  16   4
d256l16   256  16   8
d512l16   512  16  16
}"

arm_grid() {
    uv run python scripts/scaling_arms.py --steps "${FT_STEPS}" --objective ${ARM_OBJECTIVES} \
        ${ARM_OFFSETS:+--offsets ${ARM_OFFSETS}} "$@"
}

stamp() { date -u +%FT%TZ; }
say() { echo "$(stamp) $*"; }
shape_row() { echo "${SHAPES}" | awk -v n="$1" '$1==n {print; exit}'; }
run_dir() { echo "${RUNS}/sc11.$1.s${SEED}"; }
last_checkpoint() { ls -d "$1"/checkpoints/step_* 2>/dev/null | sort | tail -n1; }

retry() {  # retry CMD... up to 5 times, a minute apart; a shared card can fail to init
    local n=0
    until "$@"; do
        n=$((n + 1))
        [ "${n}" -ge 5 ] && { say "FAILED after ${n} tries: $*"; return 1; }
        say "retry ${n}: $*"; sleep 60
    done
}

mkdirs() { mkdir -p "${DEMOS}/arms" "${RUNS}" "${RESULTS}" "${LOGS}" "${LEVELS}"; }

# --- what this will cost ------------------------------------------------------
plan() {
    echo "grid"
    printf '  %-10s %8s %7s %7s %14s\n' name d_model layers heads parameters
    echo "${SHAPES}" | while read -r name d l h; do
        [ -z "${name}" ] && continue
        printf '  %-10s %8s %7s %7s %14s\n' "${name}" "${d}" "${l}" "${h}" \
            "$(python3 -c "print(f'{12*${d}*${d}*${l} + 99*${d}:,}')")"
    done
    echo
    echo "data (single epoch everywhere)"
    printf '  %-28s %14s\n' "base pool" "$(printf "%'d" ${POOL_LEVELS})"
    printf '  %-28s %14s\n' "  of which train" "$(printf "%'d" ${TRAIN_LEVELS})"
    printf '  %-28s %14s\n' "  of which valid (null arm)" "$(printf "%'d" ${VALID_LEVELS})"
    printf '  %-28s %14s\n' "  of which test (measured on)" "$(printf "%'d" ${TEST_LEVELS})"
    local tags; tags=$(arm_grid | awk '$2!="+0.00" {print $5}' | sort -u | wc -l)
    printf '  %-28s %14s\n' "arm pools (${tags} values)" "$(printf "%'d" $((tags * ARM_POOL_LEVELS)))"
    printf '  %-28s %14s\n' "levels in total" "$(printf "%'d" $((POOL_LEVELS + tags * ARM_POOL_LEVELS)))"
    echo
    echo "training"
    printf '  %-28s %s\n' "base" "${BASE_STEPS} steps x batch ${BASE_BATCH}, lr ${BASE_LR}, warmup ${BASE_WARMUP}"
    printf '  %-28s %s\n' "arm" "${FT_STEPS} steps x batch ${FT_BATCH}, constant lr, warmup ${FT_WARMUP}"
    if [ -n "${FIXED_ARM_LR}" ]; then
        printf '  %-28s %s\n' "arm learning rate" "${FIXED_ARM_LR} (fixed, no calibration)"
    else
        printf '  %-28s %s\n' "calibration ladder" "${CALIBRATION_LRS}"
    fi
    printf '  %-28s %s\n' "arms per shape" "$(arm_grid | wc -l)"
}

# --- 1. levels ----------------------------------------------------------------
levels() {
    mkdirs
    if [ -f "${BASE_POOL}/meta.json" ]; then say "have ${BASE_POOL}"; return 0; fi
    say "generating ${POOL_LEVELS} levels -> ${BASE_POOL}"
    uv run python scripts/generate_levels.py \
        --n-levels "${POOL_LEVELS}" --valid-levels "${VALID_LEVELS}" --test-levels "${TEST_LEVELS}" \
        --min-size 11 --max-size 11 --objective-values 1.0 0.5 --out "${BASE_POOL}"
}

# --- 2. demonstrations over them ---------------------------------------------
demos() {
    mkdirs
    # split rho cap: the training set is the whole train split; the misgeneralisation
    # curves need only enough valid levels to measure on.
    for spec in "train 1.0 0" "valid 1.0 0" "test 1.0 0" "valid 0.5 50000" "valid 0.0 50000"; do
        set -- ${spec}
        local split="$1" rho="$2" cap="$3"
        local tag; tag=$(printf "rho%03d" "$(python3 -c "print(int(round(${rho} * 100)))")")
        local out="${DEMOS}/${split}.${tag}"
        if [ -f "${out}/meta.json" ]; then say "have ${out}"; continue; fi
        say "demonstrating ${split} at rho=${rho} -> ${out}"
        uv run python scripts/generate_demos.py --levels "${BASE_POOL}" --split "${split}" --rho "${rho}" \
            --min-size 11 --max-size 11 --objective-values 1.0 0.5 \
            $([ "${cap}" != "0" ] && echo --n "${cap}") --out "${out}"
    done
}

# --- 3. one level pool per arm value -----------------------------------------
armlevels() {
    mkdirs
    arm_grid | awk '$2!="+0.00" {print $5}' | sort -u |
    while read -r tag; do
        local out; out=$(arm_pool "${tag}")
        if [ -f "${out}/meta.json" ]; then say "have ${out}"; continue; fi
        say "generating ${ARM_POOL_LEVELS} levels at values ${tag//-/ }"
        uv run python scripts/generate_levels.py \
            --n-levels "${ARM_POOL_LEVELS}" --valid-levels "${ARM_HOLDOUT}" --test-levels "${ARM_HOLDOUT}" \
            --min-size 11 --max-size 11 --objective-values ${tag//-/ } --out "${out}"
    done
}

# --- 4. demonstrations at each arm's values ----------------------------------
armdemos() {
    mkdirs
    arm_grid | awk '$2!="+0.00" {print $5}' | sort -u |
    while read -r tag; do
        local src; src=$(arm_pool "${tag}")
        [ -d "${src}" ] || { say "MISSING ${src}; run armlevels first"; continue; }
        for split in train test; do
            local out="${DEMOS}/arms/${tag}.${split}.rho100"
            if [ -f "${out}/meta.json" ]; then say "have ${out}"; continue; fi
            uv run python scripts/generate_demos.py --levels "${src}" --split "${split}" --rho 1.0 \
                --min-size 11 --max-size 11 --objective-values ${tag//-/ } --out "${out}"
        done
    done
}

# --- 5. the nine bases --------------------------------------------------------
base() {
    local name="$1"
    set -- $(shape_row "${name}")
    [ -n "${1:-}" ] || { say "no such shape ${name}"; return 1; }
    local d="$2" l="$3" h="$4"
    local out; out=$(run_dir "${name}")
    if [ -f "${out}/done.json" ]; then say "have ${name}"; return 0; fi
    rm -rf "${out}"
    say "training ${name} (d_model ${d}, ${l} layers, ${h} heads)"
    uv run python experiments/023_train_bc.py \
        --demos "${DEMOS}/train.rho100" --hide-values \
        --eval "rho100=${DEMOS}/valid.rho100" "rho050=${DEMOS}/valid.rho050" "rho000=${DEMOS}/valid.rho000" \
        --eval-levels "${EVAL_LEVELS}" \
        --out "${out}" --seed "${SEED}" \
        --steps "${BASE_STEPS}" --batch-size "${BASE_BATCH}" --lr "${BASE_LR}" --warmup "${BASE_WARMUP}" \
        --d-model "${d}" --layers "${l}" --heads "${h}" \
        --note "Width/depth scaling campaign, cell ${name}: hidden-value route model, d_model ${d}, ${l} layers, single-epoch data. Preregistration-scaling.md." \
        > "${LOGS}/${name}.log" 2>&1
}

bases() { echo "${SHAPES}" | while read -r name _ _ _; do [ -z "${name}" ] || base "${name}"; done; }

# --- 6. the arm learning rate for one cell -----------------------------------
# The widest positive arm at each rate on the ladder; the rate that lands closest
# to that arm's expert exchange rate is used for all of the cell's arms. Why the
# budget is fixed and only the rate is calibrated: the 2026-08-22 amendment.
calibrate() {
    local name="$1"
    local out; out=$(run_dir "${name}")
    local file="${out}/arm_lr.txt"
    if [ -f "${file}" ]; then say "have arm lr for ${name}: $(cat "${file}")"; return 0; fi
    [ -f "${out}/done.json" ] || { say "${name} has no finished base yet"; return 1; }
    if [ -n "${FIXED_ARM_LR}" ]; then
        mkdir -p "${out}"; echo "${FIXED_ARM_LR}" > "${file}"
        say "arm lr for ${name}: ${FIXED_ARM_LR} (fixed, not calibrated)"
        return 0
    fi
    read -r sweep offset seed dirname tag target < <(arm_grid --widest-only | head -1)
    local init; init=$(last_checkpoint "${out}")
    local dirs=""
    for lr in ${CALIBRATION_LRS}; do
        local cal="${out}/calibration/lr${lr}"
        dirs="${dirs} ${cal}"
        [ -f "${cal}/done.json" ] && continue
        say "calibrating ${name} at lr ${lr} on arm ${dirname} (target ${target})"
        uv run python experiments/023_train_bc.py \
            --demos "${DEMOS}/arms/${tag}.train.rho100" --init-from "${init}" --schedule constant \
            --eval "base=${DEMOS}/test.rho100" --eval-levels "${EVAL_LEVELS}" \
            --out "${cal}" --seed "${seed}" --steps "${FT_STEPS}" --batch-size "${FT_BATCH}" \
            --lr "${lr}" --warmup "${FT_WARMUP}" --checkpoint-first 100000000 \
            --note "Arm learning-rate calibration for ${name}: the widest positive arm at lr ${lr}." \
            > "${LOGS}/${name}.calibrate.lr${lr}.log" 2>&1 || say "calibration ${name} lr ${lr} FAILED"
    done
    uv run python scripts/pick_arm_lr.py --candidates ${dirs} --target "${target}" > "${file}" \
        2> "${LOGS}/${name}.calibrate.log" || { rm -f "${file}"; say "picking a rate for ${name} FAILED"; return 1; }
    say "arm lr for ${name}: $(cat "${file}")  (ladder in ${LOGS}/${name}.calibrate.log)"
}

# --- 7. that cell's arms ------------------------------------------------------
arms() {
    local name="$1"
    local out; out=$(run_dir "${name}")
    # Exactly one worker calibrates. The rate is one scalar per cell and every
    # arm of that cell must use the same one, so a fleet cannot have each shard
    # picking its own -- and three workers training the same calibration arm into
    # the same directory would corrupt it rather than merely waste the card.
    if [ "${ARM_SHARD_INDEX}" = "0" ]; then
        calibrate "${name}" || return 1
    else
        say "waiting for shard 0 to choose the arm learning rate for ${name}"
        local waited=0
        until [ -f "${out}/arm_lr.txt" ]; do
            sleep 20; waited=$((waited + 20))
            [ "${waited}" -gt 3600 ] && { say "gave up waiting for ${name}/arm_lr.txt"; return 1; }
        done
    fi
    local lr; lr=$(cat "${out}/arm_lr.txt")
    local init; init=$(last_checkpoint "${out}")
    local index=-1
    arm_grid | while read -r sweep offset seed dirname tag target; do
        index=$((index + 1))
        [ $((index % ARM_SHARD_COUNT)) -eq "${ARM_SHARD_INDEX}" ] || continue
        local arm="${out}/arms/${dirname}"
        if [ -f "${arm}/done.json" ]; then say "have ${name}/${dirname}"; continue; fi
        # The null arm's values are the base's own, so its fine-tuning data comes
        # from the base pool's valid split -- 400k levels the base never trained
        # on and disjoint from test, which keeps it single-epoch like every other
        # arm and keeps 023's overlap check happy.
        local train="${DEMOS}/arms/${tag}.train.rho100" own="${DEMOS}/arms/${tag}.test.rho100"
        if [ "${offset}" = "+0.00" ]; then train="${DEMOS}/valid.rho100"; own="${DEMOS}/test.rho100"; fi
        say "arm ${name}/${dirname} at lr ${lr}"
        uv run python experiments/023_train_bc.py \
            --demos "${train}" --init-from "${init}" --schedule constant \
            --eval "base=${DEMOS}/test.rho100" "own=${own}" --eval-levels "${EVAL_LEVELS}" \
            --out "${arm}" --seed "${seed}" --steps "${FT_STEPS}" --batch-size "${FT_BATCH}" \
            --lr "${lr}" --warmup "${FT_WARMUP}" --checkpoint-first 100000000 \
            --note "Value-axis arm ${dirname} of ${name}: ${FT_STEPS} steps at constant lr ${lr} on rho=1.0 hidden-value demonstrations at values ${tag}. Expert exchange rate ${target} steps." \
            > "${LOGS}/${name}.${dirname}.log" 2>&1 || say "${name}/${dirname} FAILED"
    done
}

# --- 8. what it all meant -----------------------------------------------------
analysis() {
    local name="$1"
    local out; out=$(run_dir "${name}")
    mkdir -p "${RESULTS}"
    for objective in ${ARM_OBJECTIVES}; do
        for script in 027_bc_value_axis 029_axis_shape; do
            local stem="${RESULTS}/${script%%_*}.${name}.o${objective}"
            uv run python "experiments/${script}.py" "${out}" --sweep "o${objective}" --steps "${FT_STEPS}" \
                --demos "${DEMOS}/test.rho100" --json "${stem}.json" > "${stem}.txt" 2> "${stem}.err" \
                || say "${script} ${name} o${objective} FAILED (see ${stem}.err)"
        done
    done
    # 028 is cos(axis_0, axis_1) and so needs both sweeps; a single-sweep run has
    # no such question to ask, and calling it anyway only produces an error file.
    if [ "$(echo ${ARM_OBJECTIVES} | wc -w)" -lt 2 ]; then
        say "only sweep(s) ${ARM_OBJECTIVES} were run, so 028 has no second axis to compare against"
        return 0
    fi
    uv run python experiments/028_bc_value_or_gap.py "${out}" --steps "${FT_STEPS}" \
        --json "${RESULTS}/028.${name}.json" > "${RESULTS}/028.${name}.txt" 2> "${RESULTS}/028.${name}.err" \
        || say "028 ${name} FAILED"
}

# --- everything, unattended ---------------------------------------------------
chain() {
    mkdirs
    say "stage levels";    retry levels
    say "stage demos";     retry demos
    say "stage armlevels"; armlevels
    say "stage armdemos";  armdemos
    echo "${SHAPES}" | while read -r name _ _ _; do
        [ -z "${name}" ] && continue
        say "stage base ${name}";     base "${name}"     || say "base ${name} FAILED, skipping its arms"; 
        [ -f "$(run_dir "${name}")/done.json" ] || continue
        say "stage arms ${name}";     arms "${name}"
        say "stage analysis ${name}"; analysis "${name}"
    done
    say "CHAIN DONE"
}

# ---- fleet stages -------------------------------------------------------------
# `data` is run once, by one worker, because the pools are shared and two
# workers generating the same dataset would race on the same directory. `work`
# is what every worker then runs over its own share of the grid.
data() { mkdirs; levels; demos; armlevels; armdemos; say "DATA DONE"; }

work() {
    echo "${SHAPES}" | while read -r name _ _ _; do
        [ -z "${name}" ] && continue
        say "base ${name}";  base "${name}" || { say "base ${name} FAILED"; continue; }
        say "arms ${name} (shard ${ARM_SHARD_INDEX}/${ARM_SHARD_COUNT})"; arms "${name}"
    done
    say "WORK DONE"
}

case "${1:-}" in
    plan) plan ;;
    data) data ;;
    work) work ;;
    levels) levels ;;
    demos) demos ;;
    armlevels) armlevels ;;
    armdemos) armdemos ;;
    base) base "$2" ;;
    bases) bases ;;
    calibrate) calibrate "$2" ;;
    arms) arms "$2" ;;
    analysis) analysis "$2" ;;
    chain) chain ;;
    *) sed -n '2,26p' "$0" >&2; exit 2 ;;
esac
