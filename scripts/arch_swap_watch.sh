#!/usr/bin/env bash
# Measure each architecture-swap agent as soon as its training finishes.
#
#   bash scripts/arch_swap_watch.sh resnet [DATA_DIR]
#
# Companion to arch_swap.sh, meant to sit in its own tmux session on the pod
# that is training the pair: it waits for RUN COMPLETE in each run's log and
# then runs arch_swap_measure.sh on it, so the measurements do not wait for
# anyone to notice the training ended.

set -euo pipefail
cd "$(dirname "$0")/.."
NET="${1:?net: resnet or vit}"
DATA="${2:-/workspace/data}"
LOGS="${DATA}/logs/arch-swap"

for name in "${NET}11.s1234" "${NET}11clean.s1234"; do
    until grep -qs "RUN COMPLETE" "${LOGS}/train-${name}.log"; do sleep 60; done
    echo "=== ${name} finished training; measuring  $(date -u +%FT%TZ) ==="
    bash scripts/arch_swap_measure.sh "${name}" "${DATA}"
done
echo "MEASURED ${NET}"
