#!/usr/bin/env bash
set -uo pipefail
cd /workspace/GoalMisgeneralisation
export PATH="$HOME/.local/bin:$PATH"
D=/workspace/data/offline/demos/scaling
mkdir -p figures/data/scaling
uv run python - > results/drc-param-count.txt 2>&1 <<'PY'
import jax, json
from cleanba.cleanba_impala import load_train_state
from goalmisgen.configs.env import MazeConfig
from goalmisgen.offline.model import parameter_count
from pathlib import Path
run = Path("/workspace/data/runs/novalue11.s1234")
payload = json.loads((run / "BASE.json").read_text())
config = MazeConfig(max_episode_steps=120, num_envs=2, min_size=11, max_size=11, n_objectives=2,
                    objective_values=tuple(payload["values"]), feature_value_correlation=1.0,
                    value_encoding="none", colour_is_the_only_value_cue=True,
                    level_dataset="/workspace/data/levels/values/1.00-0.50@1M", dataset_split="test",
                    asynchronous=False, seed=0)
_, _, _, ts, _ = load_train_state(run / payload["checkpoint"], env_cfg=config)
print("novalue11 DRC params:", parameter_count(ts.params))
PY
for shape in d128l4 d256l4 d512l4 d512l16; do
    run=/workspace/data/offline/runs/scaling/sc11.$shape.s1
    uv run python scripts/decode_h1.py $run 20000 figures/data/scaling/$shape.npz --demos $D/test.rho100 \
        > results/decode-sc11.$shape.txt 2>&1 || { echo "$shape decode FAILED" >> results/scale-sweep.status; continue; }
    uv run python scripts/distance_field_bc.py $run --probe-demos $D/train.rho100 --eval-demos $D/test.rho100 \
        > results/distance-field-sc11.$shape.txt 2> results/distance-field-sc11.$shape.log \
        && echo "$shape ok" >> results/scale-sweep.status \
        || echo "$shape fields FAILED" >> results/scale-sweep.status
done
echo done >> results/scale-sweep.status
