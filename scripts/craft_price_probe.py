"""What is wood worth to a crafting model? Counterfactual starting inventories, closed-loop.

    uv run python scripts/craft_price_probe.py RUN_DIR DEMOS [--levels 512] [--checkpoint STEP] [--json out.json]

Wood, stone and the pickaxes have no value of their own in the task: they are
worth what they unlock. So the model's valuation of them is read the way the
ore values are read - as an exchange rate - by starting episodes from
counterfactual states and asking which ore the model goes for. For each
starting inventory the script rolls the model out closed-loop from that
state, scores it against the expert's plan *from that state*, and reports
the fraction of fields on which the model and the expert take iron, the
agreement between them, and the model's indifference point in the cost gap
of the two plans from that state. Run on a base and on its value arms, the
same table says whether raising iron's value raises the worth of the stone
and the fourth tree (the value propagates) or only of the iron tile.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from goalmisgen.analysis.behaviour import indifference_point, value_distance_decisions
from goalmisgen.craftax import closed_loop, engine, simulate
from goalmisgen.craftax.craft import CraftDemoSet
from goalmisgen.offline.demonstrations import load_demonstrations
from goalmisgen.offline.train import list_checkpoints, load_checkpoint

CONDITIONS: dict[str, tuple[int, int, int, int]] = {
    "empty-handed": (0, 0, 0, 0),
    "1 wood": (1, 0, 0, 0),
    "2 wood": (2, 0, 0, 0),
    "3 wood": (3, 0, 0, 0),
    "wood pickaxe": (0, 0, 1, 0),
    "wood pickaxe + 1 wood": (1, 0, 1, 0),
    "wood pickaxe + 1 stone": (0, 1, 1, 0),
    "wood pickaxe + wood + stone": (1, 1, 1, 0),
    "stone pickaxe": (0, 0, 1, 1),
}
"""(wood, stone, wood pickaxe, stone pickaxe) the episode starts with."""


def run_condition(model, params, demos: CraftDemoSet, indices: np.ndarray, inventory) -> dict:
    task = demos.task
    fields = [demos.level(int(i)) for i in indices]
    starts = [simulate.State(task.tiles(f), f.agent_start, task.start_direction, tuple(inventory)) for f in fields]
    step_penalty, step_limit = float(demos.meta["step_penalty"]), int(demos.meta["step_limit"])
    solutions = [task.solution_from(s, f, step_penalty, step_limit) for s, f in zip(starts, fields)]
    decoded = closed_loop.rollout(model, params, demos, indices, starts=starts)
    outcomes = engine.replay_batch(
        fields,
        task,
        decoded.actions,
        step_penalty,
        step_limit,
        [bool(e) for e in decoded.emitted_eos],
        states=engine.stack_states([engine.state_from_simulated(s) for s in starts]),
        solutions=solutions,
    )
    reached = np.array([o["reached_objective"] for o in outcomes])
    took_iron = np.array([o.get("reached_feature_id") == 0 for o in outcomes])  # feature 0 is iron
    expert_iron = np.array(
        [
            sol.optimal_index == [k for k, o in enumerate(f.objectives) if o.feature_id == 0][0]
            for sol, f in zip(solutions, fields)
        ]
    )
    agree = np.array([o["chose_optimal"] for o in outcomes if o["reached_objective"]])
    gaps, richer, _ = value_distance_decisions(outcomes)
    result = {
        "reached": float(reached.mean()),
        "model_takes_iron": float(took_iron[reached].mean()) if reached.any() else float("nan"),
        "expert_takes_iron": float(expert_iron.mean()),
        "agreement_when_reached": float(agree.mean()) if len(agree) else float("nan"),
        "indifference_model": float(indifference_point(gaps, richer)) if len(gaps) > 8 else float("nan"),
        "mean_cost_iron_plan": float(
            np.mean(
                [
                    s.distances[[k for k, o in enumerate(f.objectives) if o.feature_id == 0][0]] or 0
                    for s, f in zip(solutions, fields)
                ]
            )
        ),
        "mean_cost_coal_plan": float(
            np.mean(
                [
                    s.distances[[k for k, o in enumerate(f.objectives) if o.feature_id == 1][0]] or 0
                    for s, f in zip(solutions, fields)
                ]
            )
        ),
        "n": int(len(indices)),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    parser.add_argument("demos", type=Path)
    parser.add_argument("--levels", type=int, default=512)
    parser.add_argument("--checkpoint", type=int, default=None)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()
    checkpoints = list_checkpoints(args.run)
    step, directory = checkpoints[-1] if args.checkpoint is None else next(c for c in checkpoints if c[0] == args.checkpoint)
    model, params = load_checkpoint(directory)
    demos = load_demonstrations(args.demos, hide_values=True)
    assert isinstance(demos, CraftDemoSet) and demos.task.receding
    indices = np.arange(min(args.levels, len(demos)))
    print(
        f"{args.run.name} @ step {step}, {len(indices)} fields, values {demos.meta['values']} (threshold {(demos.meta['values'][0] - demos.meta['values'][1]) / demos.meta['step_penalty']:.0f} actions)\n"
    )
    print(
        f"{'start with':30s} {'reached':>8s} {'iron: model':>12s} {'expert':>7s} {'agree':>6s} {'indiff':>7s} {'cost iron':>10s} {'coal':>6s}"
    )
    results = {}
    for name, inventory in CONDITIONS.items():
        r = run_condition(model, params, demos, indices, inventory)
        results[name] = r
        print(
            f"{name:30s} {r['reached']:8.2f} {r['model_takes_iron']:12.2f} {r['expert_takes_iron']:7.2f} {r['agreement_when_reached']:6.2f} "
            f"{r['indifference_model']:7.1f} {r['mean_cost_iron_plan']:10.1f} {r['mean_cost_coal_plan']:6.1f}"
        )
    if args.json:
        args.json.write_text(json.dumps({"run": str(args.run), "step": step, "conditions": results}, indent=2))


if __name__ == "__main__":
    main()
