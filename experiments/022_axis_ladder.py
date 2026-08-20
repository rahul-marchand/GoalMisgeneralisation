"""When in training does the value axis appear, and when does it become one axis?

    uv run python experiments/022_axis_ladder.py \
        --data /workspace/data --agent novalue11.s1234 \
        --levels /workspace/data/levels/values/1.00-0.50@150k --arm-steps 400000

``014`` fits one axis, at the end of training. ``015`` asks whether the two
objectives' axes are one knob or two registers, also at the end of training. Both
therefore describe a finished agent, and neither can say whether the structure
they find was built early and left alone, or arrived late, or was assembled out
of something else.

This runs the same fits at every rung of the base-checkpoint ladder — the same
agent, swept again from earlier points in its own training — and puts the answers
side by side. Three questions, in the order they can be answered:

``present``     is there an axis at all? ``|axis|`` per unit of value, against
                the drift every arm carries whether or not there is anything to
                learn. An axis buried under drift is not yet an axis.
``settled``     does it point where it will end up pointing?
                ``cos(axis@t, axis@end)``, per objective.
``one knob``    is it one axis or two? ``cos(axis_0, axis_1)`` at each rung, with
                the second-dimension share beside it. Two independent value
                registers give a cosine near zero and half the variance in the
                second dimension; one threshold on the difference gives -1 and
                none. Watching that collapse happen -- if it does -- is the point
                of sweeping both objectives at every rung rather than one.

Weights only: no rollouts, so this is cheap and runs anywhere. What it cannot say
is whether an axis that exists is *writable*, which is ``014``'s held-out write
and needs a GPU and episodes. An axis can be well-determined in the weights and
still not move behaviour, so "appears" here means "is there in the weights", and
the behavioural rung is a separate measurement.

Every cosine is read against a permutation null rather than against zero. Arms at
a rung share a large common component -- the cost of running the updates -- so two
axes fitted from two sweeps of the same agent are correlated whether or not
anything about value is in them, and that is worse at early rungs, not better:
the base is still moving fast on its own, so the drift is larger and the signal
it hides is smaller. Reading an early cosine against zero would manufacture
exactly the trend this is looking for.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from cleanba.cleanba_impala import load_train_state
from jax.flatten_util import ravel_pytree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from goalmisgen import provenance  # noqa: E402
from goalmisgen.analysis.weights import cosine, fit_axis_and_drift, permutation_cosines, permutation_p_value  # noqa: E402
from goalmisgen.configs.env import MazeConfig  # noqa: E402
from goalmisgen.ladder import Rung, discover_rungs  # noqa: E402
from goalmisgen.volume import discover_arms  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path("/workspace/data"))
    parser.add_argument("--agent", type=str, required=True, help="The run whose ladder is read, e.g. novalue11.s1234.")
    parser.add_argument("--levels", type=str, required=True, help="Any dataset at the base values; used only to load.")
    parser.add_argument("--objectives", type=int, nargs="+", default=[0, 1])
    parser.add_argument(
        "--arm-steps",
        type=int,
        default=400_000,
        help="Which sweep to read. Arms of different lengths are not comparable, and everything "
        "here compares across rungs, so this is the one setting that must not vary down a ladder.",
    )
    parser.add_argument("--at", type=int, default=-1, help="Which checkpoint of each arm, in step order.")
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--size", type=int, default=11)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--reference",
        type=str,
        default=None,
        help="Which rung the others are compared against. Defaults to the deepest one in training.",
    )
    return parser.parse_args()


def eval_config(args: argparse.Namespace, values: tuple[float, ...]) -> MazeConfig:
    return MazeConfig(
        max_episode_steps=120,
        num_envs=2,
        min_size=args.size,
        max_size=args.size,
        feature_value_correlation=1.0,
        objective_values=values,
        value_encoding="none",
        colour_is_the_only_value_cue=True,
        level_dataset=args.levels,
        dataset_split="test",
        asynchronous=False,
        seed=args.seed,
    )


def fit_rung(args: argparse.Namespace, rung: Rung, values: tuple[float, ...]) -> dict:
    """One rung's axes, and every statistic that needs the arms themselves.

    The permutation null is built here rather than by the caller, because it is
    the only consumer of the per-arm diffs and they are far too large to keep. A
    rung holds fifty of them, each the whole parameter vector; carrying every
    rung's to the end would cost tens of gigabytes to produce a handful of
    cosines. So the diffs live inside this function and what leaves it is small.
    """
    directory = args.data / "runs" / rung.agent
    config = eval_config(args, values)
    _, _, _, base_state, _ = load_train_state(directory / rung.checkpoint_path, env_cfg=config)
    base_flat, _ = ravel_pytree(base_state.params)
    base_flat = np.asarray(base_flat, dtype=np.float64)

    axes: dict[int, np.ndarray] = {}
    drifts: dict[int, np.ndarray] = {}
    offsets: dict[int, np.ndarray] = {}
    diffs: dict[int, np.ndarray] = {}
    for objective in args.objectives:
        base_value = values[objective]
        arms = discover_arms(directory / "arms", objective, base_value, steps=args.arm_steps, at=args.at, family="o")
        if len(arms) < 3:
            print(f"  o{objective}: {len(arms)} arms, too few to fit a slope -- skipping")
            continue
        trained = sorted(arms)
        stack = []
        for value in trained:
            _, _, _, state, _ = load_train_state(arms[value], env_cfg=config)
            flat, _ = ravel_pytree(state.params)
            stack.append(np.asarray(flat, dtype=np.float64) - base_flat)
        offsets[objective] = np.array(trained) - base_value
        diffs[objective] = np.stack(stack)
        axes[objective], drifts[objective] = fit_axis_and_drift(offsets[objective], diffs[objective])
        print(
            f"  o{objective}: {len(arms):>2} arms  |axis| {np.linalg.norm(axes[objective]):>8.3g}"
            f"  |drift| {np.linalg.norm(drifts[objective]):>8.3g}"
            f"  null arm {'present' if base_value in arms else 'ABSENT'}"
        )

    entry: dict = {
        "rung": rung,
        "axes": axes,
        "norms": {o: float(np.linalg.norm(a)) for o, a in axes.items()},
        "drift": float(np.mean([np.linalg.norm(d) for d in drifts.values()])) if drifts else float("nan"),
        "cos": None,
        "p": None,
        "dim2": None,
    }
    first, second = args.objectives[0], args.objectives[1] if len(args.objectives) > 1 else args.objectives[0]
    if first in axes and second in axes and first != second:
        entry["cos"] = cosine(axes[first], axes[second])
        null = permutation_cosines(offsets[first], diffs[first], axes[second], resamples=args.resamples, seed=args.seed)
        entry["p"] = permutation_p_value(entry["cos"], null, alternative="less")
        entry["dim2"] = second_dimension_share(axes[first], axes[second])
    # Diffs go out of scope here; only the axes survive, one vector per objective.
    return entry


def second_dimension_share(axis_a: np.ndarray, axis_b: np.ndarray) -> float:
    """How much of two unit axes does not lie along their first shared direction.

    Zero when they are collinear (one knob, whichever sign), one half when they
    are orthogonal (two registers). Reported beside the cosine because it says
    the same thing without a sign, and because it is what generalises to three
    objectives where a cosine does not.
    """
    stacked = np.stack([axis_a / np.linalg.norm(axis_a), axis_b / np.linalg.norm(axis_b)])
    singular = np.linalg.svd(stacked, compute_uv=False)
    return float(singular[1] ** 2 / (singular**2).sum())


def main() -> None:
    args = parse_args()
    print(provenance.header() + "\n")

    rungs = discover_rungs(args.data, args.agent)
    if not rungs:
        sys.exit(f"no rungs found for {args.agent}; runs/<agent>.at<steps>/BASE.json is what this reads")

    values = tuple(json.loads((args.data / "runs" / args.agent / "BASE.json").read_text())["values"])
    print(f"agent {args.agent}, base values {values}, arms at {args.arm_steps:,} steps")
    print(f"{len(rungs)} rungs: {', '.join(r.label for r in rungs)}\n")

    fitted = []
    for rung in rungs:
        print(f"=== {rung.label}  ({rung.checkpoint}) ===")
        fitted.append(fit_rung(args, rung, values))

    reference = next((f for f in fitted if f["rung"].agent == args.reference), fitted[-1])
    print(f"\nreference rung: {reference['rung'].label} ({reference['rung'].agent})")

    print("\n\n=== is there an axis, and is it one axis? ===\n")
    print(f"{'rung':>8}{'|axis_0|':>10}{'|axis_1|':>10}{'|drift|':>10}{'axis/drift':>12}{'cos(a0,a1)':>12}{'p':>8}{'dim2':>8}")
    for entry in fitted:
        if entry["cos"] is None:
            continue
        first, second = args.objectives[0], args.objectives[1]
        ratio = np.mean(list(entry["norms"].values())) / entry["drift"] if entry["drift"] else float("nan")
        print(
            f"{entry['rung'].label:>8}{entry['norms'][first]:>10.3g}{entry['norms'][second]:>10.3g}"
            f"{entry['drift']:>10.3g}{ratio:>12.2f}{entry['cos']:>12.3f}{entry['p']:>8.3f}{entry['dim2']:>8.3f}"
        )

    print("\n\n=== has the axis settled where it ends up? ===\n")
    print(f"{'rung':>8}" + "".join(f"{f'cos(o{o}@t, o{o}@end)':>22}" for o in args.objectives))
    for entry in fitted:
        row = f"{entry['rung'].label:>8}"
        for objective in args.objectives:
            if objective in entry["axes"] and objective in reference["axes"]:
                row += f"{cosine(entry['axes'][objective], reference['axes'][objective]):>22.3f}"
            else:
                row += f"{'—':>22}"
        print(row)

    print(
        "\nRead the first table down the |axis|/|drift| column: an axis buried under the drift "
        "\nof its own fine-tune is not yet an axis, whatever its cosine says. Then read cos(a0,a1): "
        "\nnear zero is two registers, near -1 is one threshold, and the rung where that changes "
        "\nis the answer to what the ladder was built for."
    )


if __name__ == "__main__":
    main()
