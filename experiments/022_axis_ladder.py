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

``writes``      and the one that decides it: written into that rung's own
                weights, does the axis move the agent? ``--write`` adds it.
                Everything above can look healthy on a direction fitted to
                noise -- a norm is a length, and two noisy estimates of the same
                noise are correlated. What noise cannot do is move a trade-off in
                the direction asked for, by an amount that tracks how much was
                asked. A rung whose axis does not write has no axis, whatever the
                weights-only columns say, and the earliest rung that does write is
                the answer to when the axis appears.

Only ``offset * axis`` is ever written. An arm is ``drift + offset * axis + eps``
and the other two terms are discarded on purpose: drift is what the updates cost
whatever they were for, which the null arm measures and which moves behaviour not
at all, and eps is what the fit could not explain. Each write is held out too --
the axis written at an offset is fitted without the arm trained at it.

Without ``--write`` this is weights only, so it is cheap and runs anywhere.
``--write`` needs a GPU and episodes.

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

from functools import partial  # noqa: E402

import jax  # noqa: E402

from goalmisgen import provenance  # noqa: E402
from goalmisgen.analysis import collect_episode_outcomes, metrics, summarise  # noqa: E402
from goalmisgen.analysis.behaviour import (  # noqa: E402
    indifference_point,
    value_distance_decisions,
    write_verdict,
)
from goalmisgen.analysis.weights import (  # noqa: E402
    cosine,
    fit_axis_and_drift,
    permutation_cosines,
    permutation_norms,
    permutation_p_value,
    split_half_reliability,
)
from goalmisgen.configs.env import MazeConfig  # noqa: E402
from goalmisgen.ladder import Rung, discover_rungs  # noqa: E402
from goalmisgen.volume import arm_is_complete, discover_arms  # noqa: E402


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
        "--rungs",
        type=float,
        nargs="+",
        default=None,
        help="Analyse only the rungs nearest these step counts, in millions. Defaults to every "
        "rung on the ladder. Use it to revisit a few rungs without paying to refit the rest, "
        "which for a full ladder is most of the run.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Also write each rung's axis into that rung's own weights and measure what the "
        "agent then does. This is what decides whether a rung has an axis at all: a direction "
        "that does not move behaviour when written is a direction fitted to noise, whatever "
        "its norm and its cosines say. Costs rollouts, so it needs a GPU.",
    )
    parser.add_argument(
        "--write-objective",
        type=int,
        default=1,
        help="Which objective's axis is written. One is enough to establish that writing works, "
        "and cos(axis_0, axis_1) already says the two are the same edit.",
    )
    parser.add_argument(
        "--write-offsets",
        type=float,
        nargs="+",
        default=[-0.45, -0.20, 0.20, 0.45],
        help="Offsets to write. The extremes carry the test and the interior points say whether "
        "the response is graded rather than a step.",
    )
    parser.add_argument(
        "--write-near",
        type=float,
        nargs="+",
        default=None,
        help="Write only the rungs nearest these step counts, in millions. Every rung is still "
        "fitted and appears in the weights tables; this decides which ones pay for rollouts. "
        "A rung with no axis and a base that cannot reach objectives has a foregone write, so "
        "spending an hour of GPU re-establishing that is not worth it once it is established.",
    )
    parser.add_argument("--episodes", type=int, default=1024)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument(
        "--reach-floor",
        type=float,
        default=0.95,
        help="Below this reach the exchange rate is not a measurement of anything: an agent that "
        "does not finish episodes has no trade-off to read. Early rungs can fail this before any "
        "write is applied, which is a fact about the base agent and not about the axis.",
    )
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
        num_envs=args.num_envs,
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


def measure(params, policy, get_action, envs, args, label: str) -> tuple[float, float, float, float]:
    """Exchange rate in extra steps, its 95% interval, and whether the agent still finishes."""
    carry = policy.apply(params, jax.random.PRNGKey(args.seed), envs.observation_space.shape, method=policy.initialize_carry)
    state = {"carry": carry, "key": jax.random.PRNGKey(args.seed)}

    def act(observations, starts):
        state["carry"], action, _, state["key"] = get_action(
            params, state["carry"], observations, starts, state["key"], temperature=0.0
        )
        return np.asarray(action)

    outcomes = collect_episode_outcomes(envs, act, args.episodes, seed=args.seed)
    gaps, took_richer, _ = value_distance_decisions(outcomes)
    point = indifference_point(gaps, took_richer)
    low, high = metrics.bootstrap_episodes(
        lambda rows: indifference_point(gaps[rows], took_richer[rows]),
        np.arange(len(gaps)),
        resamples=200,
        seed=args.seed,
    )
    summary = summarise(outcomes)
    reached, optimal = summary.reached_objective, summary.chose_optimal
    # Both, because they say different things and only one of them is about the
    # trade-off. reached_objective is whether the agent finishes at all -- it can
    # be low simply because the agent wanders -- while chose_optimal is, of the
    # episodes it did finish, how often it took the higher-utility objective.
    # An agent can reach 70% and choose well on those, which is a competent
    # trade-off wrapped in poor navigation, and reporting reach alone would call
    # that incompetent. The ladder's early rungs are exactly where the two can
    # come apart.
    print(f"    {label:>34}{point:>9.1f}  [{low:6.1f},{high:6.1f}]{reached:>10.1%}{optimal:>10.1%}")
    return point, low, high, reached


def write_test(args, rung, base_flat, unravel, trained, stack, base_value, policy, get_action, envs) -> dict:
    """Write the axis into this rung's own weights and see whether behaviour moves.

    This is the measurement that decides whether a rung has an axis. Everything
    in the weights-only table can look healthy on a direction fitted to noise: a
    norm is just a length, and a cosine between two noisy estimates of the same
    noise is not zero either. What noise cannot do is move an agent's trade-off
    in the direction asked for, by an amount that tracks how much was asked.

    Only ``offset * axis`` is written. Each arm is ``drift + offset * axis + eps``
    and the other two terms are deliberately discarded: drift is what the updates
    cost whatever they were for -- the null arm measures it, and it moves
    behaviour not at all -- and eps is what the fit could not explain. An axis
    that needs either of them to reproduce its arm is not a direction carrying
    the value, which is the whole claim under test.

    Each write is also *held out*: the axis written at offset ``o`` is fitted
    without the arm trained at ``o``, so nothing about the arm being predicted
    went into the direction that predicts it.

    Two controls. The unwritten base, which says whether the agent had a legible
    trade-off in the first place -- an early rung can fail on that alone, which is
    a fact about the agent and not about the axis. And a norm-matched random
    direction, which says whether a perturbation of that size moves the agent by
    itself.
    """
    print(f"  writing o{args.write_objective}'s axis, {args.episodes} episodes per point")
    print(f"    {'':>34}{'steps':>9}  {'95% interval':>15}{'reached':>10}{'optimal':>10}")

    base_point, base_low, base_high, base_reached = measure(
        unravel(base_flat), policy, get_action, envs, args, "unwritten base"
    )

    written: list[dict] = []
    for offset in sorted(args.write_offsets):
        value = round(base_value + offset, 10)
        keep = [i for i, v in enumerate(trained) if abs(v - value) > 1e-9]
        if len(keep) < 3:
            continue
        held_axis, _ = fit_axis_and_drift(np.array([trained[i] for i in keep]) - base_value, stack[keep])
        point, low, high, reached = measure(
            unravel(base_flat + offset * held_axis),
            policy,
            get_action,
            envs,
            args,
            f"write {offset:+.2f} (held out)",
        )
        written.append({"offset": offset, "point": point, "low": low, "high": high, "reached": reached})

    # A random direction the length of the largest write, to show that moving the
    # weights this far does not by itself move the trade-off.
    control = None
    if written:
        widest = max(written, key=lambda w: abs(w["offset"]))
        axis_all, _ = fit_axis_and_drift(np.array(trained) - base_value, stack)
        rng = np.random.default_rng(args.seed)
        direction = rng.normal(size=base_flat.shape)
        direction *= abs(widest["offset"]) * np.linalg.norm(axis_all) / np.linalg.norm(direction)
        point, low, high, reached = measure(
            unravel(base_flat + direction), policy, get_action, envs, args, "norm-matched random"
        )
        control = {"point": point, "low": low, "high": high, "reached": reached}

    decision = write_verdict(base_reached, written, args.reach_floor)
    usable = [w for w in written if w["reached"] >= args.reach_floor]
    slope = (
        float(np.polyfit([w["offset"] for w in usable], [w["point"] for w in usable], 1)[0])
        if len(usable) >= 2
        else float("nan")
    )

    return {
        "base": {"point": base_point, "low": base_low, "high": base_high, "reached": base_reached},
        "written": written,
        "control": control,
        "slope": slope,
        "moved": decision.moved,
        "verdict": decision.verdict,
        "floor_is_binding": decision.floor_is_binding,
        "min_reach": decision.min_reach,
    }


def fit_rung(args: argparse.Namespace, rung: Rung, values: tuple[float, ...], rollout: dict | None = None) -> dict:
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
    base_flat, unravel = ravel_pytree(base_state.params)
    base_flat = np.asarray(base_flat, dtype=np.float64)

    axes: dict[int, np.ndarray] = {}
    drifts: dict[int, np.ndarray] = {}
    exists: dict[int, float] = {}
    reliability: dict[int, float] = {}
    offsets: dict[int, np.ndarray] = {}
    diffs: dict[int, np.ndarray] = {}
    for objective in args.objectives:
        base_value = values[objective]
        arms = discover_arms(directory / "arms", objective, base_value, steps=args.arm_steps, at=args.at, family="o")
        # An arm still training has checkpoints, and ``at=-1`` would read its
        # latest one -- a 200k arm fitted into a 400k grid under a @400k name.
        # That is the silent incomparability the naming scheme exists to prevent,
        # and discover_arms cannot catch it because the name is not wrong, the
        # arm is merely unfinished.
        finished = {value: path for value, path in arms.items() if arm_is_complete(path.parent.parent, args.arm_steps)}
        if len(finished) < len(arms):
            print(f"  o{objective}: {len(arms) - len(finished)} arm(s) have not reached {args.arm_steps:,}, excluded")
        arms = finished
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
        if rollout is not None and objective == args.write_objective:
            written_trained, written_stack = list(trained), diffs[objective]
        axes[objective], drifts[objective] = fit_axis_and_drift(offsets[objective], diffs[objective])
        norm_null = permutation_norms(offsets[objective], diffs[objective], resamples=args.resamples, seed=args.seed)
        exists[objective] = permutation_p_value(float(np.linalg.norm(axes[objective])), norm_null, alternative="greater")
        reliability[objective] = split_half_reliability(offsets[objective], diffs[objective], seed=args.seed)
        print(
            f"  o{objective}: {len(arms):>2} arms  |axis| {np.linalg.norm(axes[objective]):>8.3g}"
            f"  |drift| {np.linalg.norm(drifts[objective]):>8.3g}"
            f"  p {exists[objective]:.4f}  reliability {reliability[objective]:+.3f}"
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
        "exists": exists,
        "reliability": reliability,
    }
    first, second = args.objectives[0], args.objectives[1] if len(args.objectives) > 1 else args.objectives[0]
    if first in axes and second in axes and first != second:
        entry["cos"] = cosine(axes[first], axes[second])
        null = permutation_cosines(offsets[first], diffs[first], axes[second], resamples=args.resamples, seed=args.seed)
        entry["p"] = permutation_p_value(entry["cos"], null, alternative="less")
        entry["dim2"] = second_dimension_share(axes[first], axes[second])

    entry["write"] = None
    if rollout is not None and args.write_objective in axes and rung.steps in rollout["write_rungs"]:
        entry["write"] = write_test(
            args,
            rung,
            base_flat,
            unravel,
            written_trained,
            written_stack,
            values[args.write_objective],
            rollout["policy"],
            rollout["get_action"],
            rollout["envs"],
        )
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
    if args.rungs is not None:
        keep = {min(rungs, key=lambda r, t=m * 1_000_000: abs(r.steps - t)).steps for m in args.rungs}
        rungs = [rung for rung in rungs if rung.steps in keep]
        if not rungs:
            sys.exit("--rungs matched nothing")

    values = tuple(json.loads((args.data / "runs" / args.agent / "BASE.json").read_text())["values"])
    print(f"agent {args.agent}, base values {values}, arms at {args.arm_steps:,} steps")
    print(f"{len(rungs)} rungs: {', '.join(r.label for r in rungs)}\n")

    rollout = None
    if args.write:
        # One environment set and one compiled action function for every rung and
        # every written point. Compiling per measurement would cost more than the
        # rollouts do, and the envs depend only on the base values, which no rung
        # changes.
        config = eval_config(args, values)
        policy, _, _, _, _ = load_train_state(args.data / "runs" / rungs[-1].agent / rungs[-1].checkpoint_path, env_cfg=config)
        if args.write_near is None:
            chosen = {rung.steps for rung in rungs}
        else:
            chosen = {min(rungs, key=lambda r, t=m * 1_000_000: abs(r.steps - t)).steps for m in args.write_near}
        rollout = {
            "policy": policy,
            "get_action": jax.jit(partial(policy.apply, method=policy.get_action), static_argnames="temperature"),
            "envs": config.make(),
            "write_rungs": chosen,
        }
        written_labels = [r.label for r in rungs if r.steps in chosen]
        print(f"writing {len(chosen)} of {len(rungs)} rungs: {', '.join(written_labels)}\n")

    fitted = []
    for rung in rungs:
        print(f"=== {rung.label}  ({rung.checkpoint}) ===")
        fitted.append(fit_rung(args, rung, values, rollout))

    reference = next((f for f in fitted if f["rung"].agent == args.reference), fitted[-1])
    print(f"\nreference rung: {reference['rung'].label} ({reference['rung'].agent})")

    print("\n\n=== is there an axis at all? ===\n")
    print(f"{'rung':>8}{'|axis_0|':>10}{'p_0':>8}{'rel_0':>8}" f"{'|axis_1|':>10}{'p_1':>8}{'rel_1':>8}   verdict")
    for entry in fitted:
        first, second = args.objectives[0], args.objectives[1]
        if first not in entry["axes"] or second not in entry["axes"]:
            continue
        p0, p1 = entry["exists"][first], entry["exists"][second]
        r0, r1 = entry["reliability"][first], entry["reliability"][second]
        # Both objectives are the same measurement of the same agent, so a rung
        # that only clears on one of them has not cleared.
        has = "axis" if max(p0, p1) < 0.05 and min(r0, r1) > 0.2 else "—"
        print(
            f"{entry['rung'].label:>8}{entry['norms'][first]:>10.3g}{p0:>8.3f}{r0:>8.2f}"
            f"{entry['norms'][second]:>10.3g}{p1:>8.3f}{r1:>8.2f}   {has}"
        )
    print(
        "\np is against a permutation null -- the length a slope of this grid reaches over\n"
        "these diffs with the offsets shuffled, so with no value in them. |axis| on its own\n"
        "says nothing: least squares returns a slope through any cloud. rel is split-half\n"
        "reliability, how much of the fitted direction is signal. A rung needs both, on both\n"
        "objectives: a large axis at low reliability is a long vector pointing nowhere in\n"
        "particular, and reliability without length is a direction with nothing along it."
    )

    print("\n\n=== one axis or two? ===\n")
    print(f"{'rung':>8}{'cos(a0,a1)':>12}{'p':>8}{'disattenuated':>15}{'dim2':>8}   reading")
    for entry in fitted:
        if entry["cos"] is None:
            continue
        first, second = args.objectives[0], args.objectives[1]
        r0, r1 = entry["reliability"][first], entry["reliability"][second]
        # Dividing by reliability is only meaningful when there is reliability to
        # divide by. results/three-objective.txt has this correction returning
        # cosines outside the range a cosine can take, which is a correction
        # announcing it has broken down; refusing is better than printing it.
        if min(r0, r1) > 0.2 and (adjusted := entry["cos"] / np.sqrt(r0 * r1)) and abs(adjusted) <= 1.0:
            # -0.61 is far closer to one knob than to two registers, and calling
            # it the latter on a single threshold overstated a real but partial
            # effect. The bands say how collinear, which is what is measured.
            if adjusted < -0.85:
                reading = "one knob"
            elif adjusted < -0.5:
                reading = "mostly one knob"
            elif adjusted < -0.2:
                reading = "loosely coupled"
            elif adjusted > 0.2:
                reading = "same sign -- not a difference"
            else:
                reading = "two registers"
            shown = f"{adjusted:.3f}"
        else:
            shown, reading = "—", "no axis to ask of"
        print(
            f"{entry['rung'].label:>8}{entry['cos']:>12.3f}{entry['p']:>8.3f}{shown:>15}" f"{entry['dim2']:>8.3f}   {reading}"
        )
    print(
        "\nThe raw cosine is attenuated toward zero by noise in both axes, so it understates\n"
        "collinearity wherever reliability is poor -- which is every early rung. Disattenuating\n"
        "divides that out, and is refused below a reliability of 0.2 rather than printed as a\n"
        "number the data cannot support.\n\n"
        "The distinction that matters: dim2 near 0.5 with a cosine near zero is what TWO value\n"
        "registers look like AND what two noise vectors look like. The table above is what\n"
        "separates them. Only a rung with an axis can be said to have one axis or two."
    )

    print("\n\n=== has the axis settled where it ends up? ===\n")
    header = "".join(f"{'cos(o%d@t,end)' % o:>17}{'disatt':>10}" for o in args.objectives)
    print(f"{'rung':>8}{header}")
    for entry in fitted:
        row = f"{entry['rung'].label:>8}"
        for objective in args.objectives:
            if objective in entry["axes"] and objective in reference["axes"]:
                raw = cosine(entry["axes"][objective], reference["axes"][objective])
                here, end = entry["reliability"].get(objective), reference["reliability"].get(objective)
                # Attenuation applies to this cosine exactly as it does to
                # cos(a0, a1): both axes are noisy estimates, so a small number
                # can mean "points somewhere else" or "points the same way, badly
                # measured", and only correcting for reliability separates them.
                # The same test the existence table applies. Gating on
                # reliability alone let 15.0M -- p = 0.21, no axis -- print a
                # corrected cosine of 0.002, a precise number about nothing.
                cleared = max(entry["exists"].get(objective, 1.0), reference["exists"].get(objective, 1.0)) < 0.05
                usable = here is not None and end is not None and min(here, end) > 0.2 and cleared
                adjusted = raw / np.sqrt(here * end) if usable else float("nan")
                shown = f"{adjusted:.3f}" if usable and abs(adjusted) <= 1.0 else "—"
                row += f"{raw:>17.3f}{shown:>10}"
            else:
                row += f"{'—':>17}{'—':>10}"
        print(row)
    print(
        "\nA rung with no reliable axis has no disattenuated column, and its raw cosine is not\n"
        "evidence that the direction rotates -- there is no direction there to rotate. Read this\n"
        "table only on the rungs the existence table cleared."
    )

    if args.write:
        print("\n\n=== does writing the axis move the agent? ===\n")
        print(
            f"{'rung':>8}{'base steps':>12}{'base reach':>12}"
            f"{'write -0.45':>13}{'write +0.45':>13}{'slope':>9}{'random':>9}  verdict"
        )
        earliest, near_miss = None, None
        for entry in fitted:
            test = entry.get("write")
            if test is None:
                continue
            base = test["base"]
            low = next((w for w in test["written"] if w["offset"] == min(args.write_offsets)), None)
            high = next((w for w in test["written"] if w["offset"] == max(args.write_offsets)), None)
            control = test["control"]
            print(
                f"{entry['rung'].label:>8}{base['point']:>12.1f}{base['reached']:>12.1%}"
                f"{(low['point'] if low else float('nan')):>13.1f}"
                f"{(high['point'] if high else float('nan')):>13.1f}"
                f"{test['slope']:>9.1f}"
                f"{(control['point'] if control else float('nan')):>9.1f}  {test['verdict']}"
                + (
                    f"  (floor binding: writes separate, min reach {test['min_reach']:.1%})"
                    if test["floor_is_binding"]
                    else ""
                )
            )
            if test["verdict"] == "writes" and earliest is None:
                earliest = entry["rung"]
            if test["floor_is_binding"] and near_miss is None:
                near_miss = entry["rung"]
        print(
            "\n'writes' means the 95% intervals at the two extreme writes are disjoint and the\n"
            "agent still finishes its episodes -- the edit moved the trade-off further than the\n"
            "measurement's own uncertainty. The random column is a norm-matched direction of the\n"
            "same length and should not move: if it does, the rung is measuring perturbation size\n"
            "rather than the axis. 'base cannot do the task' is not a verdict about the axis --\n"
            "an agent that does not reach objectives has no trade-off to write to."
        )
        if near_miss is not None:
            print(
                f"\nEarliest rung whose writes separate at all: {near_miss.label} ({near_miss.agent}) -- "
                f"but it fails the reach floor, so the floor rather than the axis decided it. "
                f"Reported because the floor was fixed in advance and must not be moved to fit; "
                f"read both numbers."
            )
        if earliest is not None:
            print(f"\nEarliest rung whose axis writes: {earliest.label} ({earliest.agent})")
        else:
            print("\nNo rung's axis moved behaviour. On this evidence there is no axis to find.")

    print(
        "\nRead the tables in order. The first says which rungs have an axis at all, and no "
        "\nlater table means anything on a rung it did not clear. The second says how collinear "
        "\nthe two objectives' axes are once that is established. The third says whether the "
        "\naxis yet points where it ends up. The write table is the only one that leaves the "
        "\nweights, and is what an axis has to survive to be called one."
    )


if __name__ == "__main__":
    main()
