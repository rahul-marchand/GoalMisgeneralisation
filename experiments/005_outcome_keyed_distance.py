"""Does the network hold a distance field for both objectives, or only its target?

    uv run python experiments/005_outcome_keyed_distance.py CHECKPOINT --levels DIR

The pilot found maze-aware distance in the recurrent state, measured to a fixed
objective. But at test rho=1.0 the agent goes to feature 0 in about three
quarters of episodes, so that measurement mixed "the objective it is heading to"
with "the one it ignored" — and a clean field for the target averaged with noise
for the other looks exactly like a weak field for both.

**The obvious fix does not work, and it is worth saying why.** Re-keying the
target to "distance to whichever objective the agent chose" makes the quantity
change identity between episodes. A 1x1 linear probe is one fixed map applied
everywhere; if the network holds the two fields in different channels, no such
map can follow that switch. Asked this way the probe scores badly for its own
reasons, and the first version of this experiment measured exactly that — the
"ignored" keying scored *higher* than the "chosen" one, which is the tell.

So the identity stays fixed and the outcome enters by splitting episodes. One
probe per objective, each decoding a quantity that never changes meaning. Then
for each episode, the probe for the objective it reached is **own** and the
other is **other**. Every episode contributes to both sides, so the comparison
is genuinely paired and the between-maze variance cancels.

Either answer is mechanistic:

**own = other** — the network computes a field for each objective and compares
them, so the choice is made *after* the distances exist. That is the substrate
a utility comparison needs.

**own > other** — it commits to a target first and computes distance second, so
the comparison happens somewhere this probe cannot see.

Read the controls first: ``oracle`` must pass and ``null`` and ``shuffled`` must
fail, or nothing below them is readable. The headline columns are ``partial``
and ``hard shape``, both immune to how the fit is scaled; ``hard R2`` is printed
as a diagnostic and runs negative for every arm, because the recalibration is
fitted over all cells while the hard subset has a systematically larger target.
"""

from __future__ import annotations

import argparse
import operator
from pathlib import Path

import jax
import numpy as np
from cleanba.cleanba_impala import load_train_state

from goalmisgen import provenance
from goalmisgen.analysis import fields, metrics, targets
from goalmisgen.analysis.activations import Capture, RolloutCache, collect_rollouts, require_one_actor
from goalmisgen.analysis.probes import Feature
from goalmisgen.configs.env import MazeConfig
from goalmisgen.configs.presets import maze_drc33

N_FEATURES = 2
TAU = 4.0
"""Detour threshold in cells, fixed before any checkpoint was looked at.

At tau=4 roughly 55% of cells on an 11x11 level are hard cells, so the subset is
the majority rather than a thin tail. Sensitivity at 2 and 8 is printed.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--levels", type=str, default=None)
    parser.add_argument("--size", type=int, default=11)
    parser.add_argument("--fit-split", type=str, default="valid", help="Probes are fitted here...")
    parser.add_argument("--split", type=str, default="test", help="...and scored here, so no level is shared.")
    parser.add_argument("--train-episodes", type=int, default=256)
    parser.add_argument("--test-episodes", type=int, default=512)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--correlation", type=float, default=1.0)
    parser.add_argument("--think", type=int, nargs="+", default=[0, 4])
    parser.add_argument("--randomise-values", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(provenance.header())

    def env_config(seed: int, split: str) -> MazeConfig:
        settings: dict[str, object] = dict(
            max_episode_steps=120,
            num_envs=args.num_envs,
            min_size=args.size,
            max_size=args.size,
            feature_value_correlation=args.correlation,
            randomise_values=args.randomise_values,
            level_dataset=args.levels,
            asynchronous=False,
            seed=seed,
        )
        if args.levels:
            settings["dataset_split"] = split
        return MazeConfig(**settings)  # type: ignore[arg-type]

    policy, _, _, train_state, update = load_train_state(args.checkpoint, env_cfg=env_config(0, args.fit_split))
    print(f"checkpoint {args.checkpoint.name}  (update {update})")

    _, _, untrained_params = maze_drc33(min_size=args.size, max_size=args.size).net.init_params(
        env_config(0, args.fit_split).make(), jax.random.PRNGKey(12345)
    )
    registry = {"agent": None, "random": untrained_params}

    def collect(split):
        def run(capture: Capture, seed: int, n_episodes: int):
            return collect_rollouts(
                env_config(seed, split).make(),
                policy,
                train_state.params,
                n_episodes,
                seed=seed,
                probe_params=registry[capture.reader],
                probe_steps_to_think=capture.steps_to_think,
            )

        return RolloutCache(run)

    fit_cache, score_cache = collect(args.fit_split), collect(args.split)

    by_feature = {
        feature: targets.DistanceToObjective(targets.fixed(feature), name=f"d->f{feature}", n_features=N_FEATURES)
        for feature in range(N_FEATURES)
    }
    activations = Feature("activations", operator.attrgetter("features"))

    header = f"{'capture':>12}{'target':>10}{'arm':>26}{'partial':>9}{'within':>8}{'hard shape':>11}"
    print(f"\n{header}{'hard R2':>9}{'MAE':>7}{'dim':>5}{'drop':>6}{'n_hard':>8}")

    for think in args.think:
        captures = [Capture(f"trained@{think}", reader="agent", steps_to_think=think)]
        if think == 0:
            captures.append(Capture("untrained@0", reader="random"))
        require_one_actor(captures)

        for capture in captures:
            fit_rollouts = fit_cache.get(capture, 0, args.train_episodes)
            score_rollouts = score_cache.get(capture, 9999, args.test_episodes)
            split = {}

            for feature, target in by_feature.items():
                arms = [activations, *targets.controls(target, score_rollouts)]
                fit_arms = [activations, *targets.controls(target, fit_rollouts)]

                scores = {}
                for arm, fit_arm in zip(arms, fit_arms):
                    train = fields.cell_data(fit_rollouts, fit_arm, target)
                    test = fields.cell_data(score_rollouts, arm, target)
                    result = fields.field_probe(arm.name, train, test, tau=TAU)
                    scores[arm.name.split(":")[0]] = result
                    if arm is activations:
                        _, prediction, _, _ = fields.fit_predict(train, test)
                        split[feature] = (test, prediction)

                    print(
                        f"{capture.name:>12}{target.name:>10}{result.name:>26}{result.partial_r:>9.3f}"
                        f"{result.partial_r_within_episode:>8.3f}{result.hard_shape_r2:>11.3f}"
                        f"{result.hard_r2:>9.3f}{result.mae:>7.2f}"
                        f"{result.depth:>5}{result.dropped_columns:>6}{result.n_hard:>8,}"
                    )
                print(f"{'':>22}{check_rig(scores)}\n")

            report_splits(split, score_rollouts, capture.name)


def check_rig(scores) -> str:
    """The controls decide whether the rows above are readable, so say it aloud.

    Left to a reader to notice, this is the check that gets skipped - the first
    distance-band result was invalidated by a confound raised and waved through.

    Judged on ``hard_shape_r2`` rather than ``hard_r2``: the recalibration is
    fitted on every cell but the hard subset has a systematically larger target,
    so the level is wrong there for every arm and R2 goes negative even when the
    ordering is perfect. Shape is immune to that by construction.
    """
    missing = {"oracle", "null", "shuffled"} - set(scores)
    if missing:
        raise ValueError(f"a headline was about to be printed without its controls: missing {sorted(missing)}")

    problems = []
    if scores["oracle"].hard_shape_r2 < 0.9:
        problems.append(f"positive control failed ({scores['oracle'].hard_shape_r2:.3f})")
    for name in ("null", "shuffled"):
        if abs(scores[name].partial_r) > 0.2:
            problems.append(f"{name} control survived stratification ({scores[name].partial_r:.3f})")

    if problems:
        return "RIG INVALID - " + "; ".join(problems) + ". No number in this block means anything."
    return "controls ok (oracle passes, null and shuffled fail)"


def report_splits(split, rollouts, capture: str) -> None:
    """Which property of an objective predicts a sharper field for it?

    Each probe is fitted to a *fixed* objective, so the quantity it decodes
    never changes identity. The property enters by splitting episodes: for each
    episode one objective is the "first" side and the other the "second", and
    every episode contributes to both, so the bootstrap is genuinely paired.

    Value and utility agree on most levels, because the value gap only loses to
    distance when the cheaper objective is much nearer. The subset where they
    disagree is what separates them, and it is reported on its own.
    """
    n = N_FEATURES
    everywhere = list(range(len(rollouts)))
    disagree = [
        index
        for index, rollout in enumerate(rollouts)
        if targets.richer(rollout, n) is not None
        and targets.best_utility(rollout, n) is not None
        and targets.richer(rollout, n) != targets.best_utility(rollout, n)
    ]

    def gap_at_most(limit):
        return [
            index
            for index, rollout in enumerate(rollouts)
            if (gap := targets.distance_gap(rollout, n)) is not None and gap <= limit
        ]

    # Where value and distance can be told apart: the two objectives are the
    # same walk away, so anything distinguishing them is not about distance.
    # On the disagreement subset the two are collinear by construction - value
    # and utility only disagree when the cheaper objective is much nearer - so
    # that subset cannot answer this and this one can.
    close, closer = gap_at_most(4), gap_at_most(2)

    named = [
        ("chosen vs ignored", targets.reached, everywhere),
        ("richer vs poorer", lambda r: targets.richer(r, n), everywhere),
        ("nearer vs further", lambda r: targets.nearer(r, n), everywhere),
        ("higher vs lower utility", lambda r: targets.best_utility(r, n), everywhere),
        ("coin flip (control)", lambda r: targets.coinflip(r), everywhere),
        ("richer vs poorer | value != utility", lambda r: targets.richer(r, n), disagree),
        ("higher utility | value != utility", lambda r: targets.best_utility(r, n), disagree),
        ("richer vs poorer | gap <= 4", lambda r: targets.richer(r, n), close),
        ("richer vs poorer | gap <= 2", lambda r: targets.richer(r, n), closer),
        ("coin flip | gap <= 4 (control)", lambda r: targets.coinflip(r), close),
    ]

    print(
        f"{'':>12}{len(disagree)} of {len(rollouts)} episodes have value and utility disagreeing; "
        f"{len(close)} have the objectives within 4 steps of each other, {len(closer)} within 2\n"
    )
    print(f"{'':>12}{'split':>36}{'first':>9}{'second':>9}{'difference':>20}")
    for label, selector, episodes in named:
        line = one_split(split, rollouts, selector, episodes)
        print(f"{'':>12}{label:>36}{line}")
    print(f"{'':>12}({capture})\n")


def one_split(split, rollouts, selector, episodes) -> str:
    """Partial r either side of one split, and a paired interval on the gap."""
    keep = set(episodes)
    first, second = {"y": [], "p": [], "s": []}, {"y": [], "p": [], "s": []}

    for feature, (data, prediction) in split.items():
        picked = np.array([selector(rollouts[index]) for index in data.episode], dtype=object)
        usable = np.array([index in keep and value is not None for index, value in zip(data.episode, picked)])
        mine = np.array([value == feature for value in picked]) & usable
        theirs = np.array([value is not None and value != feature for value in picked]) & usable

        stratum = data.episode * 1000 + data.confound[:, 0].astype(np.int64)
        for side, rows in ((first, mine), (second, theirs)):
            side["y"].append(data.y[rows])
            side["p"].append(prediction[rows])
            side["s"].append(stratum[rows])

    packed = {
        name: tuple(np.concatenate(side[key]) for key in ("y", "p", "s"))
        for name, side in (("first", first), ("second", second))
    }
    if min(len(packed[name][0]) for name in packed) < 200:
        return f"{'too few cells':>38}"

    def statistic(name):
        y, prediction, stratum = packed[name]
        episode = stratum // 1000

        def compute(ids):
            rows = metrics.select_rows(episode, ids)
            return metrics.stratified_correlation(prediction[rows], y[rows], stratum[rows])

        return compute

    scores = {
        name: metrics.stratified_correlation(prediction, y, stratum) for name, (y, prediction, stratum) in packed.items()
    }
    low, high = metrics.bootstrap_paired(
        statistic("first"), statistic("second"), np.unique(packed["first"][2] // 1000)
    )
    mark = "" if low <= 0 <= high else "  *"
    return f"{scores['first']:>9.3f}{scores['second']:>9.3f}{f'[{low:+.3f}, {high:+.3f}]':>18}{mark:>2}"


if __name__ == "__main__":
    main()
