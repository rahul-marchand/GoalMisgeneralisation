"""Does the route model hold a distance to both objectives, or only to its target?

    uv run python experiments/031_bc_outcome_keyed_distance.py RUN_DIR \\
        --demos /workspace/data/offline/demos/test.rho050 [--depths 0 1 2 3 4]

The offline twin of ``005_outcome_keyed_distance.py``, and the read the BC
stream never had: ``025`` decodes the *route*, ``030`` writes one, and nothing
has asked whether the quantity the route trades against is there at all.

The keying is the experiment, and the mistake ``005`` made is worth repeating
here because the fix is what the design is. Re-keying a probe's target to
"distance to whichever objective the model chose" makes the quantity change
identity between episodes, and one linear map cannot follow that switch; asked
that way a probe scores badly for its own reasons. So the identity stays fixed -
one probe for ``d->f0``, one for ``d->f1`` - and the outcome enters by splitting
episodes: for each episode the probe for the objective it reached is **own** and
the other is **other**. Every episode contributes to both sides, so the
comparison is paired and the between-maze variance cancels.

**own = other** - both distances exist before the choice, which is the substrate
a value-against-distance trade needs.
**own > other** - it commits to a target first and measures second, and the
comparison happens somewhere this probe cannot see.

Two sites, and separating them is what this architecture buys. The DRC's actor
is a Dense layer over its flattened recurrent grid, so a scalar held once and a
field averaged over cells reach the policy by the same path and no probe can
tell them apart. Here they are different token positions:

``cells``  the per-cell residual, decoding a *field* - how far is this cell from
           objective f. The DRC's question, ported unchanged, through
           ``analysis.fields`` and ``analysis.targets``.
``sep``    the end-of-input token, decoding a *scalar* - how far is the agent
           from objective f. One number per objective, at the position the first
           action is predicted from.

Read the controls before anything else, at either site: ``oracle`` must pass and
``null`` (Manhattan and Chebyshev from the same source, neither of which
requires solving the maze) and ``shuffled`` (a real field on the wrong maze)
must fail. Manhattan alone explains about a third of the variance in a true
field, which is more than enough to look like a finding.

No write here. Every scalar intervention in this project has been a null, and
``012``'s diagnosis was that a scalar edit describes no maze - but that argument
is about a *field*, and SEP is the first site where it does not apply. That
experiment is worth running only if this read works.
"""

from __future__ import annotations

import argparse
import json
import operator
from pathlib import Path

import jax
import numpy as np

from goalmisgen import provenance
from goalmisgen.analysis import fields, geometry, targets
from goalmisgen.analysis.probes import Feature, apply_linear
from goalmisgen.offline import summary
from goalmisgen.offline.decode import greedy_decode
from goalmisgen.offline.demos import DemoSet
from goalmisgen.offline.probe import capture
from goalmisgen.offline.train import initial_params, list_checkpoints, load_checkpoint, load_run_config

N_FEATURES = 2
TAU = 4.0
"""Detour threshold in cells, as in ``005`` - cells the free-space null gets
most wrong. Fixed there before any checkpoint was looked at; not retuned here."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    parser.add_argument("--demos", type=Path, required=True)
    parser.add_argument("--step", type=int, default=None, help="Checkpoint step; default the last.")
    parser.add_argument("--train-episodes", type=int, default=1024, help="A scalar probe reads d_model features from this many rows, not from tens of thousands of cells.")
    parser.add_argument("--test-episodes", type=int, default=1024)
    parser.add_argument("--depths", type=int, nargs="*", default=None, help="Default every depth.")
    parser.add_argument("--sites", nargs="+", choices=("cells", "sep"), default=["cells", "sep"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", type=Path, default=None)
    return parser.parse_args()


def agent_distance(rollout, feature_id: int) -> float:
    """Steps from the agent to one objective, routing around the other.

    ``inf`` where the other objective blocks the only corridor, which
    :func:`~goalmisgen.offline.summary.scalar_probe` drops rather than fills.
    """
    return targets.objective_distance(rollout, feature_id, N_FEATURES)


def label_of(feature) -> str:
    return "d0-d1" if feature == "gap" else f"d->f{feature}"


def distance_target(rollouts, feature) -> np.ndarray:
    """One number per episode: a distance, or the difference the choice turns on.

    ``inf`` - an objective the other one walls off - becomes ``nan`` so the
    probe drops the episode instead of learning a fill value.
    """
    if feature == "gap":
        first = distance_target(rollouts, 0)
        return first - distance_target(rollouts, 1)
    values = np.array([agent_distance(r, feature) for r in rollouts], dtype=float)
    values[~np.isfinite(values)] = np.nan
    return values


def check_rig(scores: dict) -> str:
    """The controls decide whether the rows above are readable, so say it aloud.

    Copied in spirit from ``005``: left to a reader to notice, this is the check
    that gets skipped, and the first distance-band result in this project was
    invalidated by a confound raised and waved through.
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


def check_scalar_rig(scores: dict) -> str:
    """The same judgement for the SEP probes, on the same statistic.

    Judged on ``partial_r`` rather than R2, and the extra clause is the one this
    site needs: the free-space null reads the distance to about R2 0.28 without
    solving any maze, so an arm below that has not shown a maze-aware quantity
    however far above zero it sits. Saying so here rather than leaving it to a
    reader is the same rule ``005``'s ``check_rig`` follows.
    """
    problems = []
    if scores["oracle"].r2 < 0.9:
        problems.append(f"positive control failed ({scores['oracle'].r2:.3f})")
    for name in ("null", "shuffled"):
        if abs(scores[name].partial_r) > 0.2:
            problems.append(f"{name} control survived stratification ({scores[name].partial_r:.3f})")
    if "untrained" in scores and scores["residual"].r2 <= scores["untrained"].r2:
        problems.append(
            f"an untrained network of the same shape reads it as well ({scores['untrained'].r2:.3f} "
            f"against {scores['residual'].r2:.3f}): this is the architecture, not the training"
        )
    if abs(scores["residual"].partial_r) <= abs(scores["null"].partial_r):
        problems.append(
            f"the residual carries no more maze-aware structure than free space "
            f"(partial {scores['residual'].partial_r:.3f} against the null's {scores['null'].partial_r:.3f})"
        )
    if problems:
        return "RIG INVALID - " + "; ".join(problems) + ". No number in this block means anything."

    note = "controls ok (oracle passes, untrained and shuffled fail)"
    if scores["residual"].r2 < scores["null"].r2:
        # Not a rig failure: the stratified column is the headline, for the
        # reason 005 makes it the headline. But a 128-dimensional probe that
        # cannot beat two numbers of free-space geometry on raw variance is a
        # weak read however clean its partial is, and saying so here keeps that
        # from being lost between a table and a summary.
        note += (
            f"; caution: raw R2 {scores['residual'].r2:.3f} is below the free-space null's "
            f"{scores['null'].r2:.3f}, so most of what is decodable here is geometry"
        )
    return note


def cells_site(fit_rollouts, score_rollouts, label: str, results: dict) -> None:
    """The field probe, one row per objective per arm - the DRC's table."""
    activations = Feature("residual", operator.attrgetter("features"))
    print(f"\n{'site':>8}{'target':>10}{'arm':>34}{'partial':>9}{'within':>8}{'hard shape':>11}{'hard R2':>9}{'MAE':>7}{'n_hard':>9}")

    for feature in range(N_FEATURES):
        target = targets.DistanceToObjective(targets.fixed(feature), name=f"d->f{feature}", n_features=N_FEATURES)
        arms = [activations, *targets.controls(target, score_rollouts)]
        fit_arms = [activations, *targets.controls(target, fit_rollouts)]

        scores = {}
        for arm, fit_arm in zip(arms, fit_arms):
            train = fields.cell_data(fit_rollouts, fit_arm, target)
            test = fields.cell_data(score_rollouts, arm, target)
            result = fields.field_probe(arm.name, train, test, tau=TAU)
            scores[arm.name.split(":")[0]] = result
            print(
                f"{label:>8}{target.name:>10}{result.name:>34}{result.partial_r:>9.3f}"
                f"{result.partial_r_within_episode:>8.3f}{result.hard_shape_r2:>11.3f}"
                f"{result.hard_r2:>9.3f}{result.mae:>7.2f}{result.n_hard:>9,}"
            )
        print(f"{'':>18}{check_rig(scores)}\n")
        results.setdefault("cells", {})[f"f{feature}"] = {
            name: {"partial_r": r.partial_r, "hard_shape_r2": r.hard_shape_r2, "hard_r2": r.hard_r2, "mae": r.mae}
            for name, r in scores.items()
        }


def sep_site(
    model, params, untrained, demos, fit_indices, score_indices, fit_rollouts, score_rollouts, depth, results, seed
) -> None:
    """The scalar probe at the end-of-input token, plus the own/other split.

    Three targets, and the third is the one the decision actually turns on. With
    the values fixed and hidden - which is what ``bcnv11`` is - the only thing
    varying between episodes is geometry, so ``d->f0 - d->f1`` *is* the quantity
    being compared against a constant. A model holding only that difference is
    the activation-space version of the one knob Experiment 2 found in weights.
    """
    fit_x = summary.sep_residuals(model, params, demos.observations(fit_indices))[depth]
    score_x = summary.sep_residuals(model, params, demos.observations(score_indices))[depth]
    fit_random = summary.sep_residuals(model, untrained, demos.observations(fit_indices))[depth]
    score_random = summary.sep_residuals(model, untrained, demos.observations(score_indices))[depth]

    print(f"\n{'target':>10}{'arm':>34}{'R2':>9}{'95% CI':>18}{'partial':>9}{'MAE':>7}{'slope':>8}{'dim':>5}{'n':>7}")
    predictions, truths = {}, {}
    for feature in list(range(N_FEATURES)) + ["gap"]:
        fit_y = distance_target(fit_rollouts, feature)
        score_y = distance_target(score_rollouts, feature)

        # The same three controls the field probes take, as one-vector features.
        oracle_fit, oracle_score = fit_y[:, None], score_y[:, None]
        null_fit = free_space(fit_rollouts, feature)
        null_score = free_space(score_rollouts, feature)
        order = summary_derangement(len(score_y), seed)
        shuffled_score = score_y[order][:, None]

        arms = {
            "residual": (fit_x, score_x),
            "untrained": (fit_random, score_random),
            "oracle": (np.nan_to_num(oracle_fit), np.nan_to_num(oracle_score)),
            "null": (null_fit, null_score),
            "shuffled": (np.nan_to_num(oracle_fit), np.nan_to_num(shuffled_score)),
        }
        stratum = null_score[:, 0]
        scores = {}
        for name, (train_x, test_x) in arms.items():
            result = summary.scalar_probe(name, train_x, fit_y, test_x, score_y, confound=stratum, seed=seed)
            scores[name] = result
            print(f"{label_of(feature):>10}{result}")

        if feature != "gap":
            weights, mean, std = summary.fit_scalar(fit_x, fit_y)
            predictions[feature] = apply_linear(score_x, weights, mean, std)
            truths[feature] = score_y
        results.setdefault("sep", {})[f"{feature}"] = {
            name: {"r2": r.r2, "partial_r": r.partial_r, "mae": r.mae, "slope": r.slope}
            for name, r in scores.items()
        }
        print(f"{'':>10}{check_scalar_rig(scores)}\n")

    reached = np.array([r.info.get("reached_feature_id", -1) for r in score_rollouts])
    own, other, (low, high) = summary.own_and_other(predictions, truths, reached, seed=seed)
    print(
        f"own {own:.2f} cells of error, other {other:.2f}, difference {other - own:+.2f} "
        f"[{low:+.2f}, {high:+.2f}]"
    )
    print(
        "Positive means the objective it went to is held more precisely than the one it passed up:\n"
        "a target chosen first and measured second. An interval spanning zero says both distances\n"
        "are there before the choice, which is what a value-against-distance trade needs."
    )
    results.setdefault("sep", {})["own_other"] = {"own": own, "other": other, "ci": [low, high]}


def free_space(rollouts, feature_id) -> np.ndarray:
    """Manhattan and Chebyshev distance from the agent to the objective.

    The null that needs no maze solved. It is why the field probes stratify by
    the confound rather than reporting a raw correlation, and the scalar probe
    needs it for the same reason.
    """
    if feature_id == "gap":
        return free_space(rollouts, 0) - free_space(rollouts, 1)
    rows = []
    for rollout in rollouts:
        agent = geometry.agent_cell(rollout.observation)
        objective = geometry.objective_cell(rollout.observation, feature_id)
        rows.append(
            [
                abs(agent[0] - objective[0]) + abs(agent[1] - objective[1]),
                max(abs(agent[0] - objective[0]), abs(agent[1] - objective[1])),
            ]
        )
    return np.array(rows, dtype=float)


def summary_derangement(count: int, seed: int) -> np.ndarray:
    order = np.random.default_rng(seed).permutation(count)
    for position, destination in enumerate(order):
        if position == destination:
            swap = (position + 1) % count
            order[position], order[swap] = order[swap], order[position]
    return order


def main() -> None:
    args = parse_args()
    print(provenance.header())
    print()

    checkpoints = list_checkpoints(args.run)
    step, directory = checkpoints[-1] if args.step is None else next(c for c in checkpoints if c[0] == args.step)
    model, params = load_checkpoint(directory)
    cfg = model.config
    hide_values = bool(load_run_config(args.run)["demos"].get("hide_values", False))
    demos = DemoSet.load(args.demos, hide_values=hide_values)
    depths = list(args.depths) if args.depths else list(range(cfg.n_layers + 1))

    untrained = initial_params(model, jax.random.PRNGKey(12345))
    fit_indices = np.arange(args.train_episodes)
    score_indices = np.arange(args.train_episodes, args.train_episodes + args.test_episodes)
    print(
        f"{args.run.name} @ step {step:,}; {args.demos.name} (rho={demos.rho}); "
        f"{args.train_episodes} fitted / {args.test_episodes} scored; {cfg.n_layers} blocks, "
        f"d_model {cfg.d_model}; values {'hidden' if hide_values else 'shown'}"
    )
    # Decode once: the routes label nothing here, but `reached_feature_id` is
    # what keys own against other, and it must be the model's own outcome.
    fit_decoded = greedy_decode(model, params, demos.observations(fit_indices))
    score_decoded = greedy_decode(model, params, demos.observations(score_indices))

    results: dict = {"run": args.run.name, "step": step, "demos": args.demos.name, "depths": {}}
    for depth in depths:
        label = "embed" if depth == 0 else f"block{depth}"
        print(f"\n{'=' * 78}\ndepth {depth}  ({label})")
        fit_rollouts = capture(model, params, demos, fit_indices, layer=depth, decoded=fit_decoded)
        score_rollouts = capture(model, params, demos, score_indices, layer=depth, decoded=score_decoded)

        at_depth: dict = {}
        if "cells" in args.sites:
            cells_site(fit_rollouts, score_rollouts, label, at_depth)
        if "sep" in args.sites:
            sep_site(
                model, params, untrained, demos, fit_indices, score_indices,
                fit_rollouts, score_rollouts, depth, at_depth, args.seed,
            )
        results["depths"][str(depth)] = at_depth

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2, default=float))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
