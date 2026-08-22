"""Write a distance into the end-of-input token, and see if the decision moves by that much.

    uv run python experiments/032_bc_write_the_distance.py RUN_DIR \\
        --demos /workspace/data/offline/demos/test.rho100 [--depths 3 4] [--targets gap]

Five scalar interventions in this project have been nulls - a distance, a value,
a choice direction, at the DRC's recurrence and at its readout (``007``-``010``)
- and ``012`` diagnosed why: a scalar written into a *field* describes no maze.
Setting one cell's distance to 3 when its neighbours say 8 is a contradiction,
and the network sensibly ignores it. Only a whole rewritten plan moved anything.

That diagnosis is an argument about fields, and this is the first site in the
project where it does not apply. The end-of-input token is one vector that is
not about any cell; it is the position the first action is predicted from, and
the place a single number summarising the level would have to live. Writing "f0
is five steps further than you think" there is a coherent thing to say.

**The prediction is a number, not a direction of change.** The probe hands back
a calibrated direction: the minimum-norm activation change that moves the
decoded distance by exactly one cell
(:func:`goalmisgen.analysis.steering.from_probe`). Adding ``alpha`` of it should
move the exchange rate - the distance gap at which the richer objective stops
being taken - by ``alpha`` cells, in the direction that makes the inflated
objective look worse. A slope of 1 says the decoded quantity *is* what the model
compares; a slope near 0 says it is decodable and ignored, which is what every
previous scalar write found.

On ``bcnv11`` the values are fixed constants the model never sees, so distance
is the only quantity that varies between episodes and ``d0 - d1`` is the whole
of what the decision turns on. That makes ``--targets gap`` the sharpest arm
and the one to read first.

Depth is not a nuisance parameter here, as it is for a per-cell write. The head
reads SEP directly, so an edit at the *last* depth changes exactly one thing -
the first action - with no attention left to carry it anywhere. Earlier depths
let the action tokens attend to the edited summary for the whole route. Both are
worth having: a decision that moves only when the write can reach later tokens
says the model re-reads the summary as it walks.

Controls, all at identical norms:

``random``    a norm-matched random direction, which must do nothing
``opposite``  the same direction with the sign flipped, which must move the
              exchange rate the other way by the same amount
``other``     the direction fitted for the *other* objective, which under one
              knob is the same edit and under two registers is not
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from goalmisgen import provenance
from goalmisgen.analysis import steering, targets
from goalmisgen.analysis.behaviour import indifference_point, value_distance_decisions
from goalmisgen.analysis.probes import apply_linear
from goalmisgen.offline import summary
from goalmisgen.offline.decode import greedy_decode, replay_all, summarise_routes
from goalmisgen.offline.demos import DemoSet
from goalmisgen.offline.probe import capture
from goalmisgen.offline.train import list_checkpoints, load_checkpoint, load_run_config

N_FEATURES = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    parser.add_argument("--demos", type=Path, required=True)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--fit-episodes", type=int, default=1024)
    parser.add_argument("--episodes", type=int, default=2048)
    parser.add_argument("--depths", type=int, nargs="*", default=None, help="Default every depth.")
    parser.add_argument(
        "--targets",
        nargs="+",
        default=["gap", "f0", "f1"],
        choices=("gap", "f0", "f1"),
        help="What to write. 'gap' is d->f0 minus d->f1, which is what the choice turns on.",
    )
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[-4.0, -2.0, 0.0, 2.0, 4.0],
        help="Cells of decoded distance to add. The units are the point: the exchange rate should move "
        "by this much, so the table is a slope rather than a direction of change.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", type=Path, default=None)
    return parser.parse_args()


def distances(rollouts, feature: int) -> np.ndarray:
    """Steps from the agent to one objective, routing around the other.

    ``nan`` where the other objective walls it off, so the probe drops that
    episode rather than fitting a fill value - the same rule ``031`` uses.
    """
    values = np.array([targets.objective_distance(r, feature, N_FEATURES) for r in rollouts], dtype=float)
    values[~np.isfinite(values)] = np.nan
    return values


def wanted(rollouts, target: str) -> np.ndarray:
    if target == "gap":
        return distances(rollouts, 0) - distances(rollouts, 1)
    return distances(rollouts, int(target[1:]))


def sep_edit(direction: np.ndarray, alpha: float, n_episodes: int, n_cells: int) -> np.ndarray:
    """``(B, n_cells + 1, d_model)`` that is zero everywhere but the SEP position.

    The hook writes over a prefix of the sequence, so reaching SEP means passing
    a grid one longer than the maze and leaving the cells alone. Writing at the
    cells as well would be a different experiment - and the one ``030`` runs.
    """
    grid = np.zeros((n_episodes, n_cells + 1, len(direction)), dtype=np.float32)
    grid[:, n_cells] = alpha * direction
    return grid


def cell_and_sep(model, params, observations, edit, depth):
    """The SEP residual at the top, with ``edit`` applied at ``depth``."""
    return summary.sep_residuals_edited(model, params, observations, edit, depth)[model.config.n_layers]


def exchange_rate(outcomes: list[dict], resamples: int = 200, seed: int = 0) -> tuple[float, float, float]:
    gaps, took_richer, _ = value_distance_decisions(outcomes)
    if not len(gaps):
        return float("nan"), float("nan"), float("nan")
    point = indifference_point(gaps, took_richer)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(resamples):
        chosen = rng.integers(0, len(gaps), len(gaps))
        value = indifference_point(gaps[chosen], took_richer[chosen])
        if np.isfinite(value):
            draws.append(value)
    if not draws:
        return point, float("nan"), float("nan")
    low, high = np.quantile(draws, [0.025, 0.975])
    return point, float(low), float(high)


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

    fit_indices = np.arange(args.fit_episodes)
    measure_indices = np.arange(args.fit_episodes, args.fit_episodes + args.episodes)
    observations = demos.observations(measure_indices)
    print(
        f"{args.run.name} @ step {step:,}; {args.demos.name} (rho={demos.rho}); "
        f"{args.fit_episodes} fitted / {args.episodes} measured; values {'hidden' if hide_values else 'shown'}"
    )

    decoded = greedy_decode(model, params, observations)
    outcomes = replay_all(demos, measure_indices, decoded)
    print(f"\nunedited: {summarise_routes(demos, measure_indices, decoded, outcomes)}")
    base_point, base_low, base_high = exchange_rate(outcomes, seed=args.seed)

    # Rollouts only to build the targets: the probe reads SEP, not the cells,
    # but the levels and the model's own outcomes come from the same capture.
    fit_rollouts = capture(model, params, demos, fit_indices, layer=0)
    fit_sep = summary.sep_residuals(model, params, demos.observations(fit_indices))

    results: dict = {
        "run": args.run.name,
        "step": step,
        "demos": args.demos.name,
        "baseline": {"indifference": base_point, "ci": [base_low, base_high]},
        "depths": {},
    }

    for depth in depths:
        print(f"\n{'=' * 78}")
        print(f"depth {depth}" + ("  (embedding)" if depth == 0 else f"  (after block {depth})"))
        if depth == cfg.n_layers:
            print(
                "The head reads SEP directly, so a write here reaches the first action and nothing else:\n"
                "no attention layer follows to carry it into the rest of the route."
            )

        fitted = {}
        for target in args.targets:
            y = wanted(fit_rollouts, target)
            weights, mean, std = summary.fit_scalar(fit_sep[depth], y, seed=args.seed)
            direction = steering.from_probe(target, weights, std)
            achieved = steering.verify(direction, weights, mean, std, alpha=1.0)
            fitted[target] = (direction, weights, mean, std)
            print(
                f"\n{target}: |one cell| = {direction.unit_norm:.4f} of activation; "
                f"re-decoding a steered vector moves it by {achieved:+.3f} cells (must be +1.000)"
            )

            # Does the write survive to where the head reads? Blocks after this
            # one can rebuild SEP by attending to the maze tokens, which the
            # write leaves untouched - the "recomputed downstream" escape that
            # makes a null ambiguous. Measured rather than argued: the top
            # probe's decoded value, before and after an edit made here.
            if depth < cfg.n_layers:
                top_weights, top_mean, top_std = summary.fit_scalar(fit_sep[cfg.n_layers], y, seed=args.seed)
                probe_x = summary.sep_residuals(model, params, observations)
                before = apply_linear(probe_x[cfg.n_layers], top_weights, top_mean, top_std)
                carried = []
                for alpha in (a for a in args.alphas if a != 0.0):
                    grid = sep_edit(direction.delta, alpha, len(observations), cfg.n_cells)
                    after_stream = np.asarray(
                        cell_and_sep(model, params, observations, grid, depth)
                    )
                    after = apply_linear(after_stream, top_weights, top_mean, top_std)
                    carried.append((alpha, float(np.mean(after - before))))
                print(
                    "   carried to the top: "
                    + "  ".join(f"{alpha:+.0f}->{moved:+.2f}" for alpha, moved in carried)
                    + "  (cells of decoded shift; equal to alpha means the edit survives)"
                )

        for target in args.targets:
            direction, weights, mean, std = fitted[target]
            arms = {
                "write": direction,
                "opposite": steering.Direction("opposite", -direction.delta),
                "random": steering.matched_random("random", direction, seed=args.seed),
            }
            other = {"f0": "f1", "f1": "f0"}.get(target)
            if other in fitted:
                arms["other"] = steering.matched(other, fitted[other][0], direction)

            print(
                f"\n{'target':>8}{'arm':>10}{'alpha':>7}{'indiff':>9}{'95% CI':>18}"
                f"{'moved':>8}{'predicted':>11}{'reached':>9}{'legal':>7}"
            )
            print(
                f"{target:>8}{'none':>10}{0.0:>7.1f}{base_point:>9.2f}"
                f"{f'[{base_low:.2f},{base_high:.2f}]':>18}{'':>8}{'':>11}"
                f"{summarise_routes(demos, measure_indices, decoded, outcomes).behaviour.reached_objective:>9.1%}"
            )
            rows = []
            for arm, vector in arms.items():
                for alpha in args.alphas:
                    if alpha == 0.0:
                        continue
                    grid = sep_edit(vector.delta, alpha, len(observations), cfg.n_cells)
                    armed = greedy_decode(model, params, observations, edit=grid, edit_depth=depth)
                    armed_outcomes = replay_all(demos, measure_indices, armed)
                    summary_row = summarise_routes(demos, measure_indices, armed, armed_outcomes)
                    point, low, high = exchange_rate(armed_outcomes, seed=args.seed)
                    # Inflating the distance to an objective should make it be
                    # abandoned sooner, so the exchange rate falls by alpha.
                    predicted = -alpha if arm in ("write", "other") else (alpha if arm == "opposite" else 0.0)
                    print(
                        f"{target:>8}{arm:>10}{alpha:>7.1f}{point:>9.2f}{f'[{low:.2f},{high:.2f}]':>18}"
                        f"{point - base_point:>8.2f}{predicted:>11.1f}"
                        f"{summary_row.behaviour.reached_objective:>9.1%}{summary_row.legal:>7.1%}"
                    )
                    rows.append(
                        {
                            "arm": arm,
                            "alpha": alpha,
                            "indifference": point,
                            "ci": [low, high],
                            "moved": point - base_point,
                            "predicted": predicted,
                            "reached": summary_row.behaviour.reached_objective,
                            "legal": summary_row.legal,
                        }
                    )

            usable = [r for r in rows if r["arm"] == "write" and r["reached"] >= 0.95 and np.isfinite(r["moved"])]
            if len(usable) >= 2:
                slope = np.polyfit([r["alpha"] for r in usable], [r["moved"] for r in usable], 1)[0]
                print(
                    f"\nslope {slope:+.3f} cells of exchange rate per cell written "
                    f"(-1 = the decoded distance is the compared quantity; 0 = decodable and ignored)"
                )
                results.setdefault("slopes", {})[f"{depth}.{target}"] = float(slope)
            results["depths"].setdefault(str(depth), {})[target] = rows

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2, default=float))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
