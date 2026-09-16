"""Where do a crafting model's decoded plans diverge from the expert's?

    uv run python scripts/craft_diagnose.py RUN_DIR DEMOS [--levels 256] [--checkpoint STEP]

For a checkpoint of a crafting base: teacher-forced token accuracy by position
in the plan, the first divergence from the expert route (which leg of the
plan it falls in), what the decoded routes contain (table placed, pickaxes
crafted, DOs), how they end in the engine (reached, illegal, wasted, walls
mined), and the same on training fields to separate under-fitting from
failing to generalise. Written for the night stage 4 came back at 3% reach.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from goalmisgen.craftax.blocks import Action
from goalmisgen.offline.decode import greedy_decode, replay_all
from goalmisgen.offline.demonstrations import load_demonstrations
from goalmisgen.offline.model import targets_from_routes
from goalmisgen.offline.train import list_checkpoints, load_checkpoint


def leg_of(route: np.ndarray, position: int) -> str:
    """Which part of the expert's plan a position falls in, by the milestones before it."""
    seen = route[:position].tolist()
    if int(Action.MAKE_STONE_PICKAXE) in seen:
        return "5 to the ore (after stone pickaxe)"
    if int(Action.MAKE_WOOD_PICKAXE) in seen:
        n_do = seen.count(int(Action.DO))
        return "4 stone / fourth tree / back to table" if n_do >= 3 else "3 after wood pickaxe"
    if int(Action.PLACE_TABLE) in seen:
        return "2 at the table"
    return f"1 gathering wood (after {seen.count(int(Action.DO))} trees)"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    parser.add_argument("demos", type=Path)
    parser.add_argument("--train-demos", type=Path, default=None)
    parser.add_argument("--levels", type=int, default=256)
    parser.add_argument("--checkpoint", type=int, default=None)
    args = parser.parse_args()

    checkpoints = list_checkpoints(args.run)
    step, directory = checkpoints[-1] if args.checkpoint is None else next(c for c in checkpoints if c[0] == args.checkpoint)
    model, params = load_checkpoint(directory)
    print(f"{args.run.name} @ step {step}: {model.config}")

    def analyse(label: str, demos):
        idx = np.arange(min(args.levels, len(demos)))
        obs = demos.observations(idx)
        routes = demos.routes(idx)
        lengths = np.asarray(demos.lengths[idx])
        # teacher-forced accuracy by position
        targets = np.asarray(targets_from_routes(jnp.asarray(routes), jnp.asarray(lengths), model.config.eos))
        logits, _ = jax.jit(model.apply)(params, jnp.asarray(obs), jnp.asarray(routes))
        pred = np.asarray(jnp.argmax(logits, -1))
        valid = targets >= 0
        correct = (pred == targets) & valid
        print(f"\n== {label}: {len(idx)} fields, expert routes mean {lengths.mean():.1f} actions")
        print(
            f"teacher-forced token accuracy {correct.sum() / valid.sum():.3f}; whole-route accuracy {np.mean([(correct[i] | ~valid[i]).all() for i in range(len(idx))]):.3f}"
        )
        by_pos = [
            (p, correct[:, p].sum() / max(1, valid[:, p].sum()))
            for p in (0, 5, 10, 15, 20, 30, 40, 50)
            if p < valid.shape[1] and valid[:, p].sum() > 10
        ]
        print("accuracy by position: " + "  ".join(f"{p}:{a:.2f}" for p, a in by_pos))
        # first teacher-forced error, by leg
        legs = Counter()
        for i in range(len(idx)):
            wrong = np.nonzero(valid[i] & ~correct[i])[0]
            if len(wrong):
                legs[leg_of(routes[i], int(wrong[0]))] += 1
            else:
                legs["no error"] += 1
        print("first teacher-forced error falls in: " + "; ".join(f"{k}: {v}" for k, v in sorted(legs.items())))
        wrong_tokens = Counter()
        for i in range(len(idx)):
            wrong = np.nonzero(valid[i] & ~correct[i])[0]
            if len(wrong):
                p = int(wrong[0])
                wrong_tokens[
                    (
                        Action(int(targets[i, p])).name if targets[i, p] < 17 else "EOS",
                        Action(int(pred[i, p])).name if pred[i, p] < 17 else "EOS",
                    )
                ] += 1
        print(
            "most common first error (expert -> model): "
            + "; ".join(f"{a}->{b}: {n}" for (a, b), n in wrong_tokens.most_common(6))
        )
        # free-running decode and the engine: closed-loop when the task is receding, else open-loop
        closed = getattr(demos, "decode_closed_loop", None)
        decoded = closed(model, params, idx) if closed is not None else greedy_decode(model, params, obs)
        outcomes = replay_all(demos, idx, decoded)
        acts = decoded.actions

        def has(a):
            return np.mean([(acts[i, : decoded.lengths[i]] == int(a)).any() for i in range(len(idx))])

        print(
            f"decoded: mean length {decoded.lengths.mean():.1f}, eos {decoded.emitted_eos.mean():.2f}; contains PLACE_TABLE {has(Action.PLACE_TABLE):.2f}, MAKE_WOOD_PICKAXE {has(Action.MAKE_WOOD_PICKAXE):.2f}, MAKE_STONE_PICKAXE {has(Action.MAKE_STONE_PICKAXE):.2f}"
        )
        print(
            f"engine: reached {np.mean([o['reached_objective'] for o in outcomes]):.2f}, illegal moves/route {np.mean([o['illegal_moves'] for o in outcomes]):.1f}, wasted/route {np.mean([o['wasted_actions'] for o in outcomes]):.1f}, walls mined/route {np.mean([o['walls_mined'] for o in outcomes]):.2f}, matched expert {np.mean([(acts[i, :lengths[i]] == routes[i, :lengths[i]]).all() and decoded.lengths[i] == lengths[i] for i in range(len(idx))]):.2f}"
        )
        # where the decoded route first departs from the expert, split by which ore the expert went for
        for label_kind, kind_id in (("iron-target fields", 0), ("coal-target fields", 1)):
            depart = Counter()
            rows = [i for i in range(len(idx)) if int(demos.feature_ids[idx[i], int(demos.target[idx[i]])]) == kind_id]
            for i in rows:
                n = int(lengths[i])
                diff = np.nonzero(acts[i, :n] != routes[i, :n])[0]
                depart[leg_of(routes[i], int(diff[0])) if len(diff) else "identical"] += 1
            reached_k = np.mean([outcomes[i]["reached_objective"] for i in rows]) if rows else float("nan")
            print(
                f"{label_kind} ({len(rows)}): reached {reached_k:.2f}; decoded route first departs in: "
                + "; ".join(f"{k}: {v}" for k, v in sorted(depart.items()))
            )

    analyse("held out", load_demonstrations(args.demos, hide_values=True))
    if args.train_demos:
        analyse("training fields", load_demonstrations(args.train_demos, hide_values=True))


if __name__ == "__main__":
    main()
