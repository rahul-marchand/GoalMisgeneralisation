"""Draw example fields with the expert's route, for either Craftax task.

    uv run python figures/plot_craftax_examples.py DEMOS [--n 6] [--out figures/craftax/fig_examples_<task>.png]

Tiles: bedrock grey, stone deposits dark grey, tree green, ores by kind, player black, table brown where
the plan puts it. The chosen objective's route is drawn as a line with the
action count; the other objective's plan cost is written beside it so the
trade-off the expert made is visible on each field.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from goalmisgen.craftax.blocks import Action, Block  # noqa: E402
from goalmisgen.offline.demonstrations import load_demonstrations  # noqa: E402

COLOUR = {
    int(Block.FURNACE): "#9a9992",
    int(Block.STONE): "#6e6c66",
    int(Block.GRASS): "#eef3e6",
    int(Block.TREE): "#3f8f4a",
    int(Block.COAL): "#2b2b2b",
    int(Block.IRON): "#c96a2b",
    int(Block.DIAMOND): "#3fb6d9",
}
KIND_NAME = {int(Block.COAL): "coal", int(Block.IRON): "iron", int(Block.DIAMOND): "diamond"}
DIR = {int(Action.UP): (-1, 0), int(Action.DOWN): (1, 0), int(Action.LEFT): (0, -1), int(Action.RIGHT): (0, 1)}


def walk(tiles: np.ndarray, start, actions):
    """Positions visited and the cells where DO/PLACE acted, replaying the route's movement rules."""
    solid = {
        int(Block.STONE),
        int(Block.TREE),
        int(Block.COAL),
        int(Block.IRON),
        int(Block.DIAMOND),
        int(Block.CRAFTING_TABLE),
        int(Block.WATER),
    }
    pos, facing = tuple(start), (1, 0)
    path, events = [pos], []
    grid = tiles.copy()
    for a in actions:
        a = int(a)
        if a < 0:
            break
        if a in DIR:
            facing = DIR[a]
            nxt = (pos[0] + facing[0], pos[1] + facing[1])
            if 0 <= nxt[0] < grid.shape[0] and 0 <= nxt[1] < grid.shape[1] and grid[nxt] not in solid:
                pos = nxt
                path.append(pos)
        else:
            faced = (pos[0] + facing[0], pos[1] + facing[1])
            if a == int(Action.DO) and grid[faced] in (
                int(Block.TREE),
                int(Block.STONE),
                int(Block.COAL),
                int(Block.IRON),
                int(Block.DIAMOND),
            ):
                events.append(("do", faced, grid[faced]))
                grid[faced] = int(Block.GRASS) if grid[faced] == int(Block.TREE) else int(Block.PATH)
            elif a == int(Action.PLACE_TABLE):
                events.append(("table", faced, None))
                grid[faced] = int(Block.CRAFTING_TABLE)
            elif a in (int(Action.MAKE_WOOD_PICKAXE), int(Action.MAKE_STONE_PICKAXE)):
                events.append(("craft", pos, a))
    return path, events


def draw(ax, demos, i) -> None:
    level = demos.level(i)
    task = demos.task
    tiles = task.tiles(level)
    size = tiles.shape[0]
    for r in range(size):
        for c in range(size):
            ax.add_patch(Rectangle((c, size - 1 - r), 1, 1, color=COLOUR.get(int(tiles[r, c]), "white"), lw=0))
    route = demos.routes([i])[0]
    path, events = walk(tiles, level.agent_start, route)
    ys = [size - 0.5 - r for r, _ in path]
    xs = [c + 0.5 for _, c in path]
    ax.plot(xs, ys, color="#0b0b0b", lw=1.6, alpha=0.85, solid_capstyle="round")
    for k, (r, c) in enumerate(path[1:], 1):
        ax.text(c + 0.5, size - 0.5 - r, str(k), ha="center", va="center", fontsize=4.5, color="white" if k % 2 else "#0b0b0b")
    for kind, (r, c), what in events:
        if kind == "table":
            ax.add_patch(Rectangle((c + 0.15, size - 1 - r + 0.15), 0.7, 0.7, color="#7a4a1e", lw=0))
        elif kind == "do":
            ax.plot(c + 0.5, size - 0.5 - r, marker="x", color="#eb6834", ms=5, mew=1.4)
    r0, c0 = level.agent_start
    ax.plot(c0 + 0.5, size - 0.5 - r0, marker="o", color="#0b0b0b", ms=6)
    target = int(demos.target[i])
    costs = [int(x) for x in demos.distances[i]]
    names = [KIND_NAME.get(task.kinds[o.feature_id], "?") for o in level.objectives]
    values = [o.value for o in level.objectives]
    title = "  |  ".join(
        f"{'>' if k == target else ' '}{names[k]} v={values[k]:.2f} cost {costs[k]}" for k in range(len(names))
    )
    ax.set_title(title, fontsize=7.5, loc="left")
    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("demos", type=Path)
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    demos = load_demonstrations(args.demos)
    task = demos.meta["task"]["task"]
    cols = 3
    rows = -(-args.n // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 4.2 * rows))
    for k, ax in enumerate(np.asarray(axes).ravel()):
        if k < args.n:
            draw(ax, demos, args.start + k)
        else:
            ax.axis("off")
    fig.suptitle(
        f"{task}: expert routes (black, numbered), DO at x, table in brown; '>' marks the chosen objective, values shown",
        fontsize=9,
    )
    out = args.out or Path(__file__).parent / "craftax" / f"fig_examples_{task.split('-')[-1]}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
