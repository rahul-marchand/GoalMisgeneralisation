"""Did the probe move before the behaviour did? Read it off an early_warning.sh sweep.

    uv run python scripts/early_warning_report.py results/early-warning.txt [more sweeps...] \
        [--out figures] [--name fig_early_warning]

``scripts/early_warning.sh`` writes, per agent and checkpoint, 002's choice
table at rho=1.0 and rho=0.0 and 003's plan-probe table, fitted at rho=1.0 and
scored at rho=1.0 and rho=0.0. This turns that text into two curves per agent:

  behavioural gap   chose_optimal(rho=1.0) - chose_optimal(rho=0.0)
  probe gap         AUC(trained, scored at rho=1.0) - AUC(trained, scored at rho=0.0)

and dates each: the first checkpoint after which the curve stays above half
its settled value. The guiding hypothesis is that the probe's date comes
first. The per-layer probe rows are parsed too and plotted faintly, since
which layer moves is part of the answer.

Control agents (trained at rho=0.5) are plotted alongside as the floor both
gaps should sit at when there is no proxy to pick up.
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BLUE, ORANGE, AQUA, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#9a9992"

HEADER = re.compile(r"^=== (\S+) @ (\d+) steps ===")
CHOICE = re.compile(r"^\s*([0-9.]+)\s+([0-9.]+)%\s+([0-9.]+)%\s+([0-9.]+)%\s+([0-9.]+)%\s+(-?[0-9.]+)\s+([0-9.]+)\s*$")
PROBE = re.compile(r"^\s*(\d+)\s+([0-9.]+)\s+(\S+)\s+([0-9.]+)\s+\[([0-9.]+), ([0-9.]+)\]\s+([0-9.]+)\s+([\d,]+)\s*$")
LAYER = re.compile(r"^\s+([0-9.]+)\s+(\S+)\s+([0-9.]+)\s+([0-9.]+)\s+\(layer only\)\s*$")


def parse(paths: list[Path]) -> dict:
    """{agent: {steps: {"choice": {rho: chose_optimal}, "probe": {(arm, rho): auc}, "layer": {(name, rho): auc}}}}"""
    out: dict = defaultdict(dict)
    agent = steps = None
    for path in paths:
        for line in path.read_text().splitlines():
            if m := HEADER.match(line):
                agent, steps = m.group(1), int(m.group(2))
                out[agent].setdefault(steps, {"choice": {}, "probe": {}, "layer": {}})
                continue
            if agent is None:
                continue
            block = out[agent][steps]
            if m := CHOICE.match(line):
                block["choice"][float(m.group(1))] = float(m.group(2)) / 100
            elif m := PROBE.match(line):
                block["probe"][(m.group(3), float(m.group(2)))] = float(m.group(4))
            elif m := LAYER.match(line):
                block["layer"][(m.group(2), float(m.group(1)))] = float(m.group(3))
    return out


def series(blocks: dict, kind: str, key_hi, key_lo):
    xs, ys = [], []
    for steps in sorted(blocks):
        table = blocks[steps][kind]
        if key_hi in table and key_lo in table:
            xs.append(steps)
            ys.append(table[key_hi] - table[key_lo])
    return np.array(xs, dtype=float), np.array(ys, dtype=float)


def arrival(xs: np.ndarray, ys: np.ndarray, settle_fraction: float = 0.5) -> tuple[float, float]:
    """(settled level, first step after which the curve stays above half of it)."""
    if len(xs) < 3:
        return float("nan"), float("nan")
    settled = float(ys[xs >= (1 - settle_fraction) * xs.max()].mean())
    if settled <= 0:
        return settled, float("nan")
    below = np.flatnonzero(ys < settled / 2)
    if not len(below):
        return settled, float(xs[0])
    later = xs[xs > xs[below[-1]]]
    return settled, float(later[0]) if len(later) else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sweeps", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, default=Path("figures"))
    parser.add_argument("--name", type=str, default="fig_early_warning")
    args = parser.parse_args()

    data = parse(args.sweeps)
    agents = sorted(data)
    if not agents:
        raise SystemExit("no '=== agent @ steps ===' blocks found")

    fig, axes = plt.subplots(1, len(agents), figsize=(4.4 * len(agents), 3.6), sharey=True, squeeze=False)
    print(f"{'agent':>22}{'checkpoints':>13}{'behaviour gap':>15}{'arrives@':>10}{'probe gap':>11}{'arrives@':>10}   first")
    for ax, agent in zip(axes[0], agents):
        blocks = data[agent]
        bx, by = series(blocks, "choice", 1.0, 0.0)
        px, py = series(blocks, "probe", ("trained", 1.0), ("trained", 0.0))
        ux, uy = series(blocks, "probe", ("untrained", 1.0), ("untrained", 0.0))
        ax.plot(bx / 1e6, by, "-o", color=ORANGE, ms=3, lw=1.8, label="behaviour: Δ chose_optimal")
        ax.plot(px / 1e6, py, "-s", color=BLUE, ms=3, lw=1.8, label="probe: Δ AUC (trained)")
        if len(ux):
            ax.plot(ux / 1e6, uy, ":", color=MUTED, lw=1.2, label="probe: Δ AUC (untrained)")
        layers = sorted({name for (name, _) in next(iter(blocks.values()))["layer"]}) if blocks else []
        for name in layers:
            lx, ly = series(blocks, "layer", (name, 1.0), (name, 0.0))
            ax.plot(lx / 1e6, ly, "-", color=BLUE, alpha=0.25, lw=0.9)
        ax.axhline(0, color=MUTED, lw=1)
        ax.set_title(agent)
        ax.set_xlabel("training steps (millions)")
        ax.grid(axis="y", alpha=0.4)

        b_level, b_at = arrival(bx, by)
        p_level, p_at = arrival(px, py)
        first = "-" if np.isnan(b_at) or np.isnan(p_at) else ("probe" if p_at < b_at else "behaviour" if b_at < p_at else "same")
        fmt = lambda v: "-" if v != v else f"{v / 1e6:.1f}M"  # noqa: E731
        print(
            f"{agent:>22}{len(blocks):>13}{b_level:>+15.3f}{fmt(b_at):>10}{p_level:>+11.3f}{fmt(p_at):>10}   {first}"
        )
    axes[0][0].set_ylabel("gap  (ρ=1.0 − ρ=0.0)")
    axes[0][0].legend(fontsize=7.5, loc="upper left")
    fig.tight_layout()
    args.out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(args.out / f"{args.name}.{ext}", bbox_inches="tight", dpi=150)
    print(f"\nwrote {args.out / args.name}.png / .pdf")
    print(
        "\narrives@ = first checkpoint after which the gap stays above half its settled level."
        "\n'first' names which of the two arrived earlier; 'probe' is what the guiding hypothesis predicts."
    )


if __name__ == "__main__":
    main()
