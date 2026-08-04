"""Figures for the goal-misgeneralisation maze experiments.

    uv run python figures/make_figures.py

Reads run metrics from figures/data/*.csv and behavioural measurements from
figures/data/measured.json (which records its own provenance). Regenerates
every figure in place.

One figure per claim. Palette is the validated categorical default
(#2a78d6 / #eb6834 / #1baf7a); aqua's contrast WARN is relieved by direct
labels on every series, per the validator's requirement.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.use("Agg")
import matplotlib.pyplot as plt

DATA = Path(__file__).parent / "data"
OUT = Path(__file__).parent
OUT.mkdir(exist_ok=True)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#9a9992"
SURFACE = "#fcfcfb"

mpl.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "text.color": INK,
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.edgecolor": MUTED,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": "#e6e5e0",
        "grid.linewidth": 0.8,
        "legend.frameon": False,
        "figure.dpi": 160,
    }
)

M = json.loads((DATA / "measured.json").read_text())


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}.png / .pdf")


# ---------------------------------------------------------------- figure 1
def fig_task():
    """What the task is and what the agent actually sees."""
    from goalmisgen.envs.observation import ObservationEncoder
    from goalmisgen.envs.rendering import render
    from goalmisgen.envs.sampling import MazeLevelSampler
    from goalmisgen.envs.solver import objective_distances

    rng = np.random.default_rng(3)
    level = MazeLevelSampler(size_range=(11, 11)).sample(rng)
    d = objective_distances(level)
    vals = [o.value for o in level.objectives]
    util = [v - 0.05 * dd for v, dd in zip(vals, d)]
    best = int(np.argmax(util))

    enc = ObservationEncoder(max_size=11, n_features=2)
    obs = enc.encode(level, level.agent_start)

    fig = plt.figure(figsize=(9.4, 3.1))
    gs = fig.add_gridspec(1, 6, width_ratios=[1.55, 1, 1, 1, 1, 1], wspace=0.18)

    ax = fig.add_subplot(gs[0])
    ax.imshow(render(level, level.agent_start, cell_pixels=14))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("rendered for humans only", color=INK2, fontsize=8.5, pad=4)
    sub = (
        f"blue = agent   ·   red = feature 0 (value {vals[0]:.1f}, {d[0]} steps away)   ·   "
        f"green = feature 1 (value {vals[1]:.1f}, {d[1]} steps away)   →   "
        f"optimal is feature {best}, by value − 0.05 × distance"
    )
    fig.text(0.5, -0.04, sub, ha="center", fontsize=8, color=INK2)

    names = ["ch0 walls", "ch1 agent", "ch2 feature 0", "ch3 feature 1", "ch4 value"]
    for i, nm in enumerate(names):
        a = fig.add_subplot(gs[i + 1])
        a.imshow(obs[:, :, i], cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
        a.set_xticks([])
        a.set_yticks([])
        a.set_title(nm, color=INK2, fontsize=8, pad=4)
        for s in a.spines.values():
            s.set_visible(True)
            s.set_color(MUTED)
            s.set_linewidth(0.6)

    fig.suptitle("The agent sees five symbolic channels — no colour, no pixels", y=1.0, fontsize=10.5, color=INK)
    save(fig, "fig1_task")


# ---------------------------------------------------------------- figure 2
def fig_rho_response():
    """The headline: what each agent was actually pursuing."""
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.3), sharey=True)
    for ax, key, title, sub in zip(
        axes,
        ["5x5", "11x11"],
        ["5×5 — proxy is free", "11×11 — proxy costs ~20%"],
        ["10M steps", "150M steps"],
    ):
        d = M["rho_response"][key]
        rho = d["rho"]
        for vals, colour, label in (
            (d["chose_optimal"], BLUE, "chose optimal"),
            (d["followed_f0"], ORANGE, "followed feature 0"),
        ):
            ax.plot(rho, vals, "-o", color=colour, lw=2, ms=8, label=label, clip_on=False, zorder=3)
            for x, y in zip(rho, vals):
                ax.annotate(
                    f"{y:.0f}%", (x, y), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=7.5, color=INK2
                )
        ax.axhline(50, color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)
        ax.set_xticks(rho)
        ax.set_xlim(1.06, -0.06)
        ax.set_ylim(-6, 112)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.grid(axis="y", zorder=0)
        ax.set_xlabel("ρ  (P colour marks the higher-value objective)")
        ax.set_title(title, color=INK, pad=18)
        ax.text(0.5, 1.02, sub, transform=ax.transAxes, ha="center", fontsize=7.5, color=MUTED)

    axes[0].set_ylabel("% of episodes")
    axes[1].text(-0.03, 50, "chance ", va="center", ha="left", fontsize=7, color=MUTED)
    axes[1].legend(loc="lower left", fontsize=8)
    fig.suptitle(
        "Trained at ρ=1.0, tested across ρ:  the 5×5 agent tracks colour, not value", y=1.09, fontsize=10.5, color=INK
    )
    save(fig, "fig2_rho_response")


# ---------------------------------------------------------------- figure 3
def fig_margin():
    """Is the choice a graded comparison or a rough rule?"""
    d = M["margin"]
    x = np.arange(len(d["bins"]))
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for vals, colour, label, dy in (
        (d["rho1"], BLUE, "ρ = 1.0  (colour agrees with value)", 10),
        (d["rho0"], ORANGE, "ρ = 0.0  (colour inverted)", -15),
    ):
        ax.plot(x, vals, "-o", color=colour, lw=2, ms=8, label=label, clip_on=False, zorder=3)
        for xi, y in zip(x, vals):
            ax.annotate(
                f"{y:.0f}%", (xi, y), textcoords="offset points", xytext=(0, dy), ha="center", fontsize=7.5, color=INK2
            )
    ax.annotate("", xy=(3, d["rho1"][3]), xytext=(3, d["rho0"][3]), arrowprops=dict(arrowstyle="<->", color=MUTED, lw=1))
    ax.annotate("27 pts", (3.08, 85), ha="left", fontsize=8, color=INK2)
    ax.annotate("what colour costs\non an easy decision", (3.08, 74), ha="left", fontsize=7.5, color=MUTED)
    ax.axhline(50, color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.set_xlim(-0.35, 4.35)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b}\nn={n}" for b, n in zip(d["bins"], d["n"])], fontsize=8)
    ax.set_ylim(-6, 116)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.grid(axis="y", zorder=0)
    ax.set_xlabel("utility margin  (how clear-cut the decision was)")
    ax.set_ylabel("% chose optimal")
    ax.set_title("The comparison is clean; colour is an additive contaminant", color=INK, pad=8)
    ax.legend(loc="upper left", fontsize=8)
    save(fig, "fig3_margin")


# ---------------------------------------------------------------- figure 4
def fig_dynamics():
    """When misgeneralisation appears — and that it needs a proxy to appear at all."""
    proxy = pd.read_csv(DATA / "maze11.csv", index_col=0).sort_index()
    control = pd.read_csv(DATA / "clean11.csv", index_col=0).sort_index()

    def arm(df, name):
        return df[f"{name}/00_episode_returns"].dropna()

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(6.6, 5.2), sharex=True, gridspec_kw={"height_ratios": [1, 1], "hspace": 0.18}
    )

    # Top: the three arms of the proxy run. They coincide until competence appears.
    for name, colour, label in (("rho100", BLUE, "ρ = 1.0"), ("rho050", AQUA, "ρ = 0.5"), ("rho000", ORANGE, "ρ = 0.0")):
        s_ = arm(proxy, name)
        ax1.plot(s_.index / 1e6, s_.values, "-", color=colour, lw=2, label=label, zorder=3)
    ax1.axhline(M["optimum"]["11x11_fixed_values"], color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
    ax1.annotate(
        "optimum",
        (2, M["optimum"]["11x11_fixed_values"]),
        textcoords="offset points",
        xytext=(0, 5),
        fontsize=7.5,
        color=MUTED,
    )
    ax1.axvspan(15, 20, color="#e6e5e0", alpha=0.7, zorder=0)
    ax1.annotate("competence\nappears", (17.5, -4.6), ha="center", fontsize=7.5, color=INK2)
    ax1.grid(axis="y", zorder=0)
    ax1.set_ylabel("evaluation return")
    ax1.legend(loc="lower right", fontsize=8, ncol=3)
    ax1.set_title("trained with a proxy available (ρ = 1.0)", color=INK, fontsize=9.5, pad=6)

    # Bottom: the gap. This is the claim, and it is invisible in the panel above.
    for df, colour, label in (
        (proxy, BLUE, "trained at ρ = 1.0  (proxy available)"),
        (control, ORANGE, "trained at chance  (control: no proxy)"),
    ):
        hi, lo = arm(df, "rho100"), arm(df, "rho000")
        idx = hi.index.intersection(lo.index)
        gap = hi.loc[idx] - lo.loc[idx]
        ax2.plot(idx / 1e6, gap.values, "-o", color=colour, lw=2, ms=4, label=label, zorder=3)
    ax2.axhline(0, color=MUTED, lw=1, zorder=1)
    ax2.axvspan(0, 20, color="#e6e5e0", alpha=0.55, zorder=0)
    ax2.annotate("before competence:\ngap is meaningless", (10, -0.135), ha="center", fontsize=7.5, color=MUTED)
    ax2.grid(axis="y", zorder=0)
    ax2.set_xlabel("training steps (millions)")
    ax2.set_ylabel("gap  (ρ=1.0 − ρ=0.0)")
    ax2.legend(loc="lower right", fontsize=8)
    ax2.text(50, 0.02, "control still running", fontsize=7, color=MUTED, ha="left")
    ax2.set_title("the misgeneralisation signal", color=INK, fontsize=9.5, pad=6)

    fig.suptitle("Misgeneralisation appears after competence, and only when a proxy exists", y=0.97, fontsize=10.5, color=INK)
    save(fig, "fig4_dynamics")


if __name__ == "__main__":
    print("writing figures to", OUT)
    fig_task()
    fig_rho_response()
    fig_margin()
    fig_dynamics()
