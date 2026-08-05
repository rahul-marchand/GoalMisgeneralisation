"""Figures for the goal-misgeneralisation maze experiments.

    uv run python figures/make_figures.py

Reads whatever experiments/002_measure_proxy.py wrote into figures/data/*.json
and the training metrics in figures/data/*.csv. Every number on every figure
comes from those files — nothing is typed in here, because a hand-copied
annotation survives a re-measurement that invalidates it.

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

AGENTS = {
    name: json.loads((DATA / f"{name}.json").read_text())
    for name in ("smoke5b", "maze11", "clean11", "clean11fv")
}


def arms(agent: str, field: str) -> tuple[list[float], list[float]]:
    """A measured field against ρ, sorted so the line is drawn in order."""
    ordered = sorted(AGENTS[agent]["arms"], key=lambda arm: -arm["rho"])
    return [arm["rho"] for arm in ordered], [100 * arm[field] for arm in ordered]


def arm_at(agent: str, rho: float) -> dict:
    for arm in AGENTS[agent]["arms"]:
        if arm["rho"] == rho:
            return arm
    raise KeyError(f"{agent} has no arm at rho={rho}")


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
    """The headline: what each agent was actually pursuing.

    The control panel is what makes the other two readable. Without it, a
    falling ``chose_optimal`` could be any weakness of the agent; beside an
    agent that stays flat, the proxy is the explanation left standing.

    The control shown is single-variable: same maze size, same value scheme,
    same level dataset and split as the proxy run, differing only in the
    training correlation. A second control trained on randomised values
    (``clean11``) agrees, but confounds the correlation with the value scheme
    and so is not what the claim rests on.
    """
    panels = [
        ("smoke5b", "5×5 — proxy is free", "trained ρ=1.0"),
        ("maze11", "11×11 — proxy costs a detour", "trained ρ=1.0"),
        ("clean11fv", "11×11 control — no proxy to learn", "trained ρ=0.5"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.4), sharey=True)
    for ax, (agent, title, sub) in zip(axes, panels):
        # One series is labelled above its markers and the other below, so the
        # two never collide where the curves meet.
        for field, colour, label, dy in (
            ("chose_optimal", BLUE, "chose optimal", 10),
            ("followed_feature_zero", ORANGE, "followed feature 0", -16),
        ):
            rho, vals = arms(agent, field)
            ax.plot(rho, vals, "-o", color=colour, lw=2, ms=8, label=label, clip_on=False, zorder=3)
            for x, y in zip(rho, vals):
                ax.annotate(
                    f"{y:.0f}%", (x, y), textcoords="offset points", xytext=(0, dy), ha="center", fontsize=7.5, color=INK2
                )
        ax.axhline(50, color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)
        ax.set_xticks(sorted({arm["rho"] for arm in AGENTS[agent]["arms"]}))
        ax.set_xlim(1.06, -0.06)
        ax.set_ylim(-12, 114)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.grid(axis="y", zorder=0)
        ax.set_title(title, color=INK, pad=18)
        ax.text(0.5, 1.02, sub, transform=ax.transAxes, ha="center", fontsize=7.5, color=MUTED)

    axes[0].set_ylabel("% of episodes")
    axes[0].annotate("chance", (1.04, 50), va="bottom", ha="left", fontsize=7, color=MUTED)
    axes[1].legend(loc="lower center", fontsize=8)
    fig.supxlabel("ρ  at test time   (P that colour marks the higher-value objective)", y=-0.04, fontsize=9, color=INK2)
    fig.suptitle(
        "Held out from training, across ρ:  only the agents that could learn a proxy lose optimality",
        y=1.08,
        fontsize=10.5,
        color=INK,
    )
    save(fig, "fig2_rho_response")


# ---------------------------------------------------------------- figure 3
def fig_margin():
    """Where the choice breaks down — a close call, or everywhere?

    The control is the reference profile: accuracy rises with the margin and
    is unmoved by ρ, which is what a value comparison with a little noise looks
    like. The proxy agent matches it at ρ=1.0 and falls apart at ρ=0.0, and the
    damage is worst in the middle, not on the close calls.

    Both panels are the same levels in the same bins — the two runs differ only
    in training correlation — so the profiles are directly comparable.
    """
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.5), sharey=True)
    panels = [("clean11fv", "control — trained at ρ = 0.5"), ("maze11", "trained with a proxy, ρ = 1.0")]
    for ax, (agent, title) in zip(axes, panels):
        labels = arm_at(agent, 1.0)["margin"]["bins"]
        x = np.arange(len(labels))
        for rho, colour, dy in ((1.0, BLUE, 10), (0.0, ORANGE, -15)):
            band = arm_at(agent, rho)["margin"]
            vals = [100 * v for v in band["chose_optimal"]]
            label = f"ρ = {rho:.1f}" + ("  (colour agrees with value)" if rho else "  (colour inverted)")
            ax.plot(x, vals, "-o", color=colour, lw=2, ms=8, label=label, clip_on=False, zorder=3)
            for xi, y in zip(x, vals):
                if np.isfinite(y):
                    ax.annotate(
                        f"{y:.0f}%", (xi, y), textcoords="offset points", xytext=(0, dy), ha="center", fontsize=7.5, color=INK2
                    )
        counts = arm_at(agent, 1.0)["margin"]["n"]
        ax.axhline(50, color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)
        ax.set_xlim(-0.35, len(labels) - 0.65)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{b}\nn={n}" for b, n in zip(labels, counts)], fontsize=8)
        ax.set_ylim(-6, 116)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.grid(axis="y", zorder=0)
        ax.set_title(title, color=INK, pad=18)
        scheme = "randomised values" if AGENTS[agent]["randomise_values"] else "fixed values"
        ax.text(0.5, 1.02, scheme, transform=ax.transAxes, ha="center", fontsize=7.5, color=MUTED)

    # The worst-hit band, named from the data rather than asserted.
    hi = np.array(arm_at("maze11", 1.0)["margin"]["chose_optimal"], dtype=float)
    lo = np.array(arm_at("maze11", 0.0)["margin"]["chose_optimal"], dtype=float)
    worst = int(np.nanargmax(hi - lo))
    axes[1].annotate(
        "",
        xy=(worst, 100 * hi[worst]),
        xytext=(worst, 100 * lo[worst]),
        arrowprops=dict(arrowstyle="<->", color=MUTED, lw=1),
    )
    axes[1].annotate(
        f"{100 * (hi[worst] - lo[worst]):.0f} pts",
        (worst + 0.12, 100 * (hi[worst] + lo[worst]) / 2),
        ha="left",
        va="center",
        fontsize=8,
        color=INK2,
    )
    axes[0].set_ylabel("% chose optimal")
    axes[0].legend(loc="lower right", fontsize=8)
    fig.supxlabel("utility margin  (how clear-cut the decision was)", y=-0.06, fontsize=9, color=INK2)
    fig.suptitle(
        "The proxy does not just add noise — it captures the decisions that should be easy",
        y=1.06,
        fontsize=10.5,
        color=INK,
    )
    save(fig, "fig3_margin")


# ---------------------------------------------------------------- figure 4
def fig_dynamics():
    """When misgeneralisation appears — and that it needs a proxy to appear at all.

    These curves come from cleanba's own in-training evaluation. The two older
    runs predate the seeding fix and are scored on a smaller batch of levels
    than a current run would use; ``clean11fv`` was trained after it. The *gap*
    between arms is unaffected either way, because both arms of a run see the
    same levels whatever those levels are — but absolute heights are not
    comparable across runs or with figures 2 and 3, so no optimum reference is
    drawn here.
    """
    proxy = pd.read_csv(DATA / "maze11.csv", index_col=0).sort_index()
    control = pd.read_csv(DATA / "clean11fv.csv", index_col=0).sort_index()

    def arm(df, name):
        return df[f"{name}/00_episode_returns"].dropna()

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(6.6, 5.2), sharex=True, gridspec_kw={"height_ratios": [1, 1], "hspace": 0.18}
    )

    # Top: the three arms of the proxy run. They coincide until competence appears.
    for name, colour, label in (("rho100", BLUE, "ρ = 1.0"), ("rho050", AQUA, "ρ = 0.5"), ("rho000", ORANGE, "ρ = 0.0")):
        s_ = arm(proxy, name)
        ax1.plot(s_.index / 1e6, s_.values, "-", color=colour, lw=2, label=label, zorder=3)
    ax1.axvspan(15, 20, color="#e6e5e0", alpha=0.7, zorder=0)
    ax1.annotate("competence\nappears", (21, -3.0), ha="left", fontsize=7.5, color=INK2)
    # A linear axis is dominated by the -5.5 to +0.3 jump and hides the thing
    # the panel is for: that the arms coincide before it and separate after.
    ax1.set_yscale("symlog", linthresh=0.05, linscale=0.4)
    ax1.set_yticks([-5, -1, 0, 0.1, 0.3])
    ax1.get_yaxis().set_major_formatter(mpl.ticker.FormatStrFormatter("%g"))
    ax1.grid(axis="y", zorder=0)
    ax1.set_ylabel("evaluation return")
    ax1.legend(loc="lower right", fontsize=8, ncol=3)
    ax1.set_title("trained with a proxy available (ρ = 1.0)", color=INK, fontsize=9.5, pad=6)

    # Bottom: the gap. This is the claim, and it is invisible in the panel above.
    for df, colour, label in (
        (proxy, BLUE, "trained at ρ = 1.0  (proxy available)"),
        (control, ORANGE, "trained at ρ = 0.5  (control: no proxy)"),
    ):
        hi, lo = arm(df, "rho100"), arm(df, "rho000")
        idx = hi.index.intersection(lo.index)
        gap = hi.loc[idx] - lo.loc[idx]
        ax2.plot(idx / 1e6, gap.values, "-o", color=colour, lw=2, ms=4, label=label, zorder=3)
    ax2.axhline(0, color=MUTED, lw=1, zorder=1)
    ax2.axvspan(0, 20, color="#e6e5e0", alpha=0.55, zorder=0)
    ax2.set_ylim(-0.24, 0.24)
    ax2.annotate("before competence:\ngap is meaningless", (10, -0.225), ha="center", va="bottom", fontsize=7.5, color=MUTED)
    ax2.grid(axis="y", zorder=0)
    ax2.set_xlabel("training steps (millions)")
    ax2.set_ylabel("gap  (ρ=1.0 − ρ=0.0)")
    ax2.legend(loc="lower right", fontsize=8)
    ax2.set_title("the misgeneralisation signal", color=INK, fontsize=9.5, pad=6)

    fig.suptitle("Misgeneralisation appears after competence, and only when a proxy exists", y=0.97, fontsize=10.5, color=INK)
    save(fig, "fig4_dynamics")


if __name__ == "__main__":
    print("writing figures to", OUT)
    fig_task()
    fig_rho_response()
    fig_margin()
    fig_dynamics()
