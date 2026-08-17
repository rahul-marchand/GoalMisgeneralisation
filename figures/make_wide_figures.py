"""The threshold-against-value plot, on the wide grid.

    uv run python figures/make_wide_figures.py

The upgrade to ``fig10`` is not that there are more points. The old grid fitted
seven arms over a gap of 0.1 to 0.7 and then *wrote* the axis out to a gap of
-0.7 to +1.3, so the interesting half of that figure was extrapolation. The wide
grid trains arms across gap 0.05 to 0.95, which means most of what used to be
extrapolated is now measured, and the two can be compared directly.

Reads JSON and never has a number typed into it, so a re-measurement cannot leave
a stale annotation behind — same rule as ``make_figures.py``, whose palette and
surface this borrows rather than inventing a second look for one project.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

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
        "axes.labelsize": 9.5,
        "axes.titlesize": 10,
        "axes.edgecolor": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.labelcolor": INK,
        "legend.frameon": False,
    }
)


def optimal_threshold(v0: np.ndarray, v1: np.ndarray, penalty: float, gamma: float) -> np.ndarray:
    """Extra steps worth walking for colour 0 rather than colour 1.

    Setting the two discounted returns equal gives g^D (c + v0) = c + v1 with
    c = penalty/(1-g), so D = log((c+v1)/(c+v0))/log g. The distance already
    walked cancels: the threshold depends on the two values alone.

    Signed, and relative to colour 0 specifically. Every arm is scored on one
    held-out set at the *base* values, where colour 0 is the richer objective, so
    that is what ``value_distance_decisions`` measures against for every arm
    whatever it was trained on. Negative therefore means something real -- the
    agent should prefer colour 1 -- and an absolute value here would fold the
    arms trained past the flip back up on top of the ones before it, hiding the
    one thing they were trained to show.
    """
    c = penalty / (1 - gamma)
    return np.log((c + v1) / (c + v0)) / np.log(gamma)


def band(ax, points, colour, label, marker="o"):
    v = np.array([p["value"] for p in points])
    s = np.array([p["steps"] for p in points])
    lo = np.array([p["lo"] for p in points])
    hi = np.array([p["hi"] for p in points])
    ax.fill_between(v, lo, hi, color=colour, alpha=0.18, lw=0, zorder=2)
    ax.plot(v, s, marker, color=colour, ms=4.2, lw=1.6, ls="-", mec=SURFACE, mew=0.7, zorder=3, label=label)
    return v, s


def main() -> None:
    d = json.loads((DATA / "wide_value_axis.json").read_text())
    entries = [e for e in d["series"] if e.get("trained")]
    # x sweeps extend an o sweep past the flip; they belong on its panel, not
    # on one of their own.
    past = {(e["agent"], "o" + e["sweep"][1:]): e for e in entries if e["sweep"].startswith("x")}
    entries = [e for e in entries if not e["sweep"].startswith("x")]
    if not entries:
        raise SystemExit("no series with trained arms — run figures/extract_wide_axis.py first")

    cols = 2 if len(entries) > 1 else 1
    rows = -(-len(entries) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5.9 * cols, 4.1 * rows), squeeze=False, sharey=True)
    flat = [a for row in axes for a in row]
    for ax in flat[len(entries) :]:
        ax.set_visible(False)

    for ax, entry in zip(flat, entries):
        other = d["other_objective"][entry["sweep"]]
        trained = entry["trained"]
        v = np.array([p["value"] for p in trained])

        swept = int(entry["sweep"][1:])
        beyond = past.get((entry["agent"], entry["sweep"]))
        span = list(v) + ([p["value"] for p in beyond["trained"]] if beyond else [])
        grid = np.linspace(min(span) - 0.02, max(span) + 0.02, 300)
        # The swept objective takes the grid value; the other keeps its fixed one.
        v0, v1 = (grid, np.full_like(grid, other)) if swept == 0 else (np.full_like(grid, other), grid)
        opt = optimal_threshold(v0, v1, d["step_penalty"], d["discount"])
        ax.plot(grid, opt, color=MUTED, lw=1.1, ls="--", zorder=1)
        peak = int(np.argmax(np.abs(opt)))
        ax.annotate(
            "optimal", xy=(grid[peak], opt[peak]), xytext=(8, -10), textcoords="offset points", color=MUTED, fontsize=8
        )
        ax.axhline(0, color=INK2, lw=0.8, alpha=0.45, zorder=1)

        base = d["base_exchange_rate"].get(entry["agent"])
        if base is not None:
            ax.axhline(base, color=MUTED, lw=0.7, ls=":", zorder=0)

        band(ax, trained, BLUE, "trained: one fine-tune per value")
        if beyond:
            # Plotted below the axis because the estimator is unsigned: a logistic
            # whose slope flips with the preference returns the mirror of the
            # crossing, not a negative one. These arms were trained with the swept
            # colour worth MORE than the other, so the sign is known from the
            # training values even though the measurement cannot carry it.
            b = beyond["trained"]
            v_b = np.array([p["value"] for p in b])
            s_b = -np.array([p["steps"] for p in b])
            ax.fill_between(
                v_b, -np.array([p["hi"] for p in b]), -np.array([p["lo"] for p in b]), color=BLUE, alpha=0.18, lw=0, zorder=2
            )
            ax.plot(
                v_b,
                s_b,
                "D-",
                color=BLUE,
                ms=4.8,
                lw=1.6,
                mfc=SURFACE,
                mec=BLUE,
                mew=1.5,
                zorder=3,
                label="trained past the flip (held out of the fit)",
            )
        # The main axis extrapolated past the flip: same orange as the other
        # written points, same diamond as the other past-the-flip points. Shape
        # says which regime, colour says trained or written.
        # Only where there are trained arms past the flip to compare against.
        # 014's --extrapolate defaults to [1.1, 0.2], so every sweep carries a
        # stray point on the far side of the crossing; plotting those put an
        # extrapolation on three panels where nothing was ever trained there, and
        # a written point with nothing to check it against says nothing.
        unseen = (entry.get("written_unseen") or []) if beyond else []
        flipped_side = [r for r in unseen if (r["value"] > other if swept == 1 else r["value"] < other)]
        if flipped_side:
            v_u = np.array([r["value"] for r in flipped_side])
            s_u = -np.array([r["steps"] for r in flipped_side])
            ax.plot(
                v_u,
                s_u,
                "D--",
                color=ORANGE,
                ms=4.8,
                lw=1.4,
                mfc=SURFACE,
                mec=ORANGE,
                mew=1.5,
                zorder=4,
                label="axis written past the flip (extrapolated)",
            )

        if entry.get("written_heldout"):
            band(ax, entry["written_heldout"], ORANGE, "predicted: axis written in, that value held out", marker="s")

        n = entry["n"]
        seed = entry["agent"].split(".")[-1]
        ax.set_title(
            f"colour {swept} swept, colour {1 - swept} fixed at {other:g}"
            f"\nseed {seed.lstrip('s')}  ·  {n.get('trained', 0)} arms",
            color=INK,
        )
        ax.set_xlabel(f"what colour {swept} is worth")
        ax.grid(axis="y", color=MUTED, alpha=0.22, lw=0.6)
        ax.set_axisbelow(True)

    for row in axes:
        row[0].set_ylabel("extra steps walked for the richer objective")
    richest = max(flat[: len(entries)], key=lambda a: len(a.get_legend_handles_labels()[0]))
    handles, labels = richest.get_legend_handles_labels()
    flat[0].legend(handles, labels, loc="upper center", fontsize=8.5)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig13_wide_threshold.{ext}", dpi=200, bbox_inches="tight")
    print(f"wrote {OUT / 'fig13_wide_threshold.png'}")

    for entry in entries:
        t = entry["trained"]
        print(
            f"  {entry['agent']} {entry['sweep']}: values {t[0]['value']:.2f}-{t[-1]['value']:.2f}, "
            f"steps {t[-1]['steps']:.1f}-{t[0]['steps']:.1f}, "
            f"min reach {min(p['reach'] for p in t):.1f}%"
        )


if __name__ == "__main__":
    main()
