"""Figures for the offline-BC (route transformer) experiments.

    uv run python figures/make_bc_figures.py

Reads what the offline-BC scripts wrote into figures/data/bc/ - the per-run
``eval.csv`` copied off the volume, ``024_bc_proxy.py``'s JSON and
``026_bc_early_warning.py``'s CSVs - and draws one figure per claim. Nothing is
typed in here. Same palette and conventions as ``make_figures.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DATA = Path(__file__).parent / "data" / "bc"
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


def save(fig, name: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {name}")


def run_curves() -> dict[str, list[pd.DataFrame]]:
    """``eval.csv`` per run, grouped by training condition."""
    groups: dict[str, list[pd.DataFrame]] = {}
    for path in sorted(DATA.glob("bc11.*.eval.csv")):
        condition = path.name.split(".")[1]  # rho100 / rho050
        groups.setdefault(condition, []).append(pd.read_csv(path))
    return groups


def fig_bc_dynamics() -> None:
    """chose_optimal at rho=1 and rho=0 across training, proxy runs against control."""
    groups = run_curves()
    if not groups:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0), sharey=True)
    for ax, (condition, title) in zip(axes, (("rho100", "trained at ρ=1.0"), ("rho050", "trained at ρ=0.5 (control)"))):
        for frame in groups.get(condition, []):
            steps = frame["step"].clip(lower=1)
            ax.plot(steps, frame["rho100/chose_optimal"], color=BLUE, alpha=0.6, lw=1.2)
            ax.plot(steps, frame["rho000/chose_optimal"], color=ORANGE, alpha=0.6, lw=1.2)
            ax.plot(steps, frame["rho000/followed_feature_zero"], color=AQUA, alpha=0.5, lw=1.0, ls="--")
        ax.set_xscale("log")
        ax.set_title(title)
        ax.set_xlabel("training step")
        ax.grid(True, axis="y")
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("fraction of held-out levels")
    axes[0].text(0.02, 0.94, "chose optimal, eval ρ=1.0", color=BLUE, transform=axes[0].transAxes, fontsize=8)
    axes[0].text(0.02, 0.87, "chose optimal, eval ρ=0.0", color=ORANGE, transform=axes[0].transAxes, fontsize=8)
    axes[0].text(0.02, 0.80, "followed colour 0, eval ρ=0.0", color=AQUA, transform=axes[0].transAxes, fontsize=8)
    fig.suptitle("Route transformer trained by next-token prediction: one line per seed", y=1.02, fontsize=10)
    save(fig, "fig_bc_dynamics")


def fig_bc_rho_response() -> None:
    """Final-checkpoint chose_optimal and followed_f0 against evaluation rho."""
    path = DATA / "bc_proxy.json"
    if not path.exists():
        return
    payload = json.loads(path.read_text())
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)
    for trained, colour in ((1.0, BLUE), (0.5, AQUA)):
        runs = [r for r in payload["runs"] if r["trained_rho"] == trained]
        if not runs:
            continue
        rhos = [a["rho"] for a in runs[0]["arms"]]
        for ax, key in zip(axes, ("chose_optimal", "followed_feature_zero")):
            values = np.array([[a[key] for a in r["arms"]] for r in runs])
            mean, sd = values.mean(0), values.std(0, ddof=1) if len(values) > 1 else np.zeros(len(rhos))
            ax.errorbar(rhos, mean, yerr=sd, color=colour, marker="o", ms=4, lw=1.4, capsize=2, label=f"trained at ρ={trained}")
    for ax, title in zip(axes, ("chose the optimal objective", "went to colour 0")):
        ax.set_title(title)
        ax.set_xlabel("evaluation ρ")
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.set_ylim(0, 1.02)
        ax.grid(True, axis="y")
    axes[0].set_ylabel("fraction of held-out levels")
    axes[1].legend(loc="lower right", fontsize=8)
    save(fig, "fig_bc_rho_response")


def fig_bc_early_warning() -> None:
    """Behavioural gap against the probe's colour-route preference, per checkpoint.

    One panel per training condition, one pair of lines per seed, so whether
    the probe moves before the behaviour can be read across seeds at a glance.
    """
    files = sorted(DATA.glob("early_warning.*.csv"))
    if not files:
        return
    groups: dict[str, list[tuple[str, pd.DataFrame]]] = {}
    for path in files:
        run = path.name.replace("early_warning.", "").replace(".csv", "")
        groups.setdefault(run.split(".")[1], []).append((run, pd.read_csv(path)))
    order = [c for c in ("rho100", "rho050") if c in groups]
    fig, axes = plt.subplots(1, len(order), figsize=(3.9 * len(order), 3.0), sharey=True, squeeze=False)
    for ax, condition in zip(axes[0], order):
        for run, frame in groups[condition]:
            steps = frame["step"].clip(lower=1)
            gap = frame["chose_optimal_rho1"] - frame["chose_optimal_rho0"]
            delta = frame["auc_rho0_colour0"] - frame["auc_rho0_optimal"]
            ax.plot(steps, gap, color=ORANGE, lw=1.2, marker="o", ms=2.5, alpha=0.75)
            ax.plot(steps, delta, color=BLUE, lw=1.2, marker="s", ms=2.5, alpha=0.75)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xscale("log")
        ax.set_title({"rho100": "trained at ρ=1.0", "rho050": "trained at ρ=0.5 (control)"}[condition] + f"  ({len(groups[condition])} seeds)")
        ax.set_xlabel("training step")
        ax.grid(True, axis="y")
    axes[0][0].set_ylabel("gap")
    axes[0][0].text(0.02, 0.94, "behaviour: optimal@ρ=1 − optimal@ρ=0", color=ORANGE, transform=axes[0][0].transAxes, fontsize=8)
    axes[0][0].text(0.02, 0.87, "probe: AUC(colour-0 route) − AUC(optimal route) at ρ=0", color=BLUE, transform=axes[0][0].transAxes, fontsize=8)
    save(fig, "fig_bc_early_warning")


def fig_bc_value_axis() -> None:
    """Leave-one-out written exchange rate against the arm's own, per sweep, seeds overlaid."""
    files = sorted(DATA.glob("value_axis.*.json"))
    if not files:
        return
    by_sweep: dict[str, list[dict]] = {}
    for path in files:
        payload = json.loads(path.read_text())
        if payload.get("behaviour", {}).get("arms"):
            by_sweep.setdefault(payload["sweep"], []).append(payload)
    if not by_sweep:
        return
    sweeps = sorted(by_sweep)
    fig, axes = plt.subplots(1, len(sweeps), figsize=(3.9 * len(sweeps), 3.2), sharey=True, squeeze=False)
    for ax, sweep in zip(axes[0], sweeps):
        for k, payload in enumerate(by_sweep[sweep]):
            arms = payload["behaviour"]["arms"]
            offsets = np.array([float(o) for o in arms])
            order = np.argsort(offsets)
            keys = [list(arms)[i] for i in order]
            offsets = offsets[order]
            expected = np.array([arms[key]["expected"] for key in keys])
            fine = np.array([arms[key]["arm"]["indifference"] for key in keys])
            written = np.array([arms[key]["written"]["indifference"] for key in keys])
            if k == 0:
                ax.plot(offsets, expected, color=MUTED, lw=1.0, ls=":", label="expert")
            ax.plot(offsets, fine, color=ORANGE, marker="o", ms=3, lw=1.0, alpha=0.7, label="fine-tuned arm" if k == 0 else None)
            ax.plot(offsets, written, color=BLUE, marker="s", ms=3, lw=1.0, alpha=0.7, label="written, arm held out" if k == 0 else None)
            base = payload["behaviour"].get("base", {}).get("indifference")
            if base is not None and np.isfinite(base):
                ax.axhline(base, color=INK2, lw=0.6, ls="--", alpha=0.5)
        ax.set_title({"o0": "colour 0's value moved", "o1": "colour 1's value moved"}.get(sweep, sweep) + f"  ({len(by_sweep[sweep])} seeds)")
        ax.set_xlabel("offset from the base value")
        ax.grid(True, axis="y")
    axes[0][0].set_ylabel("exchange rate: extra steps walked for colour 0")
    axes[0][0].legend(loc="best", fontsize=8)
    fig.suptitle("Route model (values hidden): writing a value along a fitted weight-space axis", y=1.02, fontsize=10)
    save(fig, "fig_bc_value_axis")


if __name__ == "__main__":
    fig_bc_dynamics()
    fig_bc_rho_response()
    fig_bc_early_warning()
    fig_bc_value_axis()
