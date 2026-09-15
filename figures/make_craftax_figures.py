"""Figures for the Craftax stream, one per claim, from what the chains wrote.

    uv run python figures/make_craftax_figures.py [--data figures/data/craftax] [--out figures/craftax]

Reads, per task directory (``ore`` for stage 2, ``craft`` for stage 4):

- ``<base>.eval.csv``  - the trainer's evaluation log, copied off the volume
- ``value_axis.<base>.o{0,1}.json`` - ``027_bc_value_axis.py``'s output
- ``thresholds.json`` - the task's cost-gap distribution and the arm thresholds,
  written by ``--thresholds`` from a local demonstration set

and draws: the training dynamics (competence and the proxy readout at three
correlations), the value-axis figure (fine-tuned against written exchange
rates, with the axis statistics), and the measurement window of each task
(where the thresholds sit in the cost-gap distribution). Nothing is typed in.
Same palette as ``make_bc_figures.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6f5f9c"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#9a9992"
SURFACE = "#fcfcfb"
TASKS = {"ore": "stage 2: ore field (maze task in the engine)", "craft": "stage 4: crafting chains"}
KIND = {"ore": ("coal", "diamond"), "craft": ("iron", "coal")}

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


def save(fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out / name}.png")


# ----------------------------------------------------------------------
# Dynamics: competence and the proxy readout
# ----------------------------------------------------------------------


def fig_dynamics(data: Path, out: Path, task: str) -> None:
    files = sorted((data / task).glob("*.eval.csv"))
    if not files:
        return
    runs = {path.name.replace(".eval.csv", ""): pd.read_csv(path) for path in files}
    panels = [
        ("rho100/reached", "reached an objective", "rho = 1.0"),
        ("rho100/chose_optimal", "chose the optimal objective", "rho = 1.0"),
        ("rho100/followed_feature_zero", f"took {KIND[task][0]}", "rho = 1.0"),
        ("rho050/followed_feature_zero", f"took {KIND[task][0]}", "rho = 0.5 (kind uninformative)"),
        ("rho000/followed_feature_zero", f"took {KIND[task][0]}", "rho = 0.0 (kind reversed)"),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.0 * len(panels), 2.7), sharex=True)
    for ax, (column, label, sub) in zip(axes, panels):
        for k, (name, df) in enumerate(runs.items()):
            if column not in df:
                continue
            steps = df["step"].to_numpy()
            ax.plot(np.maximum(steps, 1), df[column].to_numpy(), color=BLUE, lw=1.0, alpha=0.5 + 0.5 * (k == 0), label=name)
        ax.set_xscale("log")
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f"{label}\n{sub}", fontsize=9)
        ax.grid(True, axis="y")
        ax.set_xlabel("training step")
    axes[0].set_ylabel("fraction of held-out episodes")
    axes[0].legend(fontsize=7, loc="lower right")
    fig.suptitle(f"{TASKS[task]}: route model trained by imitation, values hidden, one line per seed", y=1.06, fontsize=10)
    save(fig, out, f"fig_{task}_dynamics")


# ----------------------------------------------------------------------
# Value axis
# ----------------------------------------------------------------------


def fig_value_axis(data: Path, out: Path, task: str) -> None:
    payloads = [json.loads(p.read_text()) for p in sorted((data / task).glob("value_axis.*.json"))]
    payloads = [p for p in payloads if p.get("behaviour", {}).get("arms")]
    if not payloads:
        return
    sweeps = sorted({p["sweep"] for p in payloads})
    gaps = [json.loads(p.read_text()) for p in sorted((data / task).glob("value_or_gap.*.json"))]
    fig, axes = plt.subplots(1, len(sweeps), figsize=(4.2 * len(sweeps), 3.3), sharey=True, squeeze=False)
    kinds = KIND[task]
    for ax, sweep in zip(axes[0], sweeps):
        group = [p for p in payloads if p["sweep"] == sweep]
        stats = []
        for k, payload in enumerate(group):
            arms = payload["behaviour"]["arms"]
            keys = sorted(arms, key=float)
            offsets = np.array([float(o) for o in keys])
            expected = np.array([arms[key]["expected"] for key in keys])
            fine = np.array([arms[key]["arm"]["indifference"] for key in keys])
            written = np.array([arms[key]["written"]["indifference"] for key in keys])
            if k == 0:
                ax.plot(offsets, expected, color=MUTED, lw=1.0, ls=":", label="expert")
            ax.plot(
                offsets, fine, color=ORANGE, marker="o", ms=3, lw=1.0, alpha=0.7, label="fine-tuned arm" if k == 0 else None
            )
            ax.plot(
                offsets,
                written,
                color=BLUE,
                marker="s",
                ms=3,
                lw=1.0,
                alpha=0.7,
                label="written, arm held out" if k == 0 else None,
            )
            base = payload["behaviour"].get("base", {}).get("indifference")
            if base is not None and np.isfinite(base):
                ax.axhline(base, color=INK2, lw=0.6, ls="--", alpha=0.5)
            stats.append(
                (payload.get("cos_opposite_raw"), payload.get("reliability"), payload["behaviour"].get("loo_error_mean_abs"))
            )
        objective = int(sweep[1:])
        ax.set_title(f"{kinds[objective]}'s value moved  ({len(group)} seed{'s' if len(group) != 1 else ''})")
        ax.set_xlabel("offset from the base value")
        ax.grid(True, axis="y")
        rel = [s[0] for s in stats if s[0] is not None]
        err = [s[1] for s in stats if s[1] is not None]
        text = []
        if rel:
            text.append(f"split-half axis reliability = {np.mean(rel):.2f}")
        if err:
            text.append(f"held-out write error = {np.mean(err):.1f} actions")
        cos = [g["cos"] for g in gaps if g.get("cos") is not None]
        dis = [g["cos_disattenuated"] for g in gaps if g.get("cos_disattenuated") is not None]
        if cos:
            text.append(
                f"cos(axis_0, axis_1) = {np.mean(cos):+.2f}" + (f"  ({np.mean(dis):+.2f} disattenuated)" if dis else "")
            )
        ax.text(0.02, 0.97, "\n".join(text), transform=ax.transAxes, va="top", fontsize=7.5, color=INK2)
    axes[0][0].set_ylabel(f"exchange rate: extra actions spent for {kinds[0]}")
    axes[0][0].legend(loc="lower right", fontsize=7.5)
    fig.suptitle(f"{TASKS[task]}: a value written along one fitted weight-space axis", y=1.02, fontsize=10)
    save(fig, out, f"fig_{task}_value_axis")


# ----------------------------------------------------------------------
# Measurement window: where the thresholds sit in the cost-gap distribution
# ----------------------------------------------------------------------


def write_thresholds(demos_dir: Path, data: Path, task: str, step_penalty: float = 0.05) -> None:
    """From a local demonstration set: the cost gap between the two kinds, and the thresholds the arms set."""
    from goalmisgen.offline.demonstrations import load_demonstrations

    demos = load_demonstrations(demos_dir)
    rows = np.arange(len(demos))
    distances = np.asarray(demos.distances)
    feature = np.asarray(demos.feature_ids)
    first = distances[rows, (feature == 0).argmax(1)]
    second = distances[rows, (feature == 1).argmax(1)]
    values = sorted(demos.meta["values"], reverse=True)
    base = (values[0] - values[1]) / step_penalty
    payload = {
        "task": task,
        "gap": (first - second).astype(int).tolist(),
        "base_threshold": base,
        "arm_thresholds": [base + o / step_penalty for o in (-0.45, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.45)],
        "values": values,
        "n": int(len(demos)),
    }
    (data / task).mkdir(parents=True, exist_ok=True)
    (data / task / "thresholds.json").write_text(json.dumps(payload))
    print(f"wrote {data / task / 'thresholds.json'}")


def fig_thresholds(data: Path, out: Path) -> None:
    tasks = [t for t in TASKS if (data / t / "thresholds.json").exists()]
    if not tasks:
        return
    fig, axes = plt.subplots(1, len(tasks), figsize=(4.2 * len(tasks), 2.8), squeeze=False)
    for ax, task in zip(axes[0], tasks):
        payload = json.loads((data / task / "thresholds.json").read_text())
        gap = np.asarray(payload["gap"])
        lo, hi = np.percentile(gap, [0.5, 99.5])
        ax.hist(gap, bins=np.arange(np.floor(lo) - 0.5, np.ceil(hi) + 1.5, 1), color=BLUE, alpha=0.55, lw=0)
        for t in payload["arm_thresholds"]:
            ax.axvline(t, color=ORANGE, lw=0.6, alpha=0.6)
        ax.axvline(payload["base_threshold"], color=INK, lw=1.2)
        kinds = KIND[task]
        ax.set_xlabel(f"extra actions for {kinds[0]} over {kinds[1]}  (cost gap)")
        ax.set_ylabel("fields")
        share = float(np.mean(gap < payload["base_threshold"]))
        ax.set_title(
            f"{TASKS[task]}\nexpert takes {kinds[0]} on {share:.0%}; base threshold {payload['base_threshold']:.0f}, arms in orange",
            fontsize=8.5,
        )
        ax.grid(True, axis="y")
    fig.suptitle("Where the thresholds sit: the window each task lets the axis be measured in", y=1.04, fontsize=10)
    save(fig, out, "fig_thresholds")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path(__file__).parent / "data" / "craftax")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "craftax")
    parser.add_argument("--thresholds", nargs=2, metavar=("TASK", "DEMOS"), action="append", default=[])
    args = parser.parse_args()
    for task, demos in args.thresholds:
        write_thresholds(Path(demos), args.data, task)
    for task in TASKS:
        fig_dynamics(args.data, args.out, task)
        fig_value_axis(args.data, args.out, task)
    fig_thresholds(args.data, args.out)


if __name__ == "__main__":
    main()
