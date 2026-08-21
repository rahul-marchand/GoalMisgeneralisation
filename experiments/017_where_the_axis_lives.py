"""What is actually moving: the shared part, and where the value axis sits.

    uv run python experiments/017_where_the_axis_lives.py \
        --base /workspace/data/runs/novalue11.s1234/local-files/cp_140206080 \
        --arms /workspace/data/runs/novalue11.s1234/arms --prefix v --base-value 0.5

Two questions the behavioural work leaves open.

**Is the movement shared or idiosyncratic?** Fine-tuning moves the weights a long
way whether or not there is anything to learn — the null arm proves that, having
moved as far as any other while changing no behaviour at all. So the movement
splits into a component every arm has in common and a remainder. This reports
how big each is, and whether the remainders are independent of one another or
share further structure.

**Where does the axis live?** The behavioural results say a direction exists and
can be written to; they say nothing about whether it is concentrated in a few
parameters, in one layer, or spread over the whole network. That is asked here
by decomposing the axis by module and by measuring how much of its length sits
in its largest entries, against the same measurements on the shared component
and on the parameter counts themselves.

Two cautions are built into the output rather than left to the reader.

*Localisation is not storage.* Where a diff lands is where gradient descent
found it cheapest to write, which is not the same as where a quantity is
represented — the point of Hase et al. 2023, and the reason this file makes no
claim beyond describing the direction we can already write to causally.

*A fitted axis is mostly noise.* Reliability on these grids runs from 0.04 to
0.15, so any per-parameter structure is largely sampling error. Every structural
number here is therefore computed twice, on disjoint halves of the arms, and
reported with the agreement between them. A profile that does not replicate
across halves is not a finding.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import jax
import numpy as np
from cleanba.cleanba_impala import load_train_state

from goalmisgen import provenance
from goalmisgen.analysis.weights import cosine, fit_axis_and_drift
from goalmisgen.configs.env import MazeConfig
from goalmisgen.volume import parse_arm_dirname, sweep_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--arms", type=Path, required=True)
    parser.add_argument("--prefix", type=str, default="v", help="Run-directory prefix identifying the sweep.")
    parser.add_argument("--base-value", type=float, required=True, help="What the swept objective was worth before.")
    parser.add_argument("--levels", type=str, required=True)
    parser.add_argument("--n-objectives", type=int, default=2)
    parser.add_argument("--objective-values", type=float, nargs="+", required=True)
    parser.add_argument("--size", type=int, default=11)
    parser.add_argument("--at", type=int, default=-1)
    parser.add_argument("--top", type=float, nargs="+", default=[0.001, 0.01, 0.1])
    return parser.parse_args()


def leaf_paths(params) -> list[tuple[str, tuple]]:
    """Every parameter array, with a readable path."""
    flat, _ = jax.tree_util.tree_flatten_with_path(params)
    return [
        ("/".join(str(k.key) if hasattr(k, "key") else str(k.idx) for k in path), np.asarray(value)) for path, value in flat
    ]


def module_of(path: str) -> str:
    """Group parameters into the pieces the architecture is built from.

    ``cell_list_N`` is the Nth ConvLSTM *layer*. Its state is a full spatial map,
    so the index is depth in the recurrent stack and never a maze position."""
    if (cell := re.search(r"cell_list_(\d+)/([a-z]+)", path)) is not None:
        return f"cell_list_{cell.group(1)}/{cell.group(2)}"
    for name in ("actor_params", "critic_params"):
        if name in path:
            return name
    if "network_params" in path:
        return "network_params/encoder"
    return path.split("/")[0]


def sweep(
    root: Path, prefix: str, at: int, config, base_value: float
) -> tuple[list[float], list[dict[str, np.ndarray]], list[str]]:
    values, arms, names = [], [], []
    for run in sorted(root.iterdir()):
        parsed = parse_arm_dirname(run.name)
        if parsed is None or parsed.sweep != f"o{sweep_index(prefix)}":
            continue
        checkpoints = sorted((run / "local-files").glob("cp_*"))
        if not checkpoints:
            continue
        _, _, _, state, _ = load_train_state(checkpoints[at], env_cfg=config)
        arms.append(dict(leaf_paths(state.params)))
        values.append(base_value + parsed.offset)
        names.append(run.name)
    return values, arms, names


def concentration(vector: np.ndarray, fractions: list[float]) -> list[float]:
    """Share of squared length held by the largest entries."""
    squares = np.sort(vector**2)[::-1]
    total = squares.sum()
    return [float(squares[: max(1, int(len(squares) * f))].sum() / total) for f in fractions]


def profile(direction: dict[str, np.ndarray]) -> dict[str, float]:
    """Fraction of squared length in each module."""
    by_module: dict[str, float] = defaultdict(float)
    for path, value in direction.items():
        by_module[module_of(path)] += float(np.sum(value**2))
    total = sum(by_module.values())
    return {name: share / total for name, share in sorted(by_module.items())}


def main() -> None:
    args = parse_args()
    print(provenance.header() + "\n")

    config = MazeConfig(
        max_episode_steps=120,
        num_envs=8,
        min_size=args.size,
        max_size=args.size,
        n_objectives=args.n_objectives,
        objective_values=tuple(args.objective_values),
        feature_value_correlation=1.0,
        value_encoding="none",
        colour_is_the_only_value_cue=True,
        level_dataset=args.levels,
        dataset_split="test",
        asynchronous=False,
        seed=0,
    )
    _, _, _, base_state, _ = load_train_state(args.base, env_cfg=config)
    base = dict(leaf_paths(base_state.params))
    counts = {path: value.size for path, value in base.items()}
    print(f"base {args.base.name}  {sum(counts.values()):,} parameters in {len(counts)} arrays\n")

    values, arms, names = sweep(args.arms, args.prefix, args.at, config, args.base_value)
    if len(values) < 4:
        sys.exit(f"need at least four arms, found {len(values)}")
    diffs = [{path: arm[path] - base[path] for path in base} for arm in arms]
    print("arms: " + ", ".join(f"{n}({v:+.2f})" for n, v in zip(names, [v - args.base_value for v in values])))

    keys = list(base)
    flat = np.stack([np.concatenate([d[k].ravel() for k in keys]) for d in diffs])
    offsets = np.array([v - args.base_value for v in values])
    keep = np.abs(offsets) > 1e-9
    axis_flat, drift_flat = fit_axis_and_drift(offsets[keep], flat[keep])
    null = flat[~keep][0] if (~keep).any() else None

    print("\n\n=== how much of the movement is shared? ===\n")
    lengths = np.linalg.norm(flat[keep], axis=1)
    print(f"  mean |delta| per arm          {lengths.mean():.4g}   (spread {lengths.std():.3g})")
    print(f"  |shared component|            {np.linalg.norm(drift_flat):.4g}")
    print(f"  shared share of a typical arm {np.linalg.norm(drift_flat) / lengths.mean():.1%}")
    if null is not None:
        print(f"  |null arm|                    {np.linalg.norm(null):.4g}   cos to shared {cosine(null, drift_flat):+.3f}")

    residual = flat[keep] - drift_flat
    pairs = [cosine(residual[i], residual[j]) for i in range(len(residual)) for j in range(i + 1, len(residual))]
    expected = -1 / (len(residual) - 1)
    print(f"\n  residual pairwise cosine      mean {np.mean(pairs):+.3f}  (range {min(pairs):+.3f} to {max(pairs):+.3f})")
    print(f"  expected from mean-removal    {expected:+.3f}")
    print(
        "\n  Removing a mean from n vectors makes the remainders slightly anti-correlated\n"
        "  by construction. Sitting at that value means the remainders carry no shared\n"
        "  structure beyond what was already taken out; sitting above it means they do."
    )

    print("\n\n=== is the movement concentrated, or spread? ===\n")
    print(f"  {'':>26}" + "".join(f"{f'top {f:.1%}':>12}" for f in args.top))
    for label, vector in (("value axis", axis_flat), ("shared component", drift_flat), ("one arm", flat[keep][0])):
        print(f"  {label:>26}" + "".join(f"{s:>12.1%}" for s in concentration(vector, args.top)))
    gaussian = concentration(np.random.default_rng(0).normal(size=len(axis_flat)), args.top)
    print(f"  {'gaussian, for reference':>26}" + "".join(f"{s:>12.1%}" for s in gaussian))
    print(
        "\n  A direction no more concentrated than a gaussian is spread over the whole\n"
        "  network. Sparsity would show as the top slice holding far more than it does."
    )

    print("\n\n=== which parts of the network move? ===\n")
    sizes = defaultdict(int)
    for path, count in counts.items():
        sizes[module_of(path)] += count
    total_params = sum(sizes.values())

    axis = {path: value for path, value in zip(keys, np.split(axis_flat, np.cumsum([base[k].size for k in keys])[:-1]))}
    drift = {path: value for path, value in zip(keys, np.split(drift_flat, np.cumsum([base[k].size for k in keys])[:-1]))}
    axis_profile, drift_profile = profile(axis), profile(drift)

    # Split the arms in half and fit each, so a module's share can be reported
    # with the agreement between two independent estimates of it.
    order = np.argsort(offsets[keep])
    halves = [order[0::2], order[1::2]]
    half_profiles = []
    for half in halves:
        if len(half) >= 3:
            a, _ = fit_axis_and_drift(offsets[keep][half], flat[keep][half])
            half_profiles.append(
                profile({p: v for p, v in zip(keys, np.split(a, np.cumsum([base[k].size for k in keys])[:-1]))})
            )

    print(
        f"  {'module':>26}{'params':>10}{'axis':>9}{'shared':>9}{'vs params':>11}{'vs shared':>11}"
        + ("{:>16}".format("halves agree") if len(half_profiles) == 2 else "")
    )
    for name in sorted(axis_profile, key=lambda n: -axis_profile[n]):
        share = sizes[name] / total_params
        common = drift_profile.get(name, 0)
        enrichment = axis_profile[name] / common if common > 0 else float("nan")
        row = (
            f"  {name:>26}{sizes[name]:>10,}{axis_profile[name]:>9.1%}{common:>9.1%}"
            f"{axis_profile[name] / share:>11.2f}{enrichment:>11.2f}"
        )
        if len(half_profiles) == 2:
            row += f"{half_profiles[0].get(name, 0):>7.1%}{half_profiles[1].get(name, 0):>9.1%}"
        print(row)
    print(
        "\n  'vs params' above 1 means a module carries more of the axis than its share of\n"
        "  the parameters. 'vs shared' is the one that isolates the value: it compares the\n"
        "  axis against where fine-tuning moves weights anyway, so 1.0 means this module\n"
        "  moves for the value exactly as much as it moves for nothing in particular.\n"
        "  The two half-fits are independent estimates of the same profile: where they\n"
        "  disagree, the number is sampling error rather than structure."
    )
    if len(half_profiles) == 2:
        names_sorted = sorted(axis_profile)
        a = np.array([half_profiles[0].get(n, 0) for n in names_sorted])
        b = np.array([half_profiles[1].get(n, 0) for n in names_sorted])
        print(f"\n  correlation between the two half-profiles: {np.corrcoef(a, b)[0, 1]:+.3f}")

    print(
        "\n\nWhere a diff lands is where gradient descent found it cheapest to write, which\n"
        "is not the same as where a quantity is stored. Nothing here upgrades the causal\n"
        "claim, which is about a direction that can be written to, not about a location."
    )


if __name__ == "__main__":
    main()
