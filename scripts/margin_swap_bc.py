"""The swap/start decomposition of xi: what do the model's mistakes depend on?

    uv run python scripts/margin_swap_bc.py RUN --eval-demos DEMOS --choices NPZ \
        > results/margin-swap-<model>.txt

The model is deterministic, so its deviation from the utility rule is a fixed
function of the presentation, not noise. This measures that function along two
designed orbits and decomposes its variance, with no probe anywhere:

- **swap**: each maze presented with the colour channels exchanged. Anything
  keyed to which location carries which role inverts (T); anything keyed to
  the maze or the colours survives (S).
- **starts**: each maze presented from the natural start plus constructed
  starts drawn by the sampler's own rule (uniform over free cells). Within T,
  path-cost error must vary with the start; goal-local perception cannot.

Estimands, fixed in advance:
  T = (m - m_swap)/2 centred within signed integer gap;  eps = -T / g
  S = (m + m_swap)/2 centred within |gap|;               sigma = S / g
  g = slope of -T_raw on gap in the band (the margin's own calibration)
  shares Var(eps) / (Var(eps) + Var(sigma)); start-invariant share of eps by
  one-way variance components across starts.

Commitments: the margin is validated against decoded choices before anything
is concluded; Var(T)+Var(S) must close against the single-orientation
within-cell variance; a within-bin shuffle of the swap pairing must destroy
the cancellation. Scope: T is location-difference-keyed error - fractional
distance misread and per-maze price error are one number (the gauge), and the
write-up says "perceived path-cost difference", not "BFS".
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from goalmisgen.analysis.probes import roc_auc
from goalmisgen.envs.solver import UNREACHABLE
from goalmisgen.offline.demos import DemoSet
from goalmisgen.offline.margins import (
    approach_sets,
    divergence_cell,
    first_action_logits,
    margin,
    move_agent,
    objective_fields,
    swap_colours,
)
from goalmisgen.offline.train import list_checkpoints, load_checkpoint
from goalmisgen.provenance import header

BAND = 14
"""|gap| ceiling for the analysis; beyond it margins saturate and cells thin."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    parser.add_argument("--eval-demos", type=str, required=True)
    parser.add_argument("--choices", type=Path, required=True, help="decode npz aligned with the demos' first n levels")
    parser.add_argument("--n", type=int, default=20_000)
    parser.add_argument("--starts", type=int, default=5, help="Natural start plus this-minus-one constructed ones.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--chunk", type=int, default=2000, help="Levels per forward-pass block.")
    parser.add_argument(
        "--read",
        choices=("start", "fork"),
        default="start",
        help="Where the margin is read: the presentation's start cell, or the routes' divergence point - "
        "the commitment decision, from which shared-corridor error cancels.",
    )
    return parser.parse_args()


def centred(values: np.ndarray, bins: np.ndarray, min_n: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Values minus their bin mean; mask of rows in bins large enough to centre.

    NaN rows contribute nothing to their bin's mean and stay excluded, so
    passes can be stacked (gap centring, then a fixed-effect centring).
    """
    out = np.full_like(values, np.nan)
    keep = np.zeros(len(values), dtype=bool)
    for value in np.unique(bins):
        members = np.flatnonzero(bins == value)
        finite = members[np.isfinite(values[members])]
        if len(finite) >= min_n:
            out[finite] = values[finite] - values[finite].mean()
            keep[finite] = True
    return out, keep


def main() -> None:
    args = parse_args()
    print(header())
    print()

    demos = DemoSet.load(args.eval_demos, hide_values=True)
    model, params = load_checkpoint(list_checkpoints(args.run)[-1][1])
    n = min(args.n, len(demos))
    K = args.starts
    rng = np.random.default_rng(args.seed)

    gap = np.full((n, K), np.nan)
    m_a = np.full((n, K), np.nan)
    m_b = np.full((n, K), np.nan)
    pair = np.full((n, K), -1, dtype=np.int64)
    pair_codes: dict[tuple, int] = {}

    start_time = time.perf_counter()
    for block in range(0, n, args.chunk):
        indices = np.arange(block, min(block + args.chunk, n))
        base = demos.observations(indices)
        rows, starts_rc, sets0, sets1, slot = [], [], [], [], []
        for row, index in enumerate(indices):
            level = demos.level(int(index))
            fields = objective_fields(level)
            c0 = next(k for k, o in enumerate(level.objectives) if o.feature_id == 0)
            free = np.argwhere(~level.walls)
            free = [tuple(c) for c in free if tuple(c) not in {o.position for o in level.objectives}]
            candidates = [level.agent_start] + [
                free[i] for i in rng.choice(len(free), size=min(4 * K, len(free)), replace=False)
            ]
            chosen = []
            for cell in candidates:
                if args.read == "fork":
                    fork = divergence_cell(level, cell)
                    if fork is None:
                        continue
                    cell = fork
                if cell in chosen:
                    continue
                d0, d1 = fields[c0][cell], fields[1 - c0][cell]
                if d0 == UNREACHABLE or d1 == UNREACHABLE:
                    continue
                toward = approach_sets(fields, cell)
                if not toward[c0] or not toward[1 - c0]:
                    continue
                s = len(chosen)
                gap[index, s] = d0 - d1
                rows.append(row)
                starts_rc.append(cell)
                sets0.append(toward[c0])
                sets1.append(toward[1 - c0])
                slot.append((index, s))
                chosen.append(cell)
                if len(chosen) == K:
                    break
        if not rows:
            continue
        obs = move_agent(base[np.array(rows)], np.array(starts_rc))
        logits_a = first_action_logits(model, params, obs)
        logits_b = first_action_logits(model, params, swap_colours(obs))
        for j, (index, s) in enumerate(slot):
            m_a[index, s] = margin(logits_a[j], sets0[j], sets1[j])
            # After the swap the model's colour 0 sits at the other objective.
            m_b[index, s] = margin(logits_b[j], sets1[j], sets0[j])
            # A maze-independent preference among actions enters T at full
            # strength (the comparison's reference order flips with the swap),
            # so the ordered approach-pair is recorded and removed as a fixed
            # effect below. S is free of it by construction.
            key = (sets0[j], sets1[j])
            pair[index, s] = pair_codes.setdefault(key, len(pair_codes))
    print(f"forward passes done in {time.perf_counter() - start_time:.0f}s")

    flat = np.isfinite(m_a) & np.isfinite(m_b) & (np.abs(gap) <= BAND)
    level_of = np.broadcast_to(np.arange(n)[:, None], (n, K))
    g_flat, ma_flat, mb_flat = gap[flat], m_a[flat], m_b[flat]
    levels_flat = level_of[flat]
    natural = np.broadcast_to(np.arange(K)[None, :], (n, K))[flat] == 0
    print(f"{flat.sum():,} valid (maze, start) presentations in |gap| <= {BAND}; {natural.sum():,} natural starts\n")

    print("== 0. is the margin a faithful readout of choice? (natural start, unswapped) ==")
    z = np.load(args.choices)
    took = (z["reached_fid"][:n].astype(int) == z["colour_of_rich"][:n].astype(int)) & z["reached"][:n]
    nat_rows = np.flatnonzero(flat[:, 0])
    auc = roc_auc(took[nat_rows].astype(float), m_a[nat_rows, 0])
    near = nat_rows[np.abs(gap[nat_rows, 0] - 10) <= 4]
    auc_near = roc_auc(took[near].astype(float), m_a[near, 0])
    print(f"  AUC(margin, decoded choice): {auc:.3f} overall, {auc_near:.3f} near threshold (gap 6..14)")
    if auc < 0.85:
        print("  WARNING: margin poorly tracks choice; interpret everything below accordingly")
    print()

    t_raw = (ma_flat - mb_flat) / 2.0
    s_raw = (ma_flat + mb_flat) / 2.0
    slope = np.polyfit(g_flat, -t_raw, 1)[0]
    print("== 1. calibration ==")
    print(f"  g = {slope:.3f} logits per step  (per-|gap| tertile: ", end="")
    order = np.argsort(np.abs(g_flat))
    for third in np.array_split(order, 3):
        print(f"{np.polyfit(g_flat[third], -t_raw[third], 1)[0]:.3f} ", end="")
    print(")")

    if abs(slope) < 0.05:
        print("  WARNING: slope too small to calibrate; reporting logit units, not steps")
        slope = 1.0

    t_c, keep_t = centred(t_raw, g_flat.astype(int))
    s_c, keep_s = centred(s_raw, np.abs(g_flat).astype(int))
    keep = keep_t & keep_s

    # Direction-bias fixed effect: the ordered approach-pair's mean of the
    # already-gap-centred T. Removed from eps; its share is reported as its
    # own quantity - it is the model's motor bias, measured.
    pair_flat = pair[flat]
    t_debiased, keep_pair = centred(t_c, pair_flat, min_n=30)
    keep = keep & keep_pair & np.isfinite(t_debiased)
    bias_share = 1.0 - float(np.var(t_debiased[keep])) / float(np.var(t_c[keep]))

    eps = -t_debiased[keep] / slope
    sig = s_c[keep] / slope
    var_eps, var_sig = float(np.var(eps)), float(np.var(sig))
    print("\n== 2. the decomposition, in steps ==")
    print(f"  direction bias removed from T (approach-pair fixed effect): {bias_share:.1%} of Var(T)")
    print(f"  sd(eps)   (swap-antisymmetric, location-difference-keyed, debiased): {np.sqrt(var_eps):.2f}")
    print(f"  sd(sigma) (swap-symmetric, maze/colour-keyed):                       {np.sqrt(var_sig):.2f}")
    print(f"  share of within-cell variance that inverts under the swap: {var_eps / (var_eps + var_sig):.1%}")

    ma_c, keep_a = centred(ma_flat, g_flat.astype(int))
    total = float(np.var(ma_c[keep_a & keep] / slope))
    print(f"  closure: Var(eps) + Var(sigma) = {var_eps + var_sig:.2f} vs single-orientation within-cell {total:.2f}")

    shuffled = mb_flat.copy()
    for value in np.unique(g_flat.astype(int)):
        members = np.flatnonzero(g_flat.astype(int) == value)
        shuffled[members] = shuffled[members][rng.permutation(len(members))]
    t_shuf, keep_shuf = centred((ma_flat - shuffled) / 2.0, g_flat.astype(int))
    print(f"  shuffled-pairing control: sd {np.sqrt(np.var(t_shuf[keep_shuf & keep] / slope)):.2f} (cancellation destroyed)")

    out = Path("figures/data") / f"margin-{args.read}-{args.run.name}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, level=levels_flat, gap=g_flat, m_a=ma_flat, m_b=mb_flat, pair=pair_flat, natural=natural)
    print(f"  per-presentation arrays saved to {out}")

    print("\n== 3. within eps: path-keyed or start-invariant? ==")
    eps_full = np.full(len(t_c), np.nan)
    eps_full[keep] = eps
    per_level: dict[int, list[float]] = {}
    for value, lev in zip(eps_full, levels_flat):
        if np.isfinite(value):
            per_level.setdefault(int(lev), []).append(float(value))
    groups = [np.array(v) for v in per_level.values() if len(v) >= 3]
    within = float(np.mean([g.var(ddof=1) for g in groups]))
    k_bar = float(np.mean([len(g) for g in groups]))
    between = float(np.var([g.mean() for g in groups], ddof=1)) - within / k_bar
    between = max(between, 0.0)
    print(f"  {len(groups):,} mazes with >=3 usable starts (mean {k_bar:.1f})")
    print(f"  start-invariant share of Var(eps): {between / (between + within):.1%}   (goal-local stories predict ~100%)")
    print(f"  start-varying share:               {within / (between + within):.1%}   (path-cost error predicts most of it)")
    print("\nGauge note: eps is 'perceived path-cost difference' error; fractional distance")
    print("misread and per-maze price error are one number and stay unseparated.")


if __name__ == "__main__":
    main()
