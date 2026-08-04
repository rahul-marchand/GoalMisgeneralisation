# Pre-bugfix results — archived 2026-08-04

Superseded. Do not quote these numbers or reuse these figures.

## Why

A review pass found defects that change measured results. Two of them
invalidate everything here:

- **Episode-start flags were never raised in the analysis rollouts.** With
  autoreset, 15 of every 16 episodes ran with the previous maze's recurrent
  state still in the carry. Correcting it moved `reached` from 64.2% to 100%,
  `chose_optimal` from 76.1% to 94.3%, and mean episode length from 49.2 steps
  to 8.0.
- **The feature-assignment RNG drew a variable number of values per
  correlation branch**, so the ρ arms did not see the same levels — the
  comparison across ρ was confounded with level difficulty.

Two further problems affect specific figures:

- `mean_return` averaged `reached_value` and ignored the step penalty, so a
  slow agent scored like a fast one.
- Figure 4's optimum reference line is drawn at 0.260, the mean over the whole
  valid split. The curves were scored on evaluation batch 0, whose mean optimal
  utility is 0.159 — 2.7 sem below the split mean. The agent was at 96% of
  optimal on the levels it actually saw, not at 59%.

The RNG fix changes which levels a dataset contains, so these numbers cannot be
carried forward or patched: the whole set has to be re-measured.

## Contents

| File | What it was |
|---|---|
| `fig1_task.*` | Task illustration — the only figure not invalidated by the above, redrawn anyway for consistency |
| `fig2_rho_response.*` | `chose_optimal` and `followed_f0` against ρ, three agents |
| `fig3_margin.*` | Utility margin between chosen and optimal objective |
| `fig4_dynamics.*` | Training curves against the (wrong) optimum reference |
| `numbers/maze11.csv` | Training metrics, 11×11 proxy run |
| `numbers/clean11.csv` | Training metrics, 11×11 control run |
| `numbers/smoke5b.csv` | Training metrics, 5×5 run |
| `numbers/measured.json` | Behavioural table behind figures 2 and 3 |

The level datasets these were measured on are archived on the pod at
`/workspace/data/archive/2026-08-04-pre-bugfix/`. They no longer load: their
source fingerprint does not match current code, and `LevelDataset` raises
`FingerprintMismatch` by design.
