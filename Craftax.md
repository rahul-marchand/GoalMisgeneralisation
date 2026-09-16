# Craftax: does the value axis appear in a transformer cloned from an optimal solver?

*Started 2026-09-15. Branch `craftax`. Design in
`~/.claude/plans/how-to-do-this-quizzical-seal.md`; handoff in
`~/projects/gm-worktrees/craftax.HANDOFF.md`.*

## Question

The maze work found a **value axis**: fine-tuning a hidden-value agent onto a
grid of shifted objective values moves its weights along one direction, that
direction is writable, and it behaves like a gain on the colour input
(UtilityRule.md, the gain-vs-factorised study). Does the same thing appear when
a transformer is trained by behavioural cloning from a utility-optimal expert
inside Craftax-Classic, a richer, recognised environment with a 17-action
vocabulary and a crafting tree?

## Engine, not task

Craftax (`craftax==1.6.1`) is a dependency used unmodified: we build its
states and step them with `craftax_step`. What we add is the world, the
expert, the observation and the bridge (`goalmisgen/craftax/`); the BC
pipeline is shared with the maze through the `Demonstrations` protocol
(`goalmisgen/offline/demonstrations.py`). CLAUDE.md lists the rules.

## Stages

| stage | world | observation | expert | what a failure means |
|---|---|---|---|---|
| 2 | 15x15 ore field: stone border and scattered stone, two ores (coal, diamond) at fixed hidden values (1.0, 0.5), player holding the pickaxes | the maze's channel stack (wall, agent, kind x2), full map | maze BFS route + DO, `value - 0.05 x actions` | a pipeline bug: the task is the maze's, only the action head and the executor differ |
| 3 (deferred) | as 2 | Craftax's own 7x9 local view + inventory, with history | full-state planner, or explore-then-plan | separate the view change from the memory change |
| 4 | fields with trees, stone, coal, iron, diamond; values on *achievements* (e.g. a diamond vs an iron pickaxe) | full map + inventory, prefix-LM | planner over the tech-tree subset, 17-action routes | a missing axis is a result |

Stage 2 is engine-first on purpose: every decoded route is executed by the
real engine, so a 17-way head that learned the wrong ids, or a DO in the wrong
place, shows there and nowhere earlier. Stage 4 is where Craftax earns its
place: raising the value of diamond raises the *instrumental* value of the
iron pickaxe and everything below it, and a gain-on-input mechanism can only
rescale the diamond cue, so H1 (gain) against H2 (factorised value circuit)
becomes sharp.

## Stage 2 protocol

The maze value-axis campaign (`scripts/offline_value_axis_pod.sh`,
Experiment2.md) transplanted:

1. **Levels.** `scripts/generate_ore_fields.py`: 300k fields at (1.00, 0.50),
   60k at each arm value; layout-held-out splits; fingerprint
   `ore_field_fingerprint`. At 15x15 and density 0.3 the distance gap
   |d0 - d1| has a 95th percentile of 16 steps and 18% of levels lie past the
   10-step trade-off threshold, like the 11x11 mazes (11x11 fields top out at
   8-10 and almost never cross it).
2. **Demonstrations.** `scripts/generate_craftax_demos.py` at rho 1.0
   (train/valid/test) and rho 0.5 / 0.0 (valid, for the misgeneralisation
   readout); routes are maze routes in engine ids plus DO, distances one
   longer.
3. **Bases.** `cxnv15.s{1,2,3}`: the `bcnv11` recipe (4 layers, d=128, 30k
   steps, batch 256) with `--hide-values` and `n_actions=17`.
   Competence gate: reach and optimal-choice rates through the engine must
   match the maze BC base on the same levels before any axis work.
4. **Arms.** `goalmisgen.design.sweep_arms` at 1k steps, constant lr 3e-5, on
   demonstrations at the shifted values (`values_tag` datasets), values still
   hidden, so the only way to move what an ore is worth is to move weights.
5. **Analysis.** `experiments/027_bc_value_axis.py` per sweep: axis fit,
   leave-one-out write, norm-matched random controls, `cos(axis_0, axis_1)`.

Everything lands under `/workspace/data/craftax/` via `scripts/craftax_chain.sh`.

## Stage 4 protocol

`goalmisgen/craftax/craft.py`, `scripts/generate_craft_demos.py`, `scripts/craft_chain.sh`.

- **World.** 15x15 field, stone border, stone at density 0.2, six trees, two
  ores, a bare-handed player. No level dataset: field ``i`` of pool
  ``(sampler, seed)`` is a deterministic draw, splits are index ranges.
- **Chains.** Coal needs a wood pickaxe: three wood (two for the table, one
  for the pickaxe). Iron needs a stone pickaxe on top: one more wood, one
  stone, and a return beside the table. Engine rules verified by stepping:
  a mined tree becomes grass; a table goes on the faced non-solid tile for
  two wood; crafting needs the table in the 8-neighbourhood.
- **Expert.** Optimal within a plan family: tree order, table on the third
  tree, stone-or-fourth-tree order, nearest stones; ranked on the static
  grid, the best two executed exactly on the changing grid, the cheapest real
  route kept. Not considered: a second table, tunnelling. Every plan is run
  through the engine in the tests: target collected in exactly its cost,
  the tool crafted, nothing wasted.
- **Values.** Iron is the rich kind (feature 0 at rho=1) at (2.0, 0.5): a
  30-action threshold at the median extra cost of the iron chain over the
  coal chain (p25/50/75 = 21/27/36 on these fields), so the expert takes iron
  on ~58% and the +-0.45 arms (thresholds 21..39) sweep the interquartile
  range. Had the cheap kind been the rich one the expert would never trade.
- **Bases.** ``cxcraft15.s{1,2,3}``: the same model, ``max_actions=128``
  (routes average ~38 actions, max ~65), 40k steps on 150k fields.
- **What it asks.** Raising iron's value must raise the worth of the stone
  pickaxe, the stone and the fourth tree, none of which carry the iron cue.
  If the axis is again a gain on the ore's input embedding, the value is
  attached to the cue and the plan is downstream of it; if it lives elsewhere,
  the model has a value variable the cue only informs.

## Engine facts the code leans on (all tested in `tests/test_craftax_engine.py`)

| fact | consequence |
|---|---|
| ores are solid; a move into a solid block only turns the player | the expert route is the maze route + DO; distance = maze + 1 |
| `do_action` runs before `move_player` within a step | DO uses the facing set by the previous step |
| each ore checks only its own pickaxe flag | the task hands the player every pickaxe its kinds need |
| a wood pickaxe (needed for coal) digs stone into path | walls are diggable by a decoded model; `walls_mined` is reported, never done by the expert |
| `OUT_OF_BOUNDS` is walkable | fields are bordered and padded with stone |
| `reset` refuses maps under 16 cells | states are built by hand, `craftax_step` only needs consistent shapes |
| DO facing grass rolls a sapling on the rng | replays use a fixed key |
| energy falls to 8 around step 31; SLEEP then freezes the player | reported as wasted actions; never reached by an expert route |
| no objective values in `EnvState` | observation = f(state) exactly under hidden values |

## Results (overnight 2026-09-15/16, one seed each; figures under `figures/craftax/`)

**Stage 2, `cxnv15.s1` (20k steps, values hidden).** Competence gate not met:
reaches an ore on 54% of held-out fields (the maze base: 98% at 30k), and the
routes that fail are illegal (walk into stone), not wrong-headed: among routes
that reach, 97% take the optimal ore, the indifference point is 9.8 actions
(expert 10), and at reversed correlation the model still takes kind 0 on 92%,
as a hidden-value model should. Reach was still climbing (16% / 31% / 54% at
6.4k / 12.8k / 20k steps), so this is under-training on a task whose shortest
paths are far from unique (open field, braided), not a pipeline fault; a
continuation to 60k is on the volume as `cxnv15.s1c`.

**The value axis appears anyway.** Nine arms per sweep at 1k steps:

| statistic | Craftax stage 2 | maze BC (3 seeds) |
|---|---|---|
| cos(axis_0, axis_1), raw / disattenuated | −0.99 / −1.14 | −0.98 / −1.03 |
| split-half reliability | 0.87 / 0.88 | 0.96 |
| held-out write error (actions) | 0.6 / 1.1 | 1.6–2.7 |
| norm-matched random directions move τ by | ≤ 0.3 | ≤ 0.2 |
| slope of τ vs value: arms / written / expert | 13 / 10.5 / 20 | 22 / 25 / 20 |
| extrapolated writes at ±0.6, ±0.9 | τ 15.3 → 1.4 (o0), 3.2 → 15.6 (o1) | – |

One knob, writable, held out of the fit; the slopes are two thirds of the
maze's, as expected of a base that has not converged. `figures/craftax/fig_ore_value_axis.png`,
`fig_ore_dynamics.png`, `fig_thresholds.png`; numbers in
`results/value_axis.cxnv15.s1.o{0,1}.txt`.

**Stage 4, `cxcraft15.s1` (30k steps).** The crafting chain is *not* learned:
3% reach, 88% legal, 85% teacher-forced token accuracy on training and
held-out fields alike (so not over-fitting). `scripts/craft_diagnose.py` puts
the first error in the *first leg* on 60% of fields, and the errors are
move-direction confusions (DOWN→RIGHT, UP→LEFT, ...): the planner's choice of
which tree to cut first is ranked over permutations of static distances and
is not inferable from the map, so the imitation target is inconsistent and
the model hedges between directions, then drifts. Every decoded route still
places a table and crafts a wood pickaxe, 60% craft a stone pickaxe, and 11
actions per route are wasted no-ops. This is the failure Rahul predicted, and
it is an expert-design problem: the fix is a canonical planner whose choices
are observation-inferable (nearest-tree-first with a fixed move tie-break,
the same tie-break for paths), not more capacity. A 60k-step continuation
(`cxcraft15.s1c`) is running as the cheap control.
