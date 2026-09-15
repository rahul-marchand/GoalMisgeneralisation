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

## Results

*(filled in as the chain reports; figures under `figures/craftax/`)*
