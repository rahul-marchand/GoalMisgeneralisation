---
date: 202607241500
title: 4YP Experiment 1 - Planning Interp Replication
status: proposal
tags: [4yp, experiment]
---

# Experiment Bush replication + extensions


#### Goal:

Building small toy models that cleanly isolate key features of model behaviour seems a useful way to test broader hypotheses about training and behaviour. A sort of "model-organism" of goal misgeneralisation. E.g. [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html), [Nanda et al. grokking/progress measures](https://arxiv.org/abs/2301.05217).

In the literature there are some simple examples of goals being isolated: https://arxiv.org/abs/2310.08043. (results seem slightly messy)

Also some clean examples of planning circuits being isolated in models: (https://arxiv.org/abs/2504.01871 Tom bushes paper). and some interesting follow up papers and blog posts. E.g the inference-time-scaling like result (agent takes non-committal moves to buy planning time, then performs better) is [Taufeeque et al. 2024 (arXiv)](https://arxiv.org/abs/2407.15421).

My goal here is to cleanly isolate competing goals in a model. i.e. a mechanism that weighs up two or more different objectives and can be intervened on to make the model switch from one target to another. 

If successful then I can use this as a nice toy model to play around with. 

#### Choice of model architecture:
DRC as it seems to have given clean results in planning papers.

See below paper for what a DRC agent is but essentially they designed an architecture with the intent of cleanly producing planning.
[https://proceedings.mlr.press/v97/guez19a.html](https://proceedings.mlr.press/v97/guez19a.html)

#### Choice of Environment:

Maze navigation is very simple to set up, has a clean ground truth and is easy to build  multiple objectives into.


#### Experiment 1 (Replicate appendix C)
https://tuphs28.github.io/projects/interpplanning/
is just train to reach a target and make sure we can isolate an objective/goal circuit.

#### Experiment 2 

We want to train the agent to weigh up two different objectives.

We set two different objectives with different reward weights. Episode ends when an objective is reached. Also we add a per step reward penalty. Essentially agent will have to pick max(objective_reward - step_penalty * distance).

(1) check that behaviourally we are observing what we are trying to train for.

(2) Probe for targets, distances

(3) try to intervene to change weightings, target directly etc

(4) check if interventions work on unseen mazes


---
## Status — stage 1 done (2026-08-05)

Two sense-check questions. Both yes, with caveats worth remembering.

#### Q1: Can we train an agent that weighs two objectives, and can we break it?

Yes to both. Four runs, all DRC(3,3), 11×11 unless stated:

| run | size | train ρ | values | steps |
|---|---|---|---|---|
| `smoke5b` | 5×5 | 1.0 | fixed | 10M |
| `maze11` | 11×11 | 1.0 | fixed | 150M |
| `clean11fv` | 11×11 | 0.5 | fixed | 150M |
| `clean11` | 11×11 | 0.5 | randomised | 130M |

All reach an objective 100% of the time. `chose_optimal` at test ρ = 1.0 / 0.5 / 0.0,
2,048 held-out episodes per arm:

| run | 1.0 | 0.5 | 0.0 | followed feature 0 |
|---|---|---|---|---|
| `smoke5b` | 100% | 50% | **0%** | 100 / 100 / 100 |
| `maze11` | 96% | 79% | **60%** | 77 / 67 / 59 |
| `clean11fv` | 95% | 96% | **96%** | 76 / 51 / 23 |
| `clean11` | 96% | 96% | **96%** | 65 / 51 / 36 |

`clean11fv` is the control that matters — same levels, same value scheme, same
split as `maze11`, differing **only** in training ρ. So the collapse is the
correlation, not the task or the value scheme. `clean11` agrees but changes two
variables at once, so it is a secondary check only.

Two things to keep in mind:

- **The 11×11 failure is partial and structured, not a flip.** 60% optimal, not
  0%. And it is not uniform noise: on the same levels the control is flat across
  ρ in every margin band, while `maze11` loses 64 points in the 0.15–0.35 band
  and 31 on the easiest decisions. It captures the decisions that should be easy.
  → `figures/fig3_margin.png`
- **The 5×5 is the degenerate limit.** A pure colour-follower, but the proxy is
  free there, so it says little.

Misgeneralisation appears only *after* competence (~20M steps) and only in the
proxy run — the control's ρ=1.0 − ρ=0.0 gap sits at 0.00 for the whole run.
→ `figures/fig4_dynamics.png`

#### Q2: Can we extract the plans?

Yes, in the *route* sense. Linear 1×1 conv on the ConvLSTM carry at t=0,
predicting which cells the agent will step on later in the episode. AUC on
held-out episodes:

| run | trained | untrained net | observation |
|---|---|---|---|
| `smoke5b` | 0.993 | 0.562 | **0.871** |
| `maze11` | **0.967** | 0.523 | 0.583 |
| `clean11fv` | 0.910 | 0.513 | 0.582 |
| `clean11` | 0.916 | 0.541 | 0.600 |

Three things make this hold up, and all three were needed:

1. **Untrained-network control.** A random conv tower already has a receptive
   field, so beating the observation proves nothing. 0.523 is the number that
   matters.
2. **Distance-matched negatives.** Pooled negatives let a pure
   `-BFS-distance` feature reproduce the whole profile. Matched, that control
   scores exactly 0.500 and the real probe stays flat at 0.955–0.983 out to step
   12 — a plan, not a next-move predictor.
3. **Bootstrap over episodes, not cells.** Cells in an episode share a maze;
   cell-level intervals are ~1.6× too narrow.

Extra thinking time sharpens it: think=4 takes `maze11` to 0.999, with the
observation row bit-identical, so the labels did not move.
→ `figures/fig5_example_plan.png`, `results/probe-*.txt`

**The 5×5 row is the weak one**: seven free cells and two-step routes mean the
observation alone gets 0.871. Little headroom for a plan to live in.

#### Figure provenance — for captions

Everything behavioural is `cp_140206080` (150M-step runs) or `cp_9011200` (5×5),
2,048 episodes per arm, on the **test** split — held out from both training and
in-training evaluation. Dataset `levels11`, fingerprint `d0e70f346ac4a46a`.

| fig | runs shown | source |
|---|---|---|
| 1 | none — a sampled level | `MazeLevelSampler`, seed 3 |
| 2 | `smoke5b`, `maze11`, `clean11fv` | `002`, held-out test episodes |
| 3 | `clean11fv` and `maze11`, ρ=1.0 vs ρ=0.0 arms | same episodes as fig 2 |
| 4 | `maze11` (both panels), `clean11fv` (gap panel) | cleanba's **in-training** eval, `metrics.csv` |
| 5 | `maze11` | probe fitted on 256 episodes, drawn on 12 it never saw |

**Fig 4 is the odd one out and needs the most caption.** It is the only figure
not drawn from `002`: it uses cleanba's own evaluation *during* training, which
is a different, smaller set of levels from the `valid` split. So its absolute
heights are not comparable with figs 2–3, and `maze11`'s curves predate the
seeding fix while `clean11fv`'s do not. Only the **within-run gap** is
comparable, because both arms of a run always see the same levels.

What fig 4 is meant to show, in order:

1. *Top panel* — the proxy run becomes competent at ~20M steps (return jumps
   from −5.5 to +0.3). Its three test correlations are **indistinguishable at
   this scale**. That is the point of the panel: returns alone hide the effect.
2. *Bottom panel* — subtract ρ=0.0 from ρ=1.0 and it appears. The proxy run
   settles at **+0.165**; the control at **−0.001**. Before 20M the gap is noise
   between two incompetent agents and means nothing.

So: misgeneralisation shows up only after the agent can solve the task at all,
and only in the run where a proxy was available. Final returns, `maze11`
ρ=1.0 / ρ=0.0 = 0.409 / 0.230; `clean11fv` = 0.441 / 0.441 — note the control
beats the proxy run even in the condition the proxy run was trained on.

#### What is *not* established

- The probe reads **routes, not targets**. It says where the agent will step,
  not which objective it intends. The project hypothesis is about the target.
- **Purely correlational.** No patching, no steering, no causal test.
- **n=1 seed** everywhere.
- `fig4`'s two older runs predate the seeding fix, so absolute heights there are
  not comparable with anything else; only the within-run gap is meaningful.
- Checkpoints are dense (1M) only to 20M. The behavioural gap grows over
  20→40M, where resolution is 10M.

#### Next

1. **What rule did it learn?** `maze11` defects on the middle margin band — read
   off which levels, from existing rollouts. Free.
2. **Does test-time compute change behaviour?** Thinking sharpens the probe to
   0.999; does it change the *choice*? Existing checkpoints, ~1h.
3. **Target probe + per-level timing.** At checkpoint *t*, does the target
   representation already encode colour-dependence on the levels where behaviour
   defects at *t+k*? Per-level beats a global lead-time comparison — thousands of
   paired observations instead of one number, and robust to coarse checkpoints.
4. A 50M run with 1M checkpointing to resolve the transition (~3.5h).
5. Dose-response: train ρ ∈ {0.7, 0.8, 0.9, 1.0}.

---
## Links
[[4YP Working Notes]] · [[4YP Literature Hub]] · [[Bush2025_EmergentPlanning]] · [[Mini2023_MazePolicy]]
