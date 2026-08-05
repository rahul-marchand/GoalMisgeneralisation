---
date: 202607241500
title: 4YP Experiment 1 - Planning Interp Replication
status: stage 1 complete
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
## Status — stage 1 complete (2026-08-05)

#### Setup

DRC(3,3) — ConvLSTM, 3 layers × 3 ticks per env step — trained with IMPALA
(cleanba, unmodified) for 150M steps, γ=0.995.

11×11 maze, two objectives, episode ends on reaching either. Reward is the
objective's value minus 0.05 per step, so **utility = value − 0.05 × distance**
and the nearer objective often wins. 120-step limit.

**ρ = P(feature 0 marks the higher-value objective).** This is the proxy knob:
fixed during training, swept at test. ρ=1.0 makes colour a perfect cue, ρ=0.5 is
chance (uninformative), ρ=0.0 inverts it.

Observation is symbolic 11×11×5 — walls, agent, feature 0, feature 1, value —
never pixels. → `fig1_task`

1M pre-generated levels, test/train/valid = 50k/900k/50k, disjoint. All numbers
below are the **test** split, held out from training *and* from in-training eval.

| run | size | train ρ | values | steps |
|---|---|---|---|---|
| `smoke5b` | 5×5 | 1.0 | fixed | 10M |
| `maze11` | 11×11 | 1.0 | fixed | 150M |
| `clean11fv` | 11×11 | 0.5 | fixed | 150M |
| `clean11` | 11×11 | 0.5 | randomised | 130M |

#### Evidence it works

**1. The agent learns the task.** All four reach an objective 100% of the time.
`clean11fv` picks the higher-utility one 95% of the time at every ρ.

**2. The proxy causes misgeneralisation.** `chose_optimal` at ρ = 1.0 / 0.5 / 0.0,
2,048 episodes per arm → `fig2_rho_response`

| run | 1.0 | 0.5 | 0.0 | followed feature 0 |
|---|---|---|---|---|
| `smoke5b` | 100% | 50% | **0%** | 100 / 100 / 100 |
| `maze11` | 96% | 79% | **60%** | 77 / 67 / 59 |
| `clean11fv` | 95% | 96% | **96%** | 76 / 51 / 23 |

`clean11fv` is the control that matters — same levels, same values, same split,
differing **only** in training ρ. So the collapse is the correlation, not the
task. (`clean11` agrees but changes two variables, so it is a secondary check.)

**3. The failure is structured, not noise.** Same levels, same bins: the control
is flat across ρ in every margin band; `maze11` loses 64 points in the 0.15–0.35
band and 31 on the easiest decisions. It captures decisions that should be easy.
→ `fig3_margin`

**4. It needs a proxy, and follows competence.** Returns hide it — the three test
correlations are indistinguishable. Subtract them: the proxy run's ρ=1.0 − ρ=0.0
gap settles at **+0.165**, the control's at **−0.001**, and neither moves before
competence appears at ~20M. → `fig4_dynamics` *(this figure alone comes from
cleanba's in-training eval, so only the within-run gap is comparable)*

**5. Plans are linearly decodable.** A linear 1×1 conv on the recurrent state at
t=0, predicting which cells the agent will later step on. → `fig5_example_plan`

| run | trained | untrained net | observation |
|---|---|---|---|
| `smoke5b` | 0.993 | 0.562 | **0.871** |
| `maze11` | **0.967** | 0.523 | 0.583 |
| `clean11fv` | 0.910 | 0.513 | 0.582 |
| `clean11` | 0.916 | 0.541 | 0.600 |

Three controls carry this, and all three were needed: an **untrained network of
the same shape** (a random conv tower already has a receptive field, so beating
the observation proves nothing); **distance-matched negatives** (pooled, a pure
`−BFS-distance` feature reproduces the whole profile — matched, it scores exactly
0.500 while the real probe stays flat at 0.955–0.983 out to step 12); and a
**bootstrap over episodes** (cells share a maze, so cell-level intervals are 1.6×
too narrow). Think=4 takes `maze11` to 0.999.

#### Not established

- The probe reads **routes, not targets** — where the agent will step, not which
  objective it wants. The project hypothesis is about the target.
- **Correlational only.** No patching, steering or causal test.
- **n=1 seed** everywhere.
- The 5×5 row proves little: seven free cells, so the observation alone gets 0.871.
- Checkpoints are dense (1M) only to 20M; the gap grows over 20→40M at 10M spacing.

#### Next

1. Which levels does `maze11` defect on? Free — existing rollouts.
2. Does test-time compute change the *choice*, not just probe sharpness? ~1h.
3. Target probe, tested per-level: at checkpoint *t*, does the representation
   already encode colour-dependence on levels where behaviour defects at *t+k*?
4. 50M run with 1M checkpointing to resolve the transition (~3.5h).
5. Dose-response: train ρ ∈ {0.7, 0.8, 0.9, 1.0}.

---
## Links
[[4YP Working Notes]] · [[4YP Literature Hub]] · [[Bush2025_EmergentPlanning]] · [[Mini2023_MazePolicy]]
