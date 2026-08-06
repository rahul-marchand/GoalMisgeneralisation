---
date: 202607241500
title: 4YP Experiment 1 - Toy Model Building
status: stage 1 complete
tags: [4yp, experiment]
---

# Experiment Bush replication + extensions


## Goal

Building small toy models that cleanly isolate key features of model behaviour seems a useful way to test broader hypotheses about training and behaviour as this project goes on. A sort of "model-organism" of goal misgeneralisation. E.g. [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html), [Nanda et al. grokking/progress measures](https://arxiv.org/abs/2301.05217).

In the literature there are some simple mechanistic examples of goals being isolated/goal misgeneralisation: https://arxiv.org/abs/2310.08043. 

Also some clean examples of planning circuits being isolated in models: (https://arxiv.org/abs/2504.01871 ). and some interesting follow up papers and blog posts [Taufeeque et al. 2024 (arXiv)](https://arxiv.org/abs/2407.15421).

My goal here is to cleanly isolate competing goals in a model. i.e. a mechanism that weighs up two or more different objectives and can be intervened on to make the model switch from one target to another.

## Tldr Plan

Maze navigation using a DRC agent seemed to give clean results here: https://arxiv.org/abs/2504.01871. Maze navigations seems simple to setup, has a clean ground truth and is easy to build multiple objectives into. So we replicate (Appendix C of [Tom Bush Blogpost](https://tuphs28.github.io/projects/interpplanning/)) and then extend to our multi-objective setup. 

## Setup/Task Details

### Environment / Agent Setup

DRC(3,3) -- ConvLSTM, 3 layers × 3 ticks per env step -- trained with IMPALA
(cleanba, unmodified) for 150M steps, γ=0.995. The DRC architecture is
[Guez et al. 2019](https://proceedings.mlr.press/v97/guez19a.html), designed with
the intent of cleanly producing planning.

11×11 maze, two objectives, episode ends on reaching either. Reward is the
objective's value minus 0.05 per step, so **utility = value − 0.05 × distance**
and the nearer objective often wins. 120-step limit.

Observation is symbolic 11×11×5. ch0 walls, ch1 agent, ch2 feature 0, ch3 feature 1, ch4 value.

ch2 and ch3 exist so I can directly create an explicit spurious correlation. The task would work with ch4 alone. It seems more interesting to indirectly force a correlation and have the proxy goal be something the agent comes up with but I use this direct setup as a starting point.

1M pre-generated levels, test/train/valid = 50k/900k/50k, disjoint. All numbers below are the test split, held out from training *and* from in-training eval.

![](figures/fig1_task.png)
*Figure 1: the task and its symbolic encoding. Left: the rendered maze, never passed to the agent. Right: the five observation channels. Here feature 0 is worth 1.0 at 11 steps and feature 1 is worth 0.5 at 4 steps, so feature 0 is the optimal choice.*

The four runs referred to in initial runs below:

| run | size | train ρ | values | steps |
|---|---|---|---|---|
| `smoke5b` | 5×5 | 1.0 | fixed | 10M |
| `maze11` | 11×11 | 1.0 | fixed | 150M |
| `clean11fv` | 11×11 | 0.5 | fixed | 150M |
| `clean11` | 11×11 | 0.5 | randomised | 130M |

## Initial Runs

### Verify setup works

- Show agent learns to navigate maze
- Show linear probes pick up planning

Trained successfully to maximise reward on a 5x5 and 11x11 maze. All four runs reach an objective 100% of the time, and `clean11fv` picks the higher-utility one 95% of the time at every ρ.

Linear probes successfully extracted plans. The agent learns to plan to move to the further away objective if reward is high enough. A linear 1×1 conv on the recurrent state at t=0, predicting which cells the agent will later step on:

| run         | linear probe on activations | linear probe on observation (control) |
| ----------- | --------------------------- | ------------------------------------- |
| `smoke5b`   | 0.993                       | 0.871                                 |
| `maze11`    | 0.967                       | 0.583                                 |
| `clean11fv` | 0.910                       | 0.582                                 |
| `clean11`   | 0.916                       | 0.600                                 |


![](figures/fig5_example_plan.png)
*Figure 2: probe scores at t=0 on four test levels, with the route actually taken overlaid. The agent walks past the nearer objective when the further one is worth more (left two panels) and takes the nearer one when it is not (right two).*

### Summary of Initial Simple Misgeneralisation Experiment

I set a direct, explicit proxy objective. **ρ = P(feature 0 marks the higher-value objective).** I fix ρ during training, and sweep it at test. ρ=1.0 makes the channel a perfect cue, ρ=0.5 is chance (uninformative), ρ=0.0 inverts it (i.e. perfect proxy the other way around).

`chose_optimal` at ρ = 1.0 / 0.5 / 0.0, 2,048 episodes per arm:

| run | 1.0 | 0.5 | 0.0 | followed feature 0 |
|---|---|---|---|---|
| `smoke5b` | 100% | 50% | **0%** | 100 / 100 / 100 |
| `maze11` | 96% | 79% | **60%** | 77 / 67 / 59 |
| `clean11fv` | 95% | 96% | **96%** | 76 / 51 / 23 |

![](figures/fig2_rho_response.png)
*Figure 3: left to right -- 5x5 and 11x11 trained at ρ=1.0 (feature 0 always marks the higher-value objective), then 11x11 control trained at ρ=0.5 (channel varied randomly).*

`clean11fv` is the control on the right hand side (trained with randomised channel)

![](figures/fig3_margin.png)
*Figure 4: % chose optimal by utility margin, on identical levels. The control is flat across ρ in every band; the proxy run loses 64 points in the 0.15–0.35 band and 31 on the easiest decisions.*

Interestingly in the 11x11 maze the model learns to complete the task optimally before it learns the direct proxy. See figure 5 below.

![](figures/fig4_dynamics.png)
*Figure 5: returns hide the effect — the three test correlations are indistinguishable. Subtracting them, the proxy run's ρ=1.0 − ρ=0.0 gap settles at +0.165 and the control's at −0.001, and neither moves before competence at ~20M. From cleanba's in-training eval, so only the within-run gap is comparable.*

## Stage 2 — what the agent computes (2026-08-06)

All on `clean11fv`, the control agent. Probes are linear 1x1 maps on the
ConvLSTM state, fitted on `valid` and scored on `test`.

### A distance field, to both objectives

At every free cell the recurrent state linearly encodes **that cell's
shortest-path distance to an objective** — a field over the whole maze, not a
scalar at the goal. Partial correlation, after conditioning away straight-line
distance:

| | d→f0 | d→f1 |
|---|---|---|
| trained | **0.705** | 0.394 |
| untrained, same architecture | 0.041 | 0.025 |
| straight-line geometry only | 0.054 | 0.056 |

Both objectives have a field, including the one the agent doesn't go to. So
the choice is made **after** both distances exist — the substrate a utility
comparison needs. Accuracy is ~3.5 cells on a field averaging 12; it degrades
where walls force long detours (r 0.84 → 0.52).

Extra thinking time **prunes the road not taken**: at think=4 the unchosen
objective's field falls 0.394 → 0.241 while the chosen one barely moves. That
also explains the pilot's puzzle, where thinking appeared to degrade the field
— it was discarding the objective being abandoned.

### The asymmetry is value, not channel, distance or utility

- **Not channel.** Re-running at ρ=0.0, so feature 0 marks the *lower*-value
  objective, mirrors the gap: 0.391 / 0.706.
- **Not utility.** The utility split *reverses sign* on the 212 levels where
  value and utility disagree.
- **Not distance.** On the 205 levels where the objectives are within 2 steps
  of each other, richer vs poorer is 0.773 / 0.423, **[+0.301, +0.397]**.

One alternative is open, and only the untrained control revealed it: on that
same subset an untrained tower still gives [+0.047, +0.133]. The richer
objective carries a literally larger number in the value channel, so its
neighbourhood has stronger activations. The trained effect is 4x larger
(difference of differences +0.265), but *value-weighted computation* and
*amplified input magnitude* are not separated. Separating them needs
`clean11`, where values vary continuously.

### The exchange rate it learned

Behavioural, no probe. The agent abandons the richer objective at **7.8 extra
steps** [7.7, 8.0]; the task's rate puts it at 10.0, or 9.3 once γ=0.995 is
accounted for. Implied step penalty **0.064** against the true 0.05 — it
over-weights distance by ~28%.

The threshold is near-deterministic: 100% below +2, 0% above +14, the whole
transition in one bin. And it is **identical at ρ=1.0 and ρ=0.0** (7.8 vs 7.9),
so the decision is about value, not colour.

**This accounts for the agent's entire error budget.** A 7.8 threshold predicts
4.2% wrong choices; measured `chose_optimal` is 95.4%. The ~4% suboptimality
carried since stage 1 is this one mis-learned exchange rate.

### Infrastructure

A target declares `labels` **and** `confound`, and `controls()` generates the
oracle, null and shuffled arms from that declaration — so a probe question
cannot be asked without the controls that make it readable. `check_rig` raises
rather than warns. Metrics split ordering from calibration exactly
(`R² = ρ² − (ρ−k)² − b²`). No `Environment` protocol: the port boundary is
`geometry.py` + `targets.py` and a test enforces it.

### Not established

- Everything is **correlational**. Nothing has been intervened on.
- The value-vs-amplitude alternative above.
- n=1 seed; only the control agent — the proxy agent has not been probed.

## Going Forwards

I think the questions I am interested in sort of cluster into two groups:

(1) What form does goal direction takes mechanistically in agents?

(2) How do goals form in agents? Which proxy goals do agents tend to learn? So more of a learning dynamics direction. Still need to read a lot more here to be comfortable with my understanding of the literature.



---
## Links
[[4YP Working Notes]] · [[4YP Literature Hub]] · [[Bush2025_EmergentPlanning]] · [[Mini2023_MazePolicy]]
