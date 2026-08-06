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

## Going Forwards

I think the questions I am interested in sort of cluster into two groups:

(1) What form does goal direction takes mechanistically in agents?

(2) How do goals form in agents? Which proxy goals do agents tend to learn? So more of a learning dynamics direction. Still need to read a lot more here to be comfortable with my understanding of the literature.



---
## Links
[[4YP Working Notes]] · [[4YP Literature Hub]] · [[Bush2025_EmergentPlanning]] · [[Mini2023_MazePolicy]]
