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
## Links
[[4YP Working Notes]] · [[4YP Literature Hub]] · [[Bush2025_EmergentPlanning]] · [[Mini2023_MazePolicy]]
