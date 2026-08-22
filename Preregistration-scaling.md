---
date: 202608221200
title: 4YP Preregistration - The width/depth scaling campaign
status: registered, no run launched yet
tags: [4yp, preregistration, scaling, offline-bc]
---

# Preregistration — the width/depth scaling campaign

Written **before** any model in the grid is trained, and not edited afterwards.
Amendments go at the bottom, dated, with the reason. The robustness campaign has
its own file, `Preregistration.md`; this is a separate campaign with separate
predictions and nothing here amends that one.

## Why this file exists

`results/offline-bc-value-axis-bcnv11.s*` established that a 0.80M-parameter
prefix-LM trained by imitation holds what an objective is worth as a single
writable direction in weight space: split-half reliability 0.963, implied-vs-
trained slope 0.97, `cos(axis_0, axis_1) = -0.98`. That is the cleanest version
of the value axis anywhere in this project, cleaner than either RL architecture.

The question this campaign asks is whether that survives scale, because the
reason to care about a route model at all is that it is the LLM-shaped member of
the family. A one-knob goal representation in a four-layer network says little
about a thirty-two-layer one if the knob is an artefact of having nowhere else
to put it.

The campaign is exploratory and is registered anyway, because the failure mode
it is most exposed to is telling a story about a monotone curve after seeing it.

## What is being decided in advance

**The grid.** Three widths by three depths, `d_model/n_heads = 32` and
`mlp_ratio = 4` throughout, so body parameters are exactly `12 d^2 L`:

| | d=128 | d=256 | d=512 |
|---|---|---|---|
| **L=4** | 0.80M | 3.17M | 12.63M |
| **L=8** | 1.59M | 6.32M | 25.22M |
| **L=16** | 3.16M | 12.61M | 50.38M |

The anti-diagonals carry **two parameter-matched pairs** — (256,4) against
(128,16), and (512,4) against (256,16). The transformer stack is `12 d^2 L` and
so is *exactly* equal within each pair; the totals differ by 0.4% because the
embeddings and head scale with `d_model` alone (3.171M against 3.158M, and
12.634M against 12.608M). Those two pairs are the load-bearing comparisons: any
difference within a pair is shape, not capacity, and needs no cross-scale
normalisation to be read.

**Unit of analysis.** The base model. One seed at first; seeds are added only
after seed 1 of the whole grid has been read, and only on the rows or columns
that moved. A grid point is not a replicate of its neighbours.

**No repeated data, decided in advance and not negotiable.** Every model sees
`batch 1024 x 30,000 steps = 30.7M` demonstrations, each from a level no model
in the grid sees twice. The existing `bcnv11` bases were trained with ~8.5
epochs of repeat and are **not** reused as the (128,4) cell; that cell is
retrained under this protocol. Overfitting is the one confound that would make
a probe or an axis result uninterpretable, and it is being removed by
construction rather than argued away afterwards.

**Fixed training recipe**, identical at every grid point except shape: batch
1024, LR 6e-4 (sqrt-scaled from the 3e-4 used at batch 256), cosine schedule,
warmup 500, weight decay 0.01, clip 1.0, `nn.remat` on the transformer block.
Remat is on everywhere, including where it is not needed, so compute per step is
comparable across the grid.

**Arms.** Twelve mirrored offsets per sweep, two sweeps (`o0`, `o1`), widest
offset first so a sweep cut short by time is still balanced. Arm demonstrations
are also unrepeated, which requires ~1.1M levels per arm value.

**The arm budget is anchored behaviourally, not by step count.** At batch 1024 a
1000-step fine-tune is four times the samples of the published `bcnv11` arms, and
a fixed step budget means a different-strength fine-tune at every width because
the same learning rate is a different-sized step at a different `d_model`. Each
arm is instead trained to a target shift in measured exchange rate. Numbers from
this campaign are therefore **not comparable to the published `bcnv11` arm
tables**, and no claim will pair them.

**Nulls.** Weight-space cosines are read against the permutation null of
`goalmisgen.analysis.weights.permutation_cosines`, never against zero, for the
reason given in `Preregistration.md`. Cosine-based quantities additionally get a
**per-shape** null, because random-vector cosine falls as `1/sqrt(d)` and a
cleanliness improvement with width would otherwise be partly free.

**Competence is a covariate, not a constant.** The (128,4) base sits at 0.929
optimal, so it is not saturated and larger models will be better at the task.
Every headline metric is reported twice: at the final checkpoint, and at a
checkpoint matched on held-out optimal rate across the grid. The log-spaced
checkpoint schedule already on disk supplies the second. **If a depth effect
disappears under competence matching, it was competence**, and will be reported
as such.

## What is measured

Four things, deliberately separable, because they can dissociate:

1. **How many axes.** Participation ratio of the drift-removed arm family, and
   the split-half reliability of the leading residual direction after the fitted
   axis is removed. The task has **one goal degree of freedom** — `015`/`028`
   showed the knob is the gap between values, not the values — so the ground
   truth is rank one and any replicating second component is the network's doing.
2. **Whether it can still be read.** Leave-one-out held-out R², implied-vs-
   trained slope, split-half reliability of the axis.
3. **Whether it can still be written.** Mean `|written - arm|` in steps on
   held-out levels, and the written-value-against-exchange-rate transfer curve.
4. **Where it lives, and whether one place is enough.** The axis refitted inside
   each block alone; that block's share of `||axis||^2`; and the behavioural
   write performed with the axis restricted to a subset of blocks.

Measurements 1 and 4 do not currently exist and are built by this campaign.

## Registered predictions

Stated as calls, so they can be wrong.

**P1 — width is benign.** Split-half reliability of the axis stays at or above
0.90 across `d = 128, 256, 512` at fixed `L = 4`. *Falsified* if it falls below
0.80 at `d = 512`.

**P2 — depth breaks the write before it breaks the read.** At `d = 512`, the
leave-one-out held-out R² at `L = 16` is within 0.10 of its value at `L = 4`,
while the leave-one-out write error in steps is **at least twice** as large.
*Falsified* if write error is flat in depth, or if the read degrades at least as
much as the write.

This is the campaign's main prediction and the reason it is worth running. A
representation that stays linearly decodable while becoming uncontrollable is
precisely the reported failure mode of steering vectors in language models, and
here the ground truth is known to be a single scalar.

**P3 — the rank stays one.** The leading residual direction fails to replicate:
split-half reliability below 0.30 at every grid point. *Falsified* by a
replicating second component above 0.50, which would mean the network splits a
one-dimensional goal parameter into two knobs — an interesting positive result,
and the one outcome that would redirect the campaign.

**P4 — the axis delocalises with depth.** The largest single block's share of
`||axis||^2` falls monotonically in `L` at fixed `d`. And the behavioural form:
at `L = 4` a write restricted to a single block reaches within 2 steps of the
full-axis write; at `L = 16` no single block does, and the number of blocks
required to get within 2 steps grows with `L`. *Falsified* if a single block
suffices at `L = 16`.

**P5 — the matched pairs separate, deep member worse on the write.** At both
3.17M and 12.6M, the `L = 16` member has strictly larger write error than the
`L = 4` member. This is the cleanest test in the campaign because capacity is
held exactly and no cross-scale normalisation enters. *Falsified* by either pair
coming back equal or reversed.

Taken together: **D-shaped along width, B-shaped along depth**, in the outcome
vocabulary the design was chosen from. The diagonal of the grid may therefore
show nothing, because the two effects run opposite ways — which is why the
campaign is a grid and not a ladder, and why a flat diagonal will not be reported
as "scale does not matter".

## What counts as an axis

Carried over unchanged from `Preregistration.md` Phase 3, so the bar is the same
one the DRC results were held to. A grid point has an axis when writing
`offset x axis` into that model's own weights moves the measured exchange rate
further than the measurement's uncertainty: 95% intervals at the two extreme
writes disjoint, the model still reaching objectives at >= 95%, and a
norm-matched random direction of the same length not moving it. Norms and
cosines are reported but do not decide.

A base that cannot do the task is reported as "base cannot do the task", not as
"no axis". This is not expected anywhere in this grid, since the smallest cell
is already competent, and is stated so that it cannot be invoked later.

## What would make this campaign uninformative, and what happens then

If all nine cells come back at reliability ~0.96, slope ~0.97, rank one and
write error ~1 step, the conclusion is **not** "goal representations are robust
to scale". It is that **63x in parameters on a task a 0.80M model already solves
at 93% never stressed the representation.**

The response registered in advance is to add task, not parameters: three
objectives (`016`) gives two goal degrees of freedom and turns "is it rank one?"
into "does the rank track the truth?", which is a much harder thing for a
network to pass by accident. Committing to that now is what makes a flat result
informative rather than a wasted week.

## What would make me abandon the account

- A replicating second component (P3 falsified) at the *smallest* grid point.
  That would mean the published `bcnv11` one-knob result was an artefact of
  twelve arms and repeated data, not a property of the model.
- Write error not better than predicting every arm by the grand mean, at any
  cell whose base is competent. The axis would be fitting noise whatever the
  cosines say.
- The depth effect of P2 and P5 vanishing entirely under competence matching,
  *and* the raw effect being large. That would mean this campaign measured how
  good the models are at mazes and called it a fact about goal representations.

## Links

`Preregistration.md` (robustness campaign) · `Experiment2.md` · `CLAIMS.md`

## Amendments

**2026-08-22 — the arm budget: behavioural *calibration of the learning rate*,
not behavioural *stopping*. Registered before any model was trained.**

As first written, this file said each arm would be "trained to a target shift in
measured exchange rate". Implementing it surfaced a flaw that would have
manufactured the campaign's most interesting result.

Stopping each arm when it reaches its own target means arms run for **different
numbers of steps**. But the axis is fitted as `diff = drift + offset * axis`,
and the drift term is the movement every arm shares *because it was fine-tuned
at all* — which grows with the number of steps taken. Arms at wide offsets need
longer, so under a behavioural stopping rule the drift would scale with
`|offset|`. That dependence is symmetric in the sign of the offset, so it does
not enter the axis; it lands in the **residual**, which is exactly where P3 looks
for a replicating second component. The rule would have planted a second
direction and then found it, at every grid point, more strongly in the shapes
that train slowest. Reported as a depth effect, it would have been an artefact of
the stopping rule.

The fix keeps the motivation and moves where the behaviour is read:

- **Within a shape, every arm runs the same fixed number of steps** (cap 1000, at
  batch 256), so the drift is common and the intercept can absorb it, as it does
  in every previous campaign.
- **Across shapes, the arm learning rate is calibrated behaviourally.** For each
  of the nine cells, the widest positive arm is trained at each of 3e-5, 1e-4 and
  3e-4, and the rate whose achieved exchange rate lands closest to that arm's
  expert target is used for all of that cell's arms. This is what the original
  rule was for — the same learning rate is a different-sized step at a different
  `d_model`, and an uncalibrated grid would report "big models moved less" as
  "big models have a cleaner axis".
- The calibration reads one arm's behaviour and sets one scalar per cell. It
  cannot leak into the axis, which is fitted from how the arms differ from *each
  other* at a fixed budget.

The consequence already registered above stands unchanged and now has a second
reason: these arm numbers are not comparable to the published `bcnv11` tables.

**Registered prediction about the calibration itself**, so it is not a free
parameter: the chosen rate falls with width, roughly as `1/d_model`, and does not
vary systematically with depth. If instead the same rate wins at every cell, the
calibration was unnecessary and the fixed-rate result would have been sound — say
so rather than claiming the calibration mattered.


**2026-08-22 (later) — three protocol decisions corrected by measurement, before
any cell was trained.**

Each was registered on reasoning that a probe on an H100 then falsified.
Recorded in full because the value of this file is that departures are visible.

1. **Batch 1024 becomes batch 256.** The registered batch was justified by "a
   larger batch is needed to feed the card". Measured throughput is *flat* from
   batch 256 to 4096 — 20.9, 22.4 and 23.1 TFLOP/s at (128,4) — so the workload
   was never launch-bound and the argument was simply wrong. At batch 256 the
   models see 7.68M samples in 30,000 steps, which is the same count the
   published `bcnv11` recipe saw, and the campaign costs a quarter as much.

2. **30.7M demonstrations becomes 7.68M.** Not a concession to cost. An 11x11
   grid holds a 5x5 cell lattice and the generator reaches only ~455,000
   distinct mazes (`results/maze-diversity.txt`), a ceiling reached well before
   7M draws. The extra 23M demonstrations would have added *zero* new layouts
   and only new goal placements. The registered plan to find the plateau with a
   data-scaling pilot is therefore withdrawn: the plateau is a property of the
   generator, and it has been measured directly rather than inferred from a
   training curve.

   The consequence for **P2 and P5** is that the largest cells are trained on a
   pool whose layouts recur about seventeen times. This does not weaken those
   predictions but it does change what confounds them, which is why:

3. **Splits hold out mazes, not levels.** `dataset.split_indices` dealt level
   indices, so with layouts recurring, nearly every held-out level was a fresh
   placement on a maze the model trained on. `goalmisgen/envs/splits.py` assigns
   whole mazes, and assigns them **by hashing the maze rather than partitioning
   the pool** — because pools in a value sweep share layouts, and splitting each
   on its own contents put 10% of an arm's training mazes into the base pool's
   test set, which is where every exchange rate in this campaign is measured.
   That was a leak straight into the primary metric and it is now zero.

   **A prediction this makes possible, registered now.** With disjoint mazes the
   train/test gap is a direct measure of memorisation, and capacity is the axis
   the grid varies. Prediction: the gap in held-out `chose_optimal` stays under
   0.02 at (128,4) and grows with parameter count, exceeding 0.05 at (512,16).
   Falsified if the largest cell shows no more gap than the smallest — which
   would settle the overfitting worry rather than confirming it. Either way this
   is reported alongside the axis results, not instead of them.

*(no further amendments)*
