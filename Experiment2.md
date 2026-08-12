---
date: 202608121000
title: 4YP Experiment 2 - Utility Threshold
status: in progress
tags: [4yp, experiment]
---

# Experiment 2 — Utility Threshold

## Tldr takeaway

Found a writable, readable one-dimensional utility threshold, not a per-objective 
value register. Raising either objective's value moves the same weights in
opposite directions (cos = −1.00), so what the fine-tunes are moving is a threshold 
on the difference in distances. 

Also tried to see if weight updates were concentrated. Found higher density in channel 1 and 7
as well as reccurence cell 0 but restricting to just these updates hurt performance significantly.

Trained an evenly-spaced three-objective task, which produces one dimension; an unevenly-spaced 
one produces a second, but it takes only 16% of the variance.

## Initial aim / hypotheses

Find a value register I can intervene on.

~achieved. Found a threshold instead

## Setups / details

Trained in the maze with the following inputs

![](figures/fig9_no_value_task.png)

*Figure 1: what the agent sees. Four channels passed in walls, the agent, and one per
objective. In the rendered maze on the left,
blue is the agent, red is colour 0 at 11 steps away and green is colour 1 at 4.
In this case colour 0 is worth the walk.


Fine-tune the same agent onto a grid of values for one objective,
then fit.

    Δθ = drift + offset × axis

across the grid. Test on an unused grid entry

I ran a second seed of this training an otherwise identical agent and reproing results on it.

I also ran two versions with three objectives.

Three objectives. (1.0, 0.65, 0.3) at 80M steps, then a redesign at
(1.0, 0.55, 0.4), also 80M. 


## Results


### Weight Changes Observed
 
Every arm ends up about the same distance from the base:  
|Δθ| = 14.7 ± 0.36 

(including the null arm at 14.4 which did not change behaviour) 

Splitting each arm into the three parts, as a share of its energy:

| offset | drift | axis | ε |
|---|---|---|---|
| 0.0, the null arm | 38% | 0% | 62% |
| ±0.2 | 38% | 15% | 47% |
| +0.4 | 33% | 54% | 13% |

drift is 8.9 and near-identical across arms; the axis is 28.5 per unit of value,
so it only overtakes everything else at the wide end of the grid; ε is what is
left, and at a typical offset it is the largest single part.

Seed 5678's arms ran 750k steps rather than 3M and everything is smaller. |Δθ|
≈ 7, drift 3.9, axis 16.4.  we have similar proportions, tilted slightly
towards the axis. Shorter fine-tunes seem to be cleaner.


### Writing a value works

Applying the axis to the non finetuned weights produces the desired behaviour,
at 100% reach throughout, and norm-matched random directions of the same length
do nothing.

![](figures/fig10_written_value.png)

Figure 2: writing a value into the unedited agent. Each written point comes
from an axis fitted on the other five arms, so nothing about the arm it predicts
went into the direction that produces it — the in-sample version cannot tell an
axis from a lookup of the arms it was built from. The dotted line is the
unedited agent. Random directions are plotted at the value their magnitude was
matched to: at v=0.8 the real edit reaches 3.3 steps while a random edit of
identical size has not moved off that line.

It fails at v=1.1, where colour 1 is worth more than colour 0 and the agent
should flip preference outright. It does not. The map is locally linear, not
globally.

Both held-out writes replicate on seed 5678, less cleanly: mean error 0.77 steps
against 0.53, and every error the same sign rather than scattered around zero.
Its axis systematically overshoots, worst at both ends of the grid — the
signature of fitting a straight line to something slightly curved, and the same
thing that shows up as the failure at 1.1.


### Where in the weights — reliable, and about one network only

The change concentrates in the input-to-hidden convolution of recurrent layer 0,
and within it in two of the 32 channels: ch07 at 2.29× and ch01 at 2.21× the
shared fine-tuning component, then a gap down to 1.31 for everything else.

That measurement is reliable. The colour-0 and colour-1 sweeps are disjoint sets
of arms and agree on the channel profile at r = +0.97, sharing 7 of their top 8
channels against 2 by chance.

The answer is not. On seed 5678 the leaders are ch15, ch04, ch25, ch16 —
flatter, no gap — and neither ch07 nor ch01 appears anywhere in its top eight.
Across seeds the overlap is 3 of 8 against 2 by chance, i.e. nothing, while
within that seed the two sweeps still agree at r = +0.79. So the method
replicates and the localisation does not: *the value lives in channels 1 and 7*
is a fact about novalue11, not about the architecture or the task.

Reading and writing also come apart. Restricting the fit to those 9,216
parameters takes reading from slope 0.03 to 0.55 — most of the value-carrying
signal really is in there. Writing only there gives 6.8 steps against a base of
7.7, where the full axis gives 2.4 — almost nothing. Concentrated enough to read
from, too distributed to write from, which is [[Hase2023_LocalizationEditing]]
turning up in our own data rather than borrowed.

The harder negative: ablating those channels in the untouched agent does nothing
beyond a random pair, and they rank 15th and 30th of 32 at predicting which
objective it takes. The channels that do predict the decision are equally inert
when ablated. What a fine-tune moves is not what the agent reads.


### threshold not value store

**cos(axis₀, axis₁) = −1.00 ± 0.03**, at every channel subset and in both gate
convolutions, with the arm counts matched on both sides of the comparison.
Raising colour 0's value and raising colour 1's move the same weights in exactly
opposite directions.
 

### Three objectives: structure forms only as far as it is forced

|  | (1.0, 0.65, 0.3) | (1.0, 0.55, 0.4) |
|---|---|---|
| gaps | 0.35, 0.35 — even | 0.45, 0.15 |
| thresholds / rank gaps | 7, 7 steps over 1, 1 | 9, 3 steps over 1, 1 |
| can one constant express it? | **yes** | **no** |
| cos(axis₁, axis₂) | +0.93 | **+0.53** |
| variance in 2nd dimension | 3% | **16%** |
| base agent chose optimal | 87.4% | 92.2% |

With ρ=1 the colour channels hand over the *ordering* for free, so what has to
be stored is the magnitudes. Evenly spaced values make every threshold a rank
gap times one constant, so a single stored number solves the task — and a short
fine-tune can only move the number that exists. That is the whole explanation
for three collinear axes, and it is a fact about the task rather than the agent.

On the redesign the second dimension appears but does not take over. The
structure is hierarchical: axis₁ lies 93% along the dominant direction and
axis₂ 81%, with their off-axis parts **opposed** — which predicts
cos(axis₁, axis₂) = 0.530 against 0.534 observed (aligned would give 0.968).
The 84/16 split tracks how often each comparison decides an episode: predicted
ratio 1.59, observed 1.60.

*That last figure is n=1 and the model was built after seeing the numbers.
Agreement to two decimals is better than the data deserve.*

### Behavioural evidence for the projection account

If the agent has one dial and the fine-tune tasks need two, arms should be
measurably worse at their own task the further its values sit from an arithmetic
progression. Controlling for difficulty — how far ahead the best objective is,
which dominates at r = +0.85 — they are:

    grid 1:  optimal = 80.7 + 22.7·topgap −  9.0·asymmetry    partial r −0.59
    grid 2:  optimal = 86.3 + 14.9·topgap − 11.5·asymmetry    partial r −0.41

The second was **pre-registered**: the model, the sign, the order of magnitude
and which term would be larger were all written down before that grid finished
training. Asymmetry alone is null in both, which is why the covariate is needed.

This test touches no weight-space quantity, so it is independent of every
estimator in the rest of the experiment.

## Links

[[4YP Working Notes]] · [[4YP Literature Hub]] · [[4YP Experiment 1 - Toy Model Building]] · [[Bush2025_EmergentPlanning]] · [[Hase2023_LocalizationEditing]]
