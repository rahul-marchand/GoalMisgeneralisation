---
date: 202608231200
title: 4YP Preregistration - The depth sweep
status: registered, no run launched yet
tags: [4yp, preregistration, depth, offline-bc]
---

# Preregistration — the depth sweep

Written before any model is trained. Amendments at the bottom, dated. The
width/depth campaign has its own file; this is its successor and inherits its
data, its controls and its mistakes.

## Why this exists, and why it is not another parameter sweep

`results/scaling-campaign.txt` ran nine cells over 63x in parameters and found
that the value representation's *fidelity* — how faithfully writing an offset
moves the exchange rate — is a function of competence and nothing else:
`correlation(competence, |slope-1|) = -0.995`. Eight of nine cells sat at or
above 0.99 competence, where that metric is pinned at ceiling, so the campaign
had no resolution on the question it was built for.

What did vary among competent models is **where the axis lives**. The number of
modules that must be written to reproduce the full axis tracks depth at +0.968
and parameter count at **-0.004 once divided by depth**. At fixed depth,
quadrupling the parameters changed it not at all (L=4: 3.5 -> 3.5 modules across
4x; L=8: 6.0 -> 5.8 across 4x).

So parameter count is the wrong x-axis. This sweep changes the axis and keeps
the range.

## What is being decided in advance

**The grid.** Fixed `d_model = 256`, `n_heads = 8`, `mlp_ratio = 4`. Depth
doubles: **L = 2, 4, 8, 16, 32, 64**, which is 1.6M to 50.4M parameters — the
same span the last campaign covered, reached by varying depth instead of width.
Three seeds at every depth except L=64, which gets two.

**One learning rate for every arm, fixed in advance, not calibrated.** At fixed
width there is no width-dependent step-size problem, and the calibration that
was supposed to solve one in the last campaign instead drove its headline metric
(`correlation(log lr, reliability) = -0.856`). A single rate is chosen once from
a five-point ladder run at L=2 and L=64 only; if those two disagree by more than
one rung, a rule `lr(L)` is fitted and registered here before the sweep runs.
Nothing is calibrated per cell.

**One sweep, o0 only.** The two sweeps agreed to 0.021 in the primary metric
against a between-depth signal of 0.25, so the second buys nothing and costs
half the arms and half the analysis. `cos(axis_0, axis_1)` is therefore not
measured here; it was -0.95 +- 0.02 across every competent cell of the last
campaign and is not in question.

**Primary metric: modules needed**, the number of modules that must be written,
largest share first, for `base + offset * axis` restricted to them to land
within 2 steps of the full-axis write. Reported both absolutely and divided by
depth. Fidelity — competence, transfer slope — is reported as a *precondition*,
not a result: a cell below 0.99 competence or with `|slope-1| > 0.15` is
excluded from the depth fit and said to be excluded.

**Data is reused unchanged** from the width/depth campaign: the same 7.68M
single-epoch demonstrations, the same 24 arm pools, the same hash-based
layout-disjoint splits. Nothing is regenerated, so nothing about the data can
differ between the two campaigns.

## Registered predictions

**D1 — the absolute count grows sublinearly.** Modules needed follows roughly
`L^0.8`: about 3.5 at L=4, 6.0 at L=8, 10 at L=16 are already observed, so the
prediction is for the new points — about 2 at L=2, 17 +- 3 at L=32, and 29 +- 6
at L=64. *Falsified* by linear growth (which would give 28 and 56) or by
saturation (a flat count past L=16).

**D2 — the fraction keeps falling and does not asymptote above 0.4.** Observed
0.88, 0.74, 0.63 at L = 4, 8, 16. Predicted below 0.55 at L=32 and below 0.50 at
L=64. *Falsified* if the fraction flattens at or above 0.60, which would mean
delocalisation is a fixed proportion of the network rather than a growing
absolute cost.

**D3 — parameter count still does nothing at fixed depth.** The control at
d=128, L=32 (6.3M against the primary sweep's 25.2M at the same depth) lands
within one seed's spread of the d=256 cell. *Falsified* by a gap larger than the
between-seed scatter, which would overturn the finding this sweep is built on.

**D4 — one module is never enough, at any depth.** The best single module leaves
the exchange rate at least 2 steps from the full write, at every L including 2.
*Falsified* by any depth where a single module suffices.

**Registered as unknown.** Whether L=64 trains stably at all. Sixty-four pre-LN
layers at this width is past anything this project has run, and a failure to
converge is a result about the recipe rather than about representations. It is
checked with a short stability probe before the seeds are committed.

## What would make this uninformative

If the fraction is flat across L = 2 to 64, the metric is measuring the fixed
cost of writing a distributed direction rather than anything about depth, and
the right response is to say so rather than to reach for a different
normalisation after the fact.

## Links

`Preregistration-scaling.md` · `results/scaling-campaign.txt` · `CLAIMS.md`

## Amendments

*(none yet)*
