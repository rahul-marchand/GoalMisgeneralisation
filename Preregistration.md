---
date: 202608131200
title: 4YP Preregistration - Robustness Campaign
status: registered, no campaign run yet
tags: [4yp, preregistration]
---

# Preregistration — the robustness campaign

Written **before** any run in the campaign is launched, and not edited
afterwards. Amendments go at the bottom, dated, with the reason.

## Why this file exists

Two results in `Experiment2.md` are currently worth less than they look, for the
same reason in both cases: the model was built after the numbers were seen.

- The three-objective hierarchy predicts `cos(axis_1, axis_2) = 0.530` against
  `0.534` observed, and a variance ratio of 1.59 against 1.60 observed. The
  writeup already flags this — *"that last figure is n = 1 and the model was
  built after seeing the numbers"*.
- The 84/16 split between the first and second dimensions was read off one
  agent.

A prediction that is written down before the data exists is a different kind of
evidence from one fitted to it. Everything below is a number this campaign could
come back with and prove wrong.

Registered predictions use the observed seed-1234 and seed-5678 values as the
best current estimates. Where the two seeds disagree, that disagreement is
itself registered as a prediction.

## What is being decided in advance

**Unit of analysis.** The seed. Every claim of the form "DRC agents trained this
way do X" gets an interval across base agents, not across arms and not across
checkpoints. Checkpoints within one run are correlated and are not replicates;
they may be used for stability checks and never for an interval.

**Primary metric.** Held-out behavioural prediction error, in extra steps —
each arm's exchange rate predicted from an axis fitted without that arm. It is
measured over thousands of episodes and needs no reliability correction.

**Weight-space cosines get a permutation null**, not an attenuation correction:
arm labels shuffled within a sweep, the axis refitted, and the observed cosine
read against that distribution. The attenuation correction stays in the output
as a secondary number, but no headline claim rests on it. It is being demoted
because in `results/three-objective.txt` it returned cosines outside the range a
cosine can take, which is a correction announcing that it has broken down.

**Stopping rule.** Arm counts are fixed by `goalmisgen/design.py` before
launch. Sweeps are not extended because a result is nearly significant, and not
truncated because it already is. A sweep cut short by time is reported with the
arms it got, which is why the design orders arms widest-offset first.

## Phase 2 — wide sweeps on the two existing seeds

The design raises leverage `Σ(oᵢ - ō)²` from 0.28 to 3.049, about 11×.

| quantity | current | predicted |
|---|---|---|
| split-half reliability of an axis | 0.14 (3M arms), 0.27–0.29 (750k) | **0.75–0.85** |
| raw `cos(axis_0, axis_1)` | −0.255 (1234), −0.428 (5678) | **more negative than −0.75, both seeds** |
| disattenuated `cos(axis_0, axis_1)` | −0.99, −0.98 | unchanged, −1.0 ± 0.1 |

The reliability prediction is the load-bearing one, and it is a prediction about
the *method*, not the agent: it assumes `r/(1-r) ∝ leverage`, extrapolated from
a single observed point. **If reliability comes back below 0.5 at 11× the
leverage, that model is wrong**, and the right response is to say so and
investigate what else scales with arm count, not to reach for the correction
again.

## Phase 3 — base-checkpoint ladder

Arms fine-tuned from `cp_70M` and `cp_100M` against the existing `cp_140M`.

- **Prediction: the axis direction is settled well before training ends.**
  `cos(axis@70M, axis@140M) > 0.8` once both sides are corrected to a common
  reliability.
- Falsified if that cosine is below 0.5, which would mean the axis is still
  being built at 140M and Phase 7 is measuring something real rather than drift.

## Phase 4 — the Exp1/Exp2 bridge

Value grids on `maze11`, which has a value channel, against `novalue11`, which
does not.

- **Prediction: `‖axis‖` on `maze11` is under half of `novalue11`'s**, per unit
  of value, at matched arm length and matched leverage. An agent that can read
  what an objective is worth off its input has no reason to compile the constant
  into its weights, and fine-tuning it onto a new value should therefore move
  much less.
- **Prediction: the null arms drift by about as much as on `novalue11`.** Drift
  is the cost of running the updates and should not care whether there is a
  value to learn. If `‖axis‖` falls and drift does not, the drop is about value
  representation; if both fall, it is about the fine-tune being easier.
- Secondary, and genuinely uncertain: whether the axis's loading on the *colour*
  channels predicts how far that agent's choice accuracy collapses as ρ is swept
  from 1.0 to 0.0. No effect size registered — this is the first look.

## Phase 5 — third two-objective seed

What replicated across seeds 1234 and 5678, and should again:

- **Exchange rate**: −15.21 and −15.63 extra steps per unit of value, against a
  task-optimal −20.0. Predicted for the third seed: **−15.4 ± 1.5**, i.e. three
  quarters of optimal, an over-valuing of distance.
- **One knob, not two registers**: `cos(axis_0, axis_1)` at −1.0 ± 0.1.

What did *not* replicate, and is predicted not to:

- **Where the agent sits when values are equal**: 7.7 steps against 8.5, implied
  step penalties 0.065 and 0.059 against a true 0.05. Predicted: the third seed
  lands somewhere in 7.0–9.5 and **does not** match either. Same slope,
  different intercept — each agent settles at its own indifference point and
  then trades at the same rate away from it.
- **Which channels carry the value**: seed 1234's ch07 and ch01 appear nowhere
  in seed 5678's top eight; cross-seed top-8 overlap is 3 of 8 against 2.0 by
  chance. Predicted: the third seed's top-8 overlap with either existing seed is
  **2–4 of 8, i.e. chance**, while its own two sweeps agree with each other at
  6–7 of 8 as both existing seeds do.

That last pair is the interesting one, because it is a positive claim in
negative clothing: **localisation is real and reproducible inside one network
and carries no information about any other network trained identically.** With
n = 2 that is indistinguishable from a measurement too noisy to replicate. With
n = 3 at reliability 0.8 it is a finding.

## Phase 6 — three-objective replication

Two new base agents at the uneven values (1.0, 0.55, 0.4), new seeds, same 80M
steps. Predictions from the hierarchical model in `Experiment2.md`, which was
fitted to one agent and has never been tested out of sample:

| quantity | seed 1234 observed | predicted for each new seed |
|---|---|---|
| `cos(axis_1, axis_2)` | 0.534 | **0.53 ± 0.12** |
| variance in the 2nd dimension | 16% | **12–22%** |
| axis_1's share along the dominant direction | 93% | 88–97% |
| axis_2's share along the dominant direction | 81% | 72–90% |
| base agent chose optimal | 92.2% | 89–95% |

Registered alternatives, so the result cannot be read as confirmation whatever
it says:

- Three **absolute value registers** predict pairwise cosines near 0.
- Three axes constrained to hold only **differences** sum to zero, which for
  symmetric axes puts every pairwise cosine at −0.5.
- **One shared knob**, where the two-objective agent sits, puts them at −1.
- The **hierarchical** account predicts +0.53, and is the only one of the four
  that predicts a positive cosine at all.

So the sign alone separates the hierarchical account from all three rivals, and
that does not depend on the reliability correction.

**Also registered: the evenly spaced design stays one-dimensional.** On (1.0,
0.65, 0.3) the observed values were `cos = +0.93` and 3% in the second
dimension. If a new seed at the even values produces a large second dimension,
the "even spacing lets one stored number solve the task" explanation is wrong.

## Phase 7 — 100M extension

`maze11`'s return over 125–150M is +0.3579 against +0.3550 over 70–100M: a gain
of +0.003 across 50M steps, under 1% of what 20M→40M bought. Behaviour on the
training distribution has flattened.

- **Prediction: the axis keeps moving after behaviour stops.** Specifically
  `cos(axis@150M, axis@250M) < cos(axis@100M, axis@150M)` is *not* expected;
  rather ‖axis‖ continues to grow while the exchange rate moves by less than 0.5
  steps. This is the same "representation moves when behaviour does not" claim
  as the early-warning experiment, at the other end of training.
- Falsified if the axis is indistinguishable at 150M and 250M, which is what
  Phase 3 predicts and which would make this phase a null worth reporting in one
  line.

## What would make me abandon the account

Stated now so it cannot be renegotiated later:

- `cos(axis_0, axis_1)` reliably positive, or near zero at reliability above
  0.7, in any two-objective agent. That is two value registers and the one-knob
  claim is wrong.
- Held-out behavioural prediction error not better than predicting every arm by
  the grand mean. That would mean the axis is fitting noise regardless of what
  the cosines say.
- The three-objective pairwise cosines coming back negative on both new seeds.
  The hierarchical account is then wrong in the direction of the difference-only
  account, and the 84/16 story goes with it.

## Links

[[4YP Experiment 1 - Toy Model Building]] · [[4YP Experiment 2 - Utility Threshold]]

## Amendments

**2026-08-14 — Phase 7 ran early, and one premise it was written on was wrong.**

Recorded because the value of this file is that departures from it are visible.
No prediction below has been changed; these are deviations in procedure and one
correction of fact.

1. **Phase 7 ran before Phase 3.** It was written as gated on the
   base-checkpoint ladder, on the reasoning that if the axis is settled by
   cp_70M then Phase 7 is a predicted null. It was run first at the author's
   request. Phase 3 is still outstanding, so the gate that would have made this
   run's outcome predictable was never evaluated. That weakens what a null
   result here would mean, and should be stated wherever it is reported.

2. **It is not a warm restart.** Phase 7 above says any extension "is a warm
   restart and must be labelled as one", inferred from a checkpoint directory
   holding only `cfg.json` and `model`. That inference was wrong: `model` is the
   serialised `TrainState`, which carries `opt_state`. The base checkpoint holds
   87 opt_state leaves, 60 non-zero, at step 219072, and
   `finetune_with_noop_head` is False so the optimiser is never rebuilt. Adam's
   moments were carried across.

3. **Learning rate was not specified in advance and had to be chosen.** The
   original run annealed 4e-4 to 4e-6 over 150M and sat near 3e-5 at 140M, and
   the schedule is undefined past 150M. A constant 1e-4 was chosen — the
   fine-tuning rate — which makes the run exactly a null arm taken to its limit
   and puts its drift in the units every arm's drift is measured in. It does not
   produce the agent a 250M anneal would have.

4. **Recorded before the axis is fitted, so it cannot be read post-hoc:**
   return fell from +0.3735 to a minimum of +0.2956 at the rate change and
   recovered to +0.3573, against the base run's ~+0.358. Training performance
   ended where it started, which is the premise the experiment rests on.

*(no further amendments)*
