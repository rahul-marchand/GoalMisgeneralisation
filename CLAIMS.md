# What is claimed, and what holds it up

`RUNS.toml` answers *why does this run exist*. `results/README.md` answers *what
produced this file*. Neither answers the question that decides what to run next:
**what supports this claim, at what n, and would another run help?**

So this is an index from claim to evidence. It is maintained by hand — unlike
`MANIFEST.md`, there is nothing to generate it from — and the rule is that a
claim which appears in a writeup appears here, with its weakest link named.

`n` is the number of **independently trained base agents**, because that is the
unit of analysis for anything of the form "DRC agents trained this way do X".
Arms are not replicates of a seed and checkpoints are not replicates of a run.

Status: **solid** (would survive a hostile reading), **thin** (true as far as it
goes, n or precision too low to carry weight), **provisional** (one agent, or
fitted after the fact).

---

## Experiment 1 — the task and the proxy

| # | claim | evidence | n | status |
|---|---|---|---|---|
| E1.1 | A DRC(3,3) agent trades value against distance, and the plan is linearly readable from the recurrent state | probe 0.910–0.993 against 0.582–0.871 on the observation | 4 | **solid** |
| E1.2 | The plan is *writable*: writing the route to the other objective moves the exchange rate 8.3 → 6.7 steps, its own route → 10.0, at 100% reach | `results/steer-variants.txt`, fig 3 | 1 | **thin** |
| E1.3 | The write enters the decision rather than overriding it — followed where switching is cheap, refused where expensive | fig 3 right panel | 1 | **thin** |
| E1.4 | Training at ρ=1.0 produces goal misgeneralisation; the ρ=0.5 control is flat at every ρ | fig 4, fig 5 | 1 proxy + 2 controls | **solid** |
| E1.5 | The proxy is learnt *after* competence, not alongside it | fig 6, gap settles at +0.165 vs control −0.001, neither moves before ~20M | 1 proxy + 1 control | **thin** |

**Weakest link: E1.5**, which is the closest thing the project has to its
guiding hypothesis and rests on one agent per condition. It is also the cheapest
to strengthen — `eval_at_steps` saves ~20 checkpoints in the first 20M steps on
all five 11×11 agents, so the pre-competence window is already on disk and needs
inference, not training. That is Phase 0.

Note what E1.5 does *not* claim: it compares behaviour against behaviour. "An
internal representation predicts misgeneralisation before it appears in
behaviour" is not yet claimed anywhere, because no experiment tests it.

---

## Experiment 2 — the value axis

| # | claim | evidence | n | status |
|---|---|---|---|---|
| E2.1 | Weight diffs across a value grid are collinear and graded — there is an axis | `results/value-axis-full.txt` | 2 | **thin** |
| E2.2 | The axis is writable: an axis fitted without an arm reproduces that arm | `--leave-one-out`, mean error 0.53 steps (s1234), 0.77 (s5678) | 2 | **solid** |
| E2.3 | Norm-matched random directions of equal length do nothing | same | 2 | **solid** |
| E2.4 | **One knob, not two value registers**: `cos(axis_0, axis_1) = −1` | raw −0.255 / −0.428, disattenuated −0.99 / −0.98 | 2 | **thin** |
| E2.5 | Saturation is one-sided — the axis carries ~2× its fitted range when deepening an existing preference, flattens when reversing it | fig 3 (`results/value-axis-ood.txt`) | 1 | **provisional** |
| E2.6 | The apparent preference flip is a broken policy, not an inversion — reach 83/69/35/8% | same | 1 | **thin** |
| E2.7 | Exchange rate is ~75% of task-optimal: −15.21 and −15.63 against −20.0 | `results/seed-comparison.txt` | 2 | **thin** |
| E2.8 | Where an agent sits at equal values does **not** replicate: 7.7 vs 8.5 steps | same | 2 | **thin** |
| E2.9 | Channel localisation is real within a seed (top-8 overlap 6–7 of 8 across disjoint sweeps) and carries **no** information across seeds (3 of 8, against 2 by chance) | `results/which-channels.txt`, `seed5678-which-channels.txt` | 2 | **thin** |
| E2.10 | Restricting the write to the enriched channels does not reproduce the effect — no channel-level circuit | `results/restricted-axis.txt` | 1 | **thin** |

**Weakest link: E2.4**, which is the headline. Raw cosines of −0.26 and −0.43
are divided by split-half reliabilities of 0.14–0.29 to reach −1. The
correction is nearly ×7 and it, not the measurement, is carrying the claim. This
is a *design* problem rather than a sample-size one — leverage `Σ(oᵢ-ō)²` is
0.28 — and Phase 2 addresses it directly at 6 GPU-hours for both existing
seeds. See `goalmisgen/design.py`.

**E2.9 is a positive claim in negative clothing** and is worth more than it
looks: localisation is reproducible inside one network and predicts nothing
about another trained identically. At n=2 and reliability 0.27 that is not
distinguishable from "too noisy to replicate". At n=3 and reliability 0.8 it is
a finding.

---

## Experiment 2 — three objectives

| # | claim | evidence | n | status |
|---|---|---|---|---|
| E2.11 | Evenly spaced values (1.0, 0.65, 0.3) collapse onto one dimension: `cos(axis_1, axis_2) = +0.93`, 3% in the second | `results/three-objective.txt` | 1 | **provisional** |
| E2.12 | Unevenly spaced (1.0, 0.55, 0.4) produce a second dimension at 16% | `results/three-objective-wide.txt` | 1 | **provisional** |
| E2.13 | The structure is hierarchical, predicting `cos = 0.530` against 0.534 observed and a variance ratio 1.59 against 1.60 | same | 1 | **provisional, fitted after the fact** |
| E2.14 | Composition: adding two axes, neither fitted on the held-out arm, reproduces it — including 83.0% optimal predicted at 83.1% | `threeobj/runs/m_*` | 1 | **thin** |
| E2.15 | Arms are measurably worse at their own task the further their values sit from an arithmetic progression | `results/own-task-threeobj2.txt` | 1 | **thin** |

**Weakest link: E2.13.** `Experiment2.md` says it outright — *n = 1 and the
model was built after seeing the numbers*. Every weight-space statistic in
`results/three-objective.txt` is additionally uninformative: split-half
reliabilities of 0.085, 0.044 and 0.061 put the attenuation-corrected cosines
outside the range a cosine can take.

This is the most interesting claim in the project and the least supported one.
Phase 6 replicates it on two new seeds against predictions registered in
`Preregistration.md` before launch. The sign alone separates the hierarchical
account from all three rivals — absolute registers predict ~0, difference-only
predicts −0.5, one shared knob predicts −1, and only the hierarchical account
predicts a positive cosine.

---

## Not yet claimed

Things the project is set up to say and has not tested. Listed so the gap is
visible rather than implied.

- **An internal representation moves before misgeneralisation appears in
  behaviour.** The guiding hypothesis in `CLAUDE.md`. E1.5 compares behaviour
  against behaviour. Phase 0 and Phase 4.
- **Whether an agent that can read value off its input stores it at all.** Every
  value-axis result is on `novalue11`, which has no value channel; every
  misgeneralisation result is on `maze11`, which does. The two experiments have
  never been run on the same agent. Phase 4.
- **Whether the writable threshold is a property of planning.** Would need a
  DRC(1,1) or feedforward agent. Not budgeted.
- **Whether the axis survives a shift in the distance distribution.** The claim
  is that the axis writes a threshold on the difference in distances; that
  predicts something about levels drawn from a different distance profile, and
  nothing has tested it.

---

## Maintaining this

Add a row when a writeup gains a claim, not when a run finishes. Update `n` and
`status` when a phase lands. If a claim's weakest link is not a run that could
be launched, say so — "needs a different task" is a legitimate and useful
answer, and is what `results/value-or-gap-cross.txt` records for the
value-versus-gap question that three objectives eventually settled.
