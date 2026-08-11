# Raw run output

Verbatim output of the measurement scripts, kept so a number in a figure or a
writeup can be traced to the run that produced it. Nothing here is edited by
hand; the figures read the JSON in `figures/data/`, not these files.

| File | Produced by |
|---|---|
| `remeasure-2026-08-04.txt` | `experiments/002_measure_proxy.py` on the first three agents, plus `scripts/optimum.py` |
| `probe-2026-08-04.txt` | `experiments/003_probe_plan.py` on `maze11` and `clean11` |
| `remeasure-clean11fv.txt` | both scripts on `clean11fv`, the single-variable control |
| `probe-smoke5b.txt` | `experiments/003_probe_plan.py` on the 5×5 agent |
| `value-axis-full.txt` | `experiments/014_value_axis_analysis.py` on the value grid, last checkpoint of each arm |
| `value-axis-quarter.txt` | the same at the first checkpoint, a quarter of the fine-tune |
| `value-axis-heldout.txt` | `--leave-one-out`: each value written from an axis fitted without its arm |
| `value-or-gap.txt` | `experiments/015_value_or_gap.py` on both sweeps |
| `value-or-gap-cross.txt` | the same with `--cross`, which turned out not to separate the hypotheses |
| `three-objective.txt` | `experiments/016_three_objective_values.py` on the three-objective grid |
| `three-objective-gate.txt` | why the base agent was swept around at 80M rather than trained further |

All four agents have current behavioural and probe numbers. The 5×5 probe is
the least informative of them: with seven free cells and two-step routes, a
linear readout of the observation alone already reaches 0.871, so the trained
network's 0.993 has little headroom to be impressive in.

Both were run against the datasets fingerprinted `d0e70f346ac4a46a` (fixed
values) and `ec1665e28a0f54c2` (randomised values), generated 2026-08-04.
Anything measured before that date is in `archive/`, and does not load against
current code.

## The value grid

`scripts/value_axis_grid.sh` fine-tunes `novalue11` onto seven values for
colour 1, one arm each, identical in seed, rate and updates. The datasets share
their layouts and differ only in what the objectives pay, and every arm is
scored on the same held-out split at the base values, so the readout is common
to all of them.

The `v050` arm changed nothing and is the drift control: it lands at 7.7 extra
steps against the base's 7.7, so at this budget there is no behavioural drift to
subtract. Its weight diff is nonetheless as large as any other arm's, which is
the whole difficulty — see the note in `goalmisgen/analysis/weights.py` on why
the fit needs an intercept.

The three files above have gymnasium's deprecation warnings filtered out and
nothing else altered.

## Value or gap

`COLOUR=0 scripts/value_axis_grid.sh` sweeps colour 0 over the same gaps the
colour 1 sweep covered, so the two are directly comparable.

The knob is real either way: both axes reproduce held-out arms of their own
sweep, and each writes the other's arms to within about a tenth of a step.

Whether it is a *value* or the *gap* is not settled here. The discriminating
number is `cos(axis_0, axis_1)`, which is -0.255 raw. Both axes are fitted from
diffs that are mostly behaviourally inert, and their split-half reliabilities are
0.144 and 0.151, so the cosine is attenuated toward zero whichever hypothesis
holds. Corrected it reads -1.73, consistent with the -1 that one shared knob
predicts and not with the 0 that two value slots predict, but the correction
multiplies by nearly seven and cannot carry a conclusion.

`--cross` was added to settle it causally and does not: raising colour 0 and
lowering colour 1 by the same amount leave the same gap, so both hypotheses
predict the agreement it found. The file is kept as the record of a test that
does not do what it was built to do. Separating a value from the gap needs a
task whose choice does not reduce to one difference.

## Three objectives

`scripts/three_objective.sh` and its composition stage train a base agent at
values (1.0, 0.65, 0.3) with no value channel, then move one objective's value
at a time over offsets of plus and minus 0.1 and 0.2, and finally move two at
once as arms held out of every fit.

The behavioural composition test carries the result. Adding two single-value
axes, neither fitted on the arm being predicted, reproduces what that arm
learned in both exchange rate and choice quality — including the drop to 83%
optimal on the arm where the values were brought closest together, which the
composed edit reproduces to a tenth of a point.

Every weight-space statistic in that file is uninformative and should not be
quoted. Split-half reliability came out at 0.085, 0.044 and 0.061, so the axes
are a few per cent signal, and the attenuation-corrected cosines fall outside
the range a cosine can take — which is the correction announcing that it has
broken down rather than a number to interpret. Arms here are 1M steps against
the two-objective grid's 3M, and offsets reach 0.2 rather than 0.4; signal
scales with the offset while noise does not, so a fourfold loss of
signal-to-noise was to be expected before the shorter arms are even counted.

The base agent is `cp_70103040` rather than the 80M steps it trained for; see
the note in `goalmisgen/configs/presets.py` on `with_final_checkpoint`.

## Where the axis lives, and how many knobs there are

`experiments/017` and `018` describe the two-objective axis rather than testing
it. Two results from `018` are worth carrying forward.

Channels differ, mildly and repeatably. The axis is enriched about 2.2x in
channels 7 and 1 of `cell_list_0`, with a secondary set around 1.2 to 1.5 and a
median channel below 1. Gates show nothing at all: input, candidate, forget and
output all sit between 0.92 and 1.06. The channel preference replicates across
the colour-0 and colour-1 sweeps as well as two halves of a single sweep
replicate each other, so it is structure rather than sampling error.

The signed comparison settles what `015` left open. Restricted to the channels
that carry the axis, and with both sides fitted from the same number of arms,
the cosine between the two sweeps' axes is -1.00 within a few per cent, at every
subset of channels and in both gate convolutions. Raising one objective's value
and raising the other's move the same weights in exactly opposite directions, so
this agent holds one knob and not two value registers: what the axis writes to
is the gap, or equivalently a threshold on the difference in distances.

That is the expected answer for a two-objective task, where the choice turns on
a single difference and one scalar suffices. It also gives the three-objective
grid a quantitative prediction. Three axes constrained to hold only differences
sum to zero, which for symmetric axes puts every pairwise cosine at -0.5; three
absolute registers put them near 0; one shared knob puts them at -1, where the
two-objective agent sits.
