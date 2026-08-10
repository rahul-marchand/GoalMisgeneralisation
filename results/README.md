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
