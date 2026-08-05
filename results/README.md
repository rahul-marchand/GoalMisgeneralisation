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

All four agents have current behavioural and probe numbers. The 5×5 probe is
the least informative of them: with seven free cells and two-step routes, a
linear readout of the observation alone already reaches 0.871, so the trained
network's 0.993 has little headroom to be impressive in.

Both were run against the datasets fingerprinted `d0e70f346ac4a46a` (fixed
values) and `ec1665e28a0f54c2` (randomised values), generated 2026-08-04.
Anything measured before that date is in `archive/`, and does not load against
current code.
