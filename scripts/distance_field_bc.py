"""Per-cell distance fields in the route model's prefix: are they there at all?

    uv run python scripts/distance_field_bc.py /workspace/data/offline/runs/bcnv11.s1 \
        > results/distance-field-bcnv11.s1.txt

Every probe so far read per-level scalars at named sites; this asks the DRC's
question of the BC transformer: one shared linear readout applied at *every*
cell token, predicting that cell's own BFS distance - to the agent, and to
each objective. The DRC reference (results/distance-pilot.txt,
results/outcome-keyed.txt): partial r ~0.70, hard R2 ~0.25, MAE ~3.5 on the
field to its preferred objective; untrained ~0.04.

Why it matters here: the gap is a difference of such fields. If the fields are
strong, the steering negatives point at a spatially structured (anti-symmetric)
write; if they are weak, the BC computes distances some other way - and its
wide threshold noise has a candidate explanation.

Method is :mod:`goalmisgen.analysis.fields` unchanged: hard cells are where
Manhattan under-shoots BFS by >= 4, scored against that subset's own mean;
partial r is stratified within integer Manhattan values.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import numpy as np

from goalmisgen.analysis import fields, targets
from goalmisgen.offline.demos import DemoSet, shared_levels
from goalmisgen.offline.probe import capture
from goalmisgen.offline.train import initial_params, list_checkpoints, load_checkpoint
from goalmisgen.provenance import header

DEFAULT_PROBE_DEMOS = "/workspace/data/offline/demos/train.rho100"
DEFAULT_EVAL_DEMOS = "/workspace/data/offline/demos/test.rho100"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    parser.add_argument("--probe-demos", type=str, default=DEFAULT_PROBE_DEMOS)
    parser.add_argument("--eval-demos", type=str, default=DEFAULT_EVAL_DEMOS)
    parser.add_argument("--n-train", type=int, default=256)
    parser.add_argument("--n-test", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(header())
    print()

    probe_demos = DemoSet.load(args.probe_demos, hide_values=True)
    eval_demos = DemoSet.load(args.eval_demos, hide_values=True)
    if shared_levels(probe_demos, eval_demos):
        raise SystemExit("probe and eval demos share levels")
    model, params = load_checkpoint(list_checkpoints(args.run)[-1][1])
    untrained = initial_params(model, jax.random.PRNGKey(1))
    d_model = model.config.d_model
    depths = model.config.n_layers + 1

    train_idx = np.arange(min(args.n_train, len(probe_demos)))
    test_idx = np.arange(min(args.n_test, len(eval_demos)))
    arms = {
        "trained": (
            capture(model, params, probe_demos, train_idx, layer=None),
            capture(model, params, eval_demos, test_idx, layer=None),
        ),
        "untrained": (
            capture(model, params, probe_demos, train_idx, layer=None, reader_params=untrained),
            capture(model, params, eval_demos, test_idx, layer=None, reader_params=untrained),
        ),
    }

    # At rho=1.0 colour 0 marks the richer objective on every level.
    field_targets = (
        targets.DistanceToAgent(),
        targets.DistanceToObjective(select=targets.fixed(0), name="d->rich (colour0)"),
        targets.DistanceToObjective(select=targets.fixed(1), name="d->poor (colour1)"),
    )

    def feature_at(depth: int | None):
        if depth is None:
            return lambda r: r.features
        return lambda r: r.features[..., depth * d_model : (depth + 1) * d_model]

    print(f"{args.n_train} train / {args.n_test} test episodes; DRC reference: partial ~0.70, hard R2 ~0.25, MAE ~3.5")
    for target in field_targets:
        print(f"\n== {target.name} ==")
        print(f"  {'arm':<22} {'partial':>8} {'within-ep':>10} {'hard R2':>8} {'CI':>16} {'MAE':>6} {'pooled':>7}")
        for arm, (train_rollouts, test_rollouts) in arms.items():
            layer_list = [*range(depths), None] if arm == "trained" else [None]
            for depth in layer_list:
                label = f"{arm} {'all' if depth is None else f'depth {depth}'}"
                train = fields.cell_data(train_rollouts, feature_at(depth), target)
                test = fields.cell_data(test_rollouts, feature_at(depth), target)
                r = fields.field_probe(label, train, test)
                ci = f"[{r.hard_interval[0]:+.2f},{r.hard_interval[1]:+.2f}]"
                print(
                    f"  {r.name:<22} {r.partial_r:>8.3f} {r.partial_r_within_episode:>10.3f}"
                    f" {r.hard_r2:>8.3f} {ci:>16} {r.mae:>6.2f} {r.pooled_r2:>7.3f}"
                )
        train = fields.cell_data(arms["trained"][0], lambda r: r.observation, target)
        test = fields.cell_data(arms["trained"][1], lambda r: r.observation, target)
        r = fields.field_probe("observation", train, test)
        ci = f"[{r.hard_interval[0]:+.2f},{r.hard_interval[1]:+.2f}]"
        print(
            f"  {r.name:<22} {r.partial_r:>8.3f} {r.partial_r_within_episode:>10.3f}"
            f" {r.hard_r2:>8.3f} {ci:>16} {r.mae:>6.2f} {r.pooled_r2:>7.3f}"
        )


if __name__ == "__main__":
    main()
