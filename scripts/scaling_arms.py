"""List the value-axis arms of the width/depth campaign, one per line, for shell loops.

    uv run python scripts/scaling_arms.py [--steps 1000] [--objective 0 1]

Each line: ``sweep offset seed dirname values_tag target``, where ``target`` is
the exchange rate in steps that arm's expert has -- the number the learning-rate
calibration aims the widest arm at.

The same machinery as ``scripts/value_axis_arms.py``, with **six magnitudes per
sweep instead of twelve**. Halving the grid is what buys the nine shapes: the
arms, not the bases, are the campaign's cost, and this is an exploratory pass
that asks whether anything moves with shape at all. Resolution is what a second
pass adds if something does.

Which six, and why these: the endpoints of the old grid are kept because leverage
is what the fit is made of, one magnitude is kept inside the 0.38-0.45 cluster
where the model's preference flips, and the rest space the middle. Six magnitudes
mirror to twelve arms plus a null, which leaves six pairs -- above the four that
:func:`goalmisgen.analysis.weights.split_half_reliability` needs to split, with
one pair of margin if an arm fails.
"""

from __future__ import annotations

import argparse

from goalmisgen.design import arm_values, check_no_preference_flip, sweep_arms
from goalmisgen.offline.axis import expected_indifference
from goalmisgen.volume import values_tag

BASE_VALUES = (1.0, 0.5)

OFFSETS: tuple[float, ...] = (0.45, 0.42, 0.39, 0.30, 0.20, 0.10)
"""Positive side; mirrored. Six pairs and a null arm, thirteen runs per sweep."""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", type=int, default=1000, help="Fine-tuning budget; part of the arm's directory name.")
    parser.add_argument("--objective", type=int, nargs="+", default=[0, 1])
    parser.add_argument(
        "--offsets",
        type=float,
        nargs="+",
        default=None,
        help="Positive side of the grid, mirrored for you. Defaults to the registered six.",
    )
    parser.add_argument("--widest-only", action="store_true", help="Just the widest positive arm, for calibration.")
    args = parser.parse_args()
    for objective in args.objective:
        arms = sweep_arms(objective, offsets=tuple(args.offsets or OFFSETS), first_seed=1234 + 100 * objective)
        problems = check_no_preference_flip(BASE_VALUES, arms)
        if problems:
            raise SystemExit("\n".join(problems))
        if args.widest_only:
            arms = [max(arms, key=lambda arm: arm.offset)]
        for arm in arms:
            print(
                arm.sweep,
                f"{arm.offset:+.2f}",
                arm.seed,
                arm.dirname(args.steps),
                values_tag(arm_values(BASE_VALUES, arm)),
                f"{expected_indifference(BASE_VALUES, objective, arm.offset):.2f}",
            )


if __name__ == "__main__":
    main()
