"""List the value-axis arms of the offline-BC stream, one per line, for shell loops.

    uv run python scripts/value_axis_arms.py [--steps 1000] [--objective 0 1]

Each line: ``sweep offset seed dirname values_tag``. The design is
``goalmisgen.design.sweep_arms`` - the same symmetric 25-arm grid per objective
the DRC value-axis campaign used - at base values (1.0, 0.5). The
``values_tag`` names the level dataset ``levels/values/<tag>@150k`` the arm's
demonstrations come from.
"""

from __future__ import annotations

import argparse

from goalmisgen.design import arm_values, check_no_preference_flip, sweep_arms
from goalmisgen.volume import values_tag

BASE_VALUES = (1.0, 0.5)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", type=int, default=1000, help="Fine-tuning steps; part of the arm's directory name.")
    parser.add_argument("--objective", type=int, nargs="+", default=[0, 1])
    args = parser.parse_args()
    for objective in args.objective:
        arms = sweep_arms(
            objective, first_seed=1234 + 100 * objective
        )  # distinct seeds per sweep, so the two null arms differ
        problems = check_no_preference_flip(BASE_VALUES, arms)
        if problems:
            raise SystemExit("\n".join(problems))
        for arm in arms:
            print(arm.sweep, f"{arm.offset:+.2f}", arm.seed, arm.dirname(args.steps), values_tag(arm_values(BASE_VALUES, arm)))


if __name__ == "__main__":
    main()
