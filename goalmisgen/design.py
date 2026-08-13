"""How many arms to train, and at what offsets.

The value axis is a fitted slope: ``diff = offset * axis``. So the precision of
everything downstream — the cosine between two sweeps, the channel profile, the
held-out write — is set by

    Var(axis) ∝ σ² / Σ(oᵢ - ō)²

and the denominator is the only part the experimenter controls. Call it the
*leverage* of a design. The first grid had seven arms over offsets −0.2 to +0.4,
giving a leverage of 0.28, and produced split-half reliabilities of 0.14 to 0.29.
Every cosine measured from it was attenuated toward zero by roughly a factor of
four, and the headline result had to be recovered by dividing by that — a
correction of nearly seven, which is the correction carrying the claim rather
than the measurement.

This is a problem of design, not of sample size. Halving the spacing over the
same range adds arms that each contribute almost nothing, because contribution
goes as the *square* of the offset. ``Experiment2.md`` already observed the same
thing from the other end: the axis share of ‖Δθ‖² runs 4% at an offset of 0.1 and
54% at 0.4. Signal grows with the offset and noise does not.

So the design here spends arms on range, not resolution:

* **Most arms at the extremes.** For estimating a slope, the optimal design puts
  every point at the ends of the usable interval.
* **Clustered near the extremes rather than repeated on them.** The ideal is
  several arms at exactly ±0.45, whose spread would measure ‖ε‖ directly. That
  is not available: ``014``'s ``arm_checkpoints`` keys arms by the value they
  were trained at, so two arms at one value collide in that dict and the second
  silently replaces the first. A tight band — 0.38 to 0.45, eight distinct
  offsets — recovers 91% of the leverage that exact repetition would give, keeps
  every key distinct, and needs no change to the analysis. The spread among
  near-neighbours still estimates the noise floor, just with a little signal
  mixed in.
* **Some interior coverage.** Not for precision — for detecting that the map is
  curved, which a design confined to one band cannot do at all, and which the
  writes outside the fitted grid say is real.
* **Symmetric about the base.** With offsets balanced around zero the common
  fine-tuning component cannot leak into the fitted axis. That lesson was paid
  for by the first two-objective grid, whose asymmetry produced a
  confident-looking null.
* **A cluster at zero.** The null arms are the drift floor, and one of them is
  a single measurement of a quantity every other arm is compared against.

The default design gives 25 arms at a leverage of 3.05, about eleven times the
first grid. Carrying the observed ``r/(1-r) ∝ leverage`` across, that is a
predicted split-half reliability near 0.8, where the attenuation correction
becomes a footnote instead of the argument.

The range is bounded by the task rather than by taste: the offsets must not
reverse which objective is worth more, because at that point the arm is learning
a different preference and not a further step along the same one.
"""

from __future__ import annotations

from dataclasses import dataclass

from goalmisgen.volume import arm_dirname

# Offsets from the base value, positive side; mirrored, plus a single null arm
# at zero. The band from 0.38 to 0.45 carries the leverage, and the four sparse
# points below it are what would show the map bending.
DEFAULT_OFFSETS: tuple[float, ...] = (0.45, 0.44, 0.43, 0.42, 0.41, 0.40, 0.39, 0.38, 0.30, 0.20, 0.10, 0.05)


@dataclass(frozen=True)
class Arm:
    """One fine-tune: which objective moved, by how much, and at what seed."""

    sweep: str
    objective: int
    offset: float
    seed: int

    def dirname(self, steps: int) -> str:
        return arm_dirname(self.sweep, self.offset, steps)


def leverage(offsets: list[float]) -> float:
    """``Σ(oᵢ - ō)²`` — the quantity the axis's precision is inversely proportional to."""
    if not offsets:
        return 0.0
    mean = sum(offsets) / len(offsets)
    return sum((o - mean) ** 2 for o in offsets)


def sweep_arms(
    objective: int,
    offsets: tuple[float, ...] | None = None,
    first_seed: int = 1234,
) -> list[Arm]:
    """Every arm of one objective's sweep, deepest offsets first.

    ``offsets`` are the positive side; each is mirrored, and one null arm is
    added at zero to measure drift.

    Ordered by ``|offset|`` descending so that an interrupted sweep still has the
    arms that carry the most leverage. A sweep cut off half way through should
    degrade to a smaller version of itself rather than to its least informative
    half.
    """
    offsets = DEFAULT_OFFSETS if offsets is None else offsets
    if any(o <= 0 for o in offsets):
        raise ValueError(f"give the positive side only; it is mirrored for you, got {offsets}")
    if len(set(offsets)) != len(offsets):
        raise ValueError(
            f"offsets must be distinct: 014 keys arms by the value they trained at, so a repeat "
            f"would silently replace its twin. Got {offsets}"
        )
    arms: list[Arm] = []
    seed = first_seed
    for magnitude in sorted(offsets, reverse=True):
        for offset in (magnitude, -magnitude):
            arms.append(Arm(sweep=f"o{objective}", objective=objective, offset=offset, seed=seed))
            seed += 1
    arms.append(Arm(sweep=f"o{objective}", objective=objective, offset=0.0, seed=seed))
    return arms


def arm_values(base_values: tuple[float, ...], arm: Arm) -> tuple[float, ...]:
    """What the objectives pay in one arm's levels."""
    if not 0 <= arm.objective < len(base_values):
        raise ValueError(f"objective {arm.objective} does not exist among {base_values}")
    values = list(base_values)
    values[arm.objective] = round(values[arm.objective] + arm.offset, 10)
    return tuple(values)


def check_no_preference_flip(base_values: tuple[float, ...], arms: list[Arm]) -> list[str]:
    """Arms whose values reorder the objectives, which makes them a different task.

    Past the flip the agent is not being pushed further along a preference, it is
    being asked to hold the opposite one, and the fitted map is not linear
    through that point — the writes outside the fitted grid already fail exactly
    there, and at 8% reach rather than by giving a wrong number.
    """
    ranking = sorted(range(len(base_values)), key=lambda i: base_values[i], reverse=True)
    problems = []
    for arm in arms:
        values = arm_values(base_values, arm)
        if sorted(range(len(values)), key=lambda i: values[i], reverse=True) != ranking:
            problems.append(f"{arm.sweep}{arm.offset:+.2f} gives {values}, reordering the objectives")
    return sorted(set(problems))
