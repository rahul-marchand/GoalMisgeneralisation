"""A colour assignment that ignores value rank, for arms that cross parity.

:class:`~goalmisgen.envs.features.CorrelatedFeatures` paints feature 0 on
whichever objective is worth *most*. Inside a value sweep's fitted window that
is invisible, because the swept objective never overtakes the other and the
colours therefore stay where they were. Past parity it is decisive: an arm
trained at ``(1.0, 1.4)`` has colour 0 painted on the 1.4 objective, so the
agent still learns "colour 0 is the valuable one" and the measured threshold
tracks the *absolute* value gap. Sweeping further does not reverse a
preference; it re-labels which objective is being preferred, and the threshold
turns around at parity instead of passing through zero:

    colour 1 trained at    1.0   1.1   1.2   1.3   1.4
    measured threshold     2.5   2.9   3.7   5.0   6.0     (|gap| / 0.05, biased low)

This scheme keeps colour ``i`` on objective ``i`` whatever they pay, so a sweep
can carry the swept objective past the other and the agent has to learn that
*this colour* is now the poorer one. That is the manipulation a signed
threshold needs.

**It cannot change what a dataset contains.** Level generation normalises the
scheme through ``FeatureScheme.canonical()`` and throws the resulting colours
away, and this class's ``canonical()`` is the ordinary correlated scheme, so
generation consumes the same random draws and stores the same layouts either
way. Colours are attached when a dataset is *loaded*. That is also why this
lives here rather than in ``features``: every module in
``dataset.CONTENT_MODULES`` is hashed into the fingerprint that guards stored
levels, and adding a class there would invalidate every dataset on the volume
to express a change that provably alters none of them.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from goalmisgen.envs.features import CorrelatedFeatures


@dataclasses.dataclass(frozen=True)
class ColourKeyedFeatures:
    """Objective ``i`` always wears colour ``i``, whatever the objectives pay."""

    def chance(self, n_objectives: int) -> float:
        """The rate at which colour 0 marks the richest objective by luck alone.

        Reported for the same reason :class:`CorrelatedFeatures` reports it: a
        misgeneralisation number is only interpretable against what following
        the colour would score without any relationship to value. Here colour 0
        marks the richest objective exactly when objective 0 happens to be the
        richest, which for a fixed value tuple is not a chance event at all --
        so this is the honest baseline only for sweeps that place the objectives
        on both sides of parity.
        """
        return 1.0 / n_objectives

    def canonical(self) -> CorrelatedFeatures:
        """What generation should use, so stored content is unchanged."""
        return CorrelatedFeatures(correlation=1.0)

    def assign(self, values: tuple[float, ...], rng: np.random.Generator) -> tuple[int, ...]:
        del rng  # nothing is drawn: the assignment is fixed
        return tuple(range(len(values)))
