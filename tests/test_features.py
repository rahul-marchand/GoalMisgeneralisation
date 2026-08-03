"""Tests for feature schemes — the proxy relationship under study.

The chance-level test matters most: with more than two objectives, a
correlation of 0.5 is *not* an uninformative control, and treating it as one
would mean reporting "the agent followed colour even when colour carried no
information" about a cue that was genuinely informative.
"""

from __future__ import annotations

import numpy as np
import pytest

from goalmisgen.envs.features import CorrelatedFeatures


def held_by_best(scheme: CorrelatedFeatures, values, trials=4000, seed=0) -> float:
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(trials):
        ids = scheme.assign(values, rng)
        hits += ids[int(np.argmax(values))] == 0
    return hits / trials


@pytest.mark.parametrize("correlation", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_realised_correlation_matches_the_requested_one(correlation):
    assert held_by_best(CorrelatedFeatures(correlation), (1.0, 0.5)) == pytest.approx(correlation, abs=0.03)


def test_chance_level_depends_on_objective_count():
    """0.5 is the uninformative baseline only for two objectives."""
    scheme = CorrelatedFeatures(0.5)
    assert scheme.chance(2) == pytest.approx(0.5)
    assert scheme.chance(3) == pytest.approx(1 / 3)
    assert scheme.chance(4) == pytest.approx(0.25)


def test_correlation_at_chance_is_uninformative_only_for_two_objectives():
    """Concretely: rho=0.5 with three objectives beats chance by 50%."""
    three = CorrelatedFeatures(0.5)
    realised = held_by_best(three, (1.0, 0.6, 0.3))
    assert realised == pytest.approx(0.5, abs=0.03)
    assert realised > three.chance(3) + 0.1, "rho=0.5 is a correlated condition here, not a control"


def test_features_other_than_zero_carry_no_value_information():
    """Only feature 0 may be predictive; the rest must be exchangeable."""
    scheme = CorrelatedFeatures(1.0)
    rng = np.random.default_rng(0)
    counts = {1: 0, 2: 0}
    trials = 4000
    for _ in range(trials):
        ids = scheme.assign((1.0, 0.6, 0.3), rng)
        counts[ids[1]] = counts.get(ids[1], 0) + 1  # which feature the middle value got
    assert counts[1] / trials == pytest.approx(0.5, abs=0.04)


def test_canonical_neutralises_the_proxy_but_keeps_the_draw_count():
    """Datasets are generated with the canonical scheme so layouts do not depend
    on the correlation; it must still consume the same random draws."""
    scheme = CorrelatedFeatures(0.3)
    assert scheme.canonical().correlation == 1.0

    a, b = np.random.default_rng(0), np.random.default_rng(0)
    scheme.assign((1.0, 0.5), a)
    scheme.canonical().assign((1.0, 0.5), b)
    assert a.integers(1_000_000) == b.integers(1_000_000), "streams diverged"


def test_ties_are_broken_at_random():
    scheme = CorrelatedFeatures(1.0)
    rng = np.random.default_rng(0)
    holders = [scheme.assign((1.0, 1.0), rng).index(0) for _ in range(400)]
    assert set(holders) == {0, 1}


def test_rejects_out_of_range_correlation():
    with pytest.raises(ValueError, match="correlation must be in"):
        CorrelatedFeatures(1.5)


def test_single_objective_is_always_feature_zero():
    assert CorrelatedFeatures(0.3).assign((1.0,), np.random.default_rng(0)) == (0,)
