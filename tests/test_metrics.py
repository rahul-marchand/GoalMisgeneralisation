"""Tests for probe scoring.

The calibration identity is the one that matters. It is what lets us say whether
a low R² means the network's magnitudes are wrong or only that the fit shrank
them — a distinction the distance-field pilot could not make.
"""

from __future__ import annotations

import ast
import pathlib

import numpy as np
import pytest

from goalmisgen.analysis import metrics

ANALYSIS = pathlib.Path(metrics.__file__).parent


def test_the_calibration_identity_is_exact():
    """r2 == shape_r2 - scale_loss - offset_loss, for arbitrary predictions."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        y = rng.normal(size=200) * rng.uniform(0.5, 5.0) + rng.uniform(-10, 10)
        prediction = y * rng.uniform(-2, 2) + rng.normal(size=200) * rng.uniform(0, 3) + rng.uniform(-5, 5)

        split = metrics.calibration(y, prediction)
        assert split.r2 == pytest.approx(split.shape_r2 - split.scale_loss - split.offset_loss, abs=1e-9)
        assert split.r2 == pytest.approx(metrics.r2(y, prediction), abs=1e-9)


def test_a_shrunk_prediction_is_reported_as_shrunk():
    """Halving the spread about the mean keeps the ordering perfect and costs
    exactly a quarter of the variance. This is the signature of a penalised fit,
    and the decomposition must name it rather than hiding it in a low R²."""
    y = np.arange(100, dtype=np.float64)
    prediction = y.mean() + 0.5 * (y - y.mean())

    split = metrics.calibration(y, prediction)
    assert split.shape_r2 == pytest.approx(1.0), "the ordering is perfect and must be reported as such"
    assert split.r2 == pytest.approx(0.75)
    assert split.slope == pytest.approx(2.0), "the prediction must be stretched by 2 to fit"
    assert split.bias == pytest.approx(0.0)
    assert split.scale_loss == pytest.approx(0.25)


def test_shape_r2_ignores_any_affine_map():
    """The ordering ceiling must not move when the prediction is rescaled."""
    rng = np.random.default_rng(1)
    y = rng.normal(size=300)
    prediction = y * 0.7 + rng.normal(size=300) * 0.5

    base = metrics.calibration(y, prediction).shape_r2
    for scale, offset in ((3.0, 0.0), (0.1, 20.0), (-2.0, -5.0)):
        assert metrics.calibration(y, scale * prediction + offset).shape_r2 == pytest.approx(base, abs=1e-9)


def test_recalibration_recovers_the_ordering_ceiling():
    """Fitting scale and offset on one split and applying to another must lift
    R² to shape_r2 — that is what makes shape_r2 the honest upper bound."""
    rng = np.random.default_rng(2)
    y_train, y_test = rng.normal(size=400) * 4 + 12, rng.normal(size=400) * 4 + 12
    shrink = lambda v: v.mean() + 0.3 * (v - v.mean())  # noqa: E731

    scale, offset = metrics.affine_fit(y_train, shrink(y_train))
    corrected = scale * shrink(y_test) + offset

    assert metrics.r2(y_test, corrected) == pytest.approx(metrics.calibration(y_test, shrink(y_test)).shape_r2, abs=0.02)


def test_stratified_correlation_removes_what_the_stratum_determines():
    stratum = np.repeat(np.arange(20), 8)
    assert abs(metrics.stratified_correlation(stratum * 2.5, stratum * -1.5 + 3.0, stratum)) < 1e-9


def test_stratified_correlation_needs_comparable_rows():
    """One member per stratum is missing power, not a zero result."""
    values = np.random.default_rng(0).normal(size=20)
    assert np.isnan(metrics.stratified_correlation(values, values, np.arange(20)))


def test_stratified_auc_only_counts_comparable_pairs():
    """A score that ranks perfectly inside every stratum must reach 1.0 even
    when it ranks badly across them — that is the whole point of matching."""
    stratum = np.repeat([0, 1], 6)
    y = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1], dtype=np.float64)
    # Stratum 1's scores are all below stratum 0's, so the pooled AUC is poor.
    scores = np.array([10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5], dtype=np.float64)

    matched = metrics.stratified_auc(y, scores, stratum)
    pooled = metrics.roc_auc(y, scores)
    assert matched == pytest.approx(1.0)
    assert pooled < matched, f"the between-stratum offset should cost the pooled AUC: {pooled:.3f} vs {matched:.3f}"


def test_a_pure_function_of_the_stratum_scores_at_chance_under_matching():
    """The classification form of the confound that invalidated the first
    distance-band result."""
    rng = np.random.default_rng(3)
    stratum = rng.integers(0, 8, 400)
    y = (rng.random(400) < 0.5).astype(np.float64)
    assert metrics.stratified_auc(y, stratum.astype(np.float64), stratum) == pytest.approx(0.5, abs=1e-9)


def test_paired_bootstrap_is_tighter_than_comparing_two_intervals():
    """Two statistics that share between-episode variance must have a much
    tighter interval on their difference than either has alone."""
    rng = np.random.default_rng(4)
    episode = np.repeat(np.arange(40), 10)
    shared = np.repeat(rng.normal(size=40) * 5.0, 10)
    a, b = shared + 1.0, shared

    stat_a = lambda rows: float(a[rows].mean())  # noqa: E731
    stat_b = lambda rows: float(b[rows].mean())  # noqa: E731

    low_a, high_a = metrics.bootstrap_episodes(stat_a, episode)
    low_d, high_d = metrics.bootstrap_paired(stat_a, stat_b, episode)

    assert (high_d - low_d) < 0.2 * (high_a - low_a), "pairing failed to cancel the shared variance"
    assert low_d < 1.0 < high_d, "the true difference must lie inside the interval"


def test_uncertainty_is_estimated_over_episodes_not_cells():
    episode = np.repeat(np.arange(12), 30)
    values = np.repeat(np.random.default_rng(0).normal(size=12), 30)

    low, high = metrics.bootstrap_episodes(lambda rows: float(values[rows].mean()), episode)
    assert (high - low) > 4 * values.std() / np.sqrt(len(values))


def test_the_scoring_layer_does_not_know_about_mazes():
    """metrics takes (n,) arrays and nothing else. If it learns what a wall is,
    porting this to another environment stops being two new files."""
    tree = ast.parse((ANALYSIS / "metrics.py").read_text())
    imported = [
        name.name
        for node in ast.walk(tree)
        for name in (node.names if isinstance(node, (ast.Import, ast.ImportFrom)) else [])
    ] + [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]

    leaked = [name for name in imported if "envs" in name or "geometry" in name]
    assert not leaked, f"the scoring layer imported environment knowledge: {leaked}"


def test_a_capture_is_collected_once_however_many_arms_read_it():
    """Reading a different grid off the same rollouts is free; collecting them
    is not. The pilot gathered four sets per think value where two would do."""
    from goalmisgen.analysis.activations import Capture, RolloutCache

    calls = []
    cache = RolloutCache(lambda capture, seed, n: calls.append((capture.name, seed)) or [f"{capture.name}:{seed}"])

    trained = Capture("trained", reader="agent")
    untrained = Capture("untrained", reader="random")
    for _ in range(5):
        cache.get(trained, seed=0, n_episodes=8)
    cache.get(untrained, seed=0, n_episodes=8)
    cache.get(trained, seed=9999, n_episodes=8)

    assert cache.collections == 3, f"expected 3 distinct collections, made {len(calls)}"


def test_arms_in_one_table_must_share_an_actor():
    """Different actors mean different episodes and different labels, so any
    difference between arms would be unattributable."""
    from goalmisgen.analysis.activations import Capture, require_one_actor

    require_one_actor([Capture("a", reader="x"), Capture("b", reader="y")])
    with pytest.raises(ValueError, match="share an actor"):
        require_one_actor([Capture("a", reader="x"), Capture("b", reader="y", actor="other")])
