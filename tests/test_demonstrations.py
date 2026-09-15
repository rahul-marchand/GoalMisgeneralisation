"""Every task's demonstration set carries the whole surface the pipeline reads.

``analysis.behaviour`` reads outcome dicts with ``.get``, so a task that forgot
a key would degrade to ``nan`` in a table rather than fail. The protocol is the
list of what the pipeline needs; this test holds each implementation to it.
"""

from __future__ import annotations

import numpy as np
import pytest

from goalmisgen.envs.dataset import LevelDataset
from goalmisgen.envs.sampling import MazeLevelSampler
from goalmisgen.envs.solver import MOVES
from goalmisgen.offline.demonstrations import PROTOCOL_ATTRIBUTES, Demonstrations, load_demonstrations, task_name
from goalmisgen.offline.demos import DemoSet


@pytest.fixture(scope="module")
def maze_demos() -> DemoSet:
    dataset = LevelDataset.generate(MazeLevelSampler(size_range=(7, 9)), n_levels=30, seed=0, block_size=15)
    return DemoSet.generate(dataset, np.arange(30), rho=1.0)


def implementations(maze_demos) -> list[Demonstrations]:
    return [maze_demos]


def test_the_protocol_lists_the_surface_the_pipeline_reads():
    for name in ("observations", "routes", "replay", "level", "n_actions", "move_actions", "distances", "meta"):
        assert name in PROTOCOL_ATTRIBUTES


def test_maze_demonstrations_carry_every_protocol_attribute(maze_demos):
    for demos in implementations(maze_demos):
        missing = [name for name in PROTOCOL_ATTRIBUTES if not hasattr(demos, name)]
        assert not missing, f"{type(demos).__name__} lacks {missing}"
        assert isinstance(demos, Demonstrations)


def test_maze_move_actions_are_the_identity(maze_demos):
    assert maze_demos.n_actions == len(MOVES)
    assert maze_demos.move_actions == tuple(range(len(MOVES)))


def test_replay_on_the_set_matches_the_free_function(maze_demos):
    from goalmisgen.offline.demos import replay

    for index in range(5):
        route = maze_demos.routes([index])[0]
        via_set = maze_demos.replay(index, route)
        direct = replay(maze_demos.level(index), route, maze_demos.meta["step_penalty"], maze_demos.meta["step_limit"])
        assert via_set.keys() == direct.keys()
        assert via_set["reached_index"] == direct["reached_index"] == maze_demos.target[index]


def test_counterfactual_views_keep_shape_and_do_not_alias(maze_demos):
    swapped = maze_demos.with_values(np.asarray(maze_demos.values)[:, ::-1])
    assert swapped.values.shape == maze_demos.values.shape
    assert np.array_equal(swapped.values[:, 0], maze_demos.values[:, 1])
    with pytest.raises(ValueError, match="shaped"):
        maze_demos.with_feature_ids(np.zeros(3, dtype=np.int8))


def test_a_maze_set_loads_as_a_maze_set(maze_demos, tmp_path):
    maze_demos.save(tmp_path / "demos")
    assert task_name(tmp_path / "demos") is None
    loaded = load_demonstrations(tmp_path / "demos", hide_values=True)
    assert isinstance(loaded, DemoSet)
    assert loaded.hide_values
    assert len(loaded) == len(maze_demos)
