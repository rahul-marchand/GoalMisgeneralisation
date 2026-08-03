# Goal Misgeneralisation in DRC Maze Agents

4YP research code. Builds a multi-objective maze environment and trains DRC
(Deep Repeated ConvLSTM) agents on it, in order to study goal
misgeneralisation mechanistically — in particular, whether an agent's internal
plan representations predict goal misgeneralisation before it shows up in
behaviour.

## Structure

| Path | Contents |
|---|---|
| `goalmisgen/` | The maze environment, training configs, and analysis code |
| `experiments/` | Runnable experiment scripts |
| `tests/` | Unit and integration tests |
| `third_party/train-learned-planner` | far.ai's JAX/cleanba DRC training stack, as a pinned submodule |

The training stack is used **unmodified**. Our environment plugs into it by
subclassing `cleanba.environments.EnvConfig`, so upstream fixes can be pulled
in with a submodule update.

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```sh
git clone --recurse-submodules https://github.com/rahul-marchand/GoalMisgeneralisation
cd GoalMisgeneralisation
uv sync
```

On a GPU machine, add the CUDA build of JAX:

```sh
uv sync --extra gpu
```

Note that `third_party/train-learned-planner` has its own `envpool` submodule,
which is a large C++ build used only for fast Sokoban. We do not need it — it is
imported lazily — so it is left uninitialised. `gym-sokoban` *is* required, as
`cleanba.environments` imports it at module scope.

## Tests

```sh
uv run pytest
```
