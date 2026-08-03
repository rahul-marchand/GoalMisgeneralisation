# Goal Misgeneralisation 4YP

Research code studying goal misgeneralisation in DRC (Deep Repeated ConvLSTM)
agents on multi-objective mazes. The guiding hypothesis is that an agent's
internal plan representations predict goal misgeneralisation *before* it
appears in behaviour.

## Environment and tooling

Use `uv` for everything. There is no `pip install` step and no Docker.

```sh
uv sync                 # install (CPU)
uv sync --extra gpu     # install with CUDA JAX, on GPU machines
uv run pytest           # tests
uv run python ...       # anything else
```

Python is pinned to 3.11 — far.ai's stack pins `jax==0.4.34`, `gymnasium~=0.29`
and `numpy~=1.26`, which do not all have wheels on newer interpreters.

## The training stack is a dependency, not a fork

`third_party/train-learned-planner` is a **git submodule pinned to a specific
commit**. Only the pointer is tracked in this repo.

**Never edit anything under `third_party/`.** Our code plugs in by subclassing
`cleanba.environments.EnvConfig`; `tests/test_cartpole.py` upstream proves that
custom environments and training work with zero source modification. If
something genuinely cannot be done without changing their code, raise it rather
than patching in place — it usually means we should subclass instead.

Two non-obvious facts about that stack:

- `gym-sokoban` is **required** even though we never use Sokoban, because
  `cleanba/environments.py` imports it at module scope. It is installed from the
  nested submodule.
- `envpool` is **not** required and its nested submodule is deliberately left
  uninitialised. It is a heavy C++ build used only for fast Sokoban and is
  imported lazily inside `EnvpoolEnvConfig.make`.

## Known upstream limitation

The DRC head is **not size-agnostic**: `cleanba/convlstm.py` flattens the
spatial dimensions before the MLP, and the actor/critic heads are `nn.Dense` on
that flattened vector, so input dimensions are baked into parameter shapes.
Mazes are therefore padded to a fixed maximum size. Out-of-distribution *size*
generalisation would need a custom pooled or fully-convolutional head, added as
a `PolicySpec` subclass in our code — not by editing theirs.

## Design conventions

The environment gets extended repeatedly over the project (new correlation
structures, objective counts, distribution shifts). The organising rule is that
**`MazeEnv` does not change** — it consumes a `Level` and steps it. Anything
experiment-specific goes behind a protocol in a swappable strategy object.

- `Level` fully specifies one episode; `LevelSampler` produces them. A new
  experiment should be a new sampler, not an edit to an existing one.
- **Ground truth (BFS distances, optimal target) belongs in the `info` dict,
  never in the observation.** There is a test asserting this.
- Prefer small, typed, single-purpose modules. Favour clarity over cleverness
  and over premature optimisation; profile before optimising the env.
- Every layer gets tests before the next layer is built on it.

## Commit conventions

Commit to `main`. Keep commits small and reviewable, one logical change each.
**Do not add `Co-Authored-By` trailers or any other attribution to commit
messages.**
