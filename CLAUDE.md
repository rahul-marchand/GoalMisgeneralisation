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

## Known upstream limitations

### Evaluation assumes Sokoban

`cleanba.evaluate.get_cycles` asserts 3-channel square RGB observations, and its
caller assumes a positive reward means a box was pushed onto a target. Neither
holds here: our observations are symbolic multi-channel, and *every* objective
gives positive reward. Training runs fine and then dies at the first
evaluation — and since the exception is raised on the evaluation thread, the
process **hangs rather than exiting**, so it reads as a stall while still
billing for the GPU.

`goalmisgen/configs/compat.py` adapts from our side rather than editing
`third_party`. It is applied automatically on importing `goalmisgen.configs`.

The lesson that cost a run: **the evaluation path is not exercised by training
tests.** `test_training_survives_an_evaluation_pass` exists specifically to run
an evaluation, and profiling runs that set `eval_at_steps = frozenset()` do not
cover it.

### The DRC head is not size-agnostic

It is **not size-agnostic**: `cleanba/convlstm.py` flattens the
spatial dimensions before the MLP, and the actor/critic heads are `nn.Dense` on
that flattened vector, so input dimensions are baked into parameter shapes.
Mazes are therefore padded to a fixed maximum size. Out-of-distribution *size*
generalisation would need a custom pooled or fully-convolutional head, added as
a `PolicySpec` subclass in our code — not by editing theirs.

## Level datasets

Generating a level costs ~3.4 ms (maze construction plus a breadth-first search
per objective); stepping one costs ~40 us. A trained agent finishes in about 8
steps, so reset is ~90% of environment time — it is only ~40% under a random
policy that runs the full 120-step limit. Either way, real runs should use a
pre-generated dataset:

```sh
uv run python scripts/generate_levels.py --n-levels 1000000 --out data/levels
```

Roughly 20-40 minutes on eight cores (about two minutes on 48), ~36 MB. Set `MazeConfig.level_dataset` to
the directory and reset becomes an array lookup.

Three things about datasets are easy to get wrong:

- **One dataset serves an entire correlation sweep.** Layout, placement, values
  and distances do not depend on `feature_value_correlation`; only `feature_id`
  assignment does, and that happens at load time. Do not generate one dataset
  per rho.
- **Datasets are fingerprinted** over the sampler config *and* the source of
  every module in `dataset.CONTENT_MODULES`. Editing any of them invalidates
  existing datasets and loading one raises `FingerprintMismatch`. That is
  deliberate — regenerate rather than disabling the check. Comments and
  docstrings are stripped before hashing, so prose edits are free; anything else
  is not. The hash is computed **once per process**: the guard fires at startup,
  where a stale dataset is a real error, and never mid-run, where re-reading the
  files would only describe the working tree rather than the code that is
  running. Before that fix, a `git pull` on the training machine killed a
  150M-step run at 140M.
- **Training and evaluation must never share levels.** `maze_drc33` puts
  training on the `train` split and evaluation on `valid`; keep it that way or
  misgeneralisation becomes confounded with memorisation.

Datasets are stored as directories of plain `.npy` files, not compressed
archives, so they can be memory-mapped and shared across actor processes.
`LevelDataset` pickles by path for the same reason. Do not "simplify" either.

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
