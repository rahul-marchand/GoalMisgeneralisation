"""Fitting the route model by next-token prediction on demonstrations.

Plain supervised learning: sample a batch of demonstrations, rebuild their
observations, compute the cross-entropy of the expert's moves, AdamW with
warmup and cosine decay. What is *not* plain is the checkpoint schedule. The
early-warning question needs the window in which the model is becoming
competent, and that window is early and short, so checkpoints are log-spaced:
dense at the start, sparse later, plus step 0 (the untrained network, which is
the baseline the probes are read against) and the final step.

Each checkpoint is a directory of one ``params.msgpack``; the run directory
holds ``config.json`` (model, training, and the demonstration set's header) so
a checkpoint can be loaded knowing nothing else, ``metrics.csv`` for the
training loss and ``eval.csv`` for whatever the evaluation callback returns.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import pathlib
import time
from typing import Callable, Iterable

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state

from goalmisgen.offline.demos import DemoSet
from goalmisgen.offline.model import (
    ModelConfig,
    RoutePrefixLM,
    cross_entropy,
    parameter_count,
    targets_from_routes,
    token_accuracy,
)


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    total_steps: int = 30_000
    batch_size: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    clip_norm: float = 1.0
    seed: int = 0
    log_every: int = 50
    checkpoint_first: int = 25
    checkpoint_ratio: float = 1.4
    schedule: str = "cosine"
    """``cosine`` (warmup then cosine decay to zero) or ``constant`` (warmup then flat).

    A fine-tune onto a shifted value uses ``constant``: the arms are compared by
    how far they moved under a fixed budget, so every step should be worth the
    same whichever arm it belongs to.
    """
    dtype: str = "float32"
    """``float32`` or ``bfloat16`` -- the precision the matmuls run in.

    Parameters, LayerNorms and the attention softmax stay float32 either way;
    see :class:`~goalmisgen.offline.model.RoutePrefixLM`. Recorded here rather
    than in ``ModelConfig`` because it is a property of how a run was trained,
    not of the network: a checkpoint written under one precision loads and
    decodes identically under the other.

    It moves the noise floor, so a grid comparing fine-tuning signal against
    fine-tuning noise has to fix it once for every cell.
    """

    init_from: str | None = None
    """A checkpoint directory to start from instead of a fresh initialisation.

    Set for the value-axis arms, which are the base model trained a little
    further on demonstrations at different values.

    **Parameters only. The optimiser starts fresh.** A checkpoint here holds
    params and nothing else, so Adam's moments and the step count do not carry
    across and the schedule restarts from zero. Worth stating outright: the DRC
    side of this project assumed the opposite of its own checkpoints, ran a 110M
    step extension believing it was a warm restart, and found afterwards that
    ``save_train_state`` had been carrying ``opt_state`` all along. The arms pass
    ``--schedule constant`` so the restart costs them no warmup, but anything
    that switches them to a decaying schedule is starting that decay again.
    """

    def __post_init__(self) -> None:
        if self.schedule not in ("cosine", "constant"):
            raise ValueError(f"schedule should be 'cosine' or 'constant', got {self.schedule!r}")
        if self.dtype not in ("float32", "bfloat16"):
            raise ValueError(f"dtype should be 'float32' or 'bfloat16', got {self.dtype!r}")

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TrainConfig":
        """Absent keys fall back to the field default.

        A config written before a field existed does not carry it, and every
        field added since has a default chosen to reproduce the behaviour of the
        runs that predate it. Insisting on the key instead means a new option
        makes every existing run on the volume unloadable, which is how the
        first attempt at adding ``dtype`` broke the analysis of `bcnv11`.
        """
        return cls(**{f.name: data[f.name] for f in dataclasses.fields(cls) if f.name in data})


def checkpoint_schedule(total_steps: int, first: int = 25, ratio: float = 1.4) -> tuple[int, ...]:
    """Step 0, then geometric from ``first`` by ``ratio``, then ``total_steps``.

    Geometric rather than linear because competence arrives on a log scale:
    the interesting part of a 30k-step run is its first two thousand steps,
    and a linear grid fine enough to resolve it would save hundreds of
    checkpoints later that all look the same.
    """
    if total_steps < 1:
        raise ValueError(f"total_steps must be positive, got {total_steps}")
    steps = {0, total_steps}
    step = float(first)
    while step < total_steps:
        steps.add(int(round(step)))
        step *= ratio
    return tuple(sorted(steps))


def checkpoint_dir(run_dir: pathlib.Path, step: int) -> pathlib.Path:
    return pathlib.Path(run_dir) / "checkpoints" / f"step_{step:08d}"


def save_checkpoint(run_dir: pathlib.Path, step: int, params) -> pathlib.Path:
    directory = checkpoint_dir(run_dir, step)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "params.msgpack").write_bytes(flax.serialization.to_bytes(params))
    return directory


def list_checkpoints(run_dir: pathlib.Path) -> list[tuple[int, pathlib.Path]]:
    """Every saved step, ascending."""
    root = pathlib.Path(run_dir) / "checkpoints"
    if not root.exists():
        return []
    found = []
    for directory in root.iterdir():
        if directory.is_dir() and directory.name.startswith("step_") and (directory / "params.msgpack").exists():
            found.append((int(directory.name.split("_")[1]), directory))
    return sorted(found)


def load_run_config(run_dir: pathlib.Path) -> dict:
    return json.loads((pathlib.Path(run_dir) / "config.json").read_text())


def load_checkpoint(directory: pathlib.Path) -> tuple[RoutePrefixLM, dict]:
    """The model and its parameters, from a checkpoint directory alone.

    The run's ``config.json`` is one level up, so a checkpoint path suffices.
    """
    directory = pathlib.Path(directory)
    config = load_run_config(directory.parent.parent)
    model = RoutePrefixLM(ModelConfig.from_dict(config["model"]))
    template = initial_params(model, jax.random.PRNGKey(0))
    params = flax.serialization.from_bytes(template, (directory / "params.msgpack").read_bytes())
    return model, params


def initial_params(model: RoutePrefixLM, key) -> dict:
    cfg = model.config
    observations = jnp.zeros((1, cfg.size, cfg.size, cfg.n_channels), dtype=jnp.float32)
    actions = jnp.full((1, cfg.max_actions), -1, dtype=jnp.int32)
    return model.init(key, observations, actions)


class _Csv:
    """Append rows to a CSV, writing the header from the first row's keys."""

    def __init__(self, path: pathlib.Path) -> None:
        self.path = pathlib.Path(path)
        self._columns: list[str] | None = None

    def write(self, row: dict) -> None:
        if self._columns is None:
            self._columns = list(row)
            with self.path.open("w", newline="") as handle:
                csv.DictWriter(handle, fieldnames=self._columns).writeheader()
        with self.path.open("a", newline="") as handle:
            csv.DictWriter(handle, fieldnames=self._columns, extrasaction="ignore").writerow(row)


def batches(demos: DemoSet, batch_size: int, rng: np.random.Generator) -> Iterable[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Shuffled epochs of ``(observations, actions, lengths)``, forever."""
    n = len(demos)
    while True:
        order = rng.permutation(n)
        for start in range(0, n - batch_size + 1, batch_size):
            index = np.sort(order[start : start + batch_size])  # sorted: memory-mapped reads stay sequential
            yield demos.observations(index), demos.routes(index), np.asarray(demos.lengths[index], dtype=np.int32)


Evaluator = Callable[[dict, int], dict[str, float]]
"""Called at every checkpoint with ``(params, step)``; returns a row for ``eval.csv``."""


def train(
    demos: DemoSet,
    model_config: ModelConfig,
    train_config: TrainConfig,
    run_dir: pathlib.Path,
    evaluate: Evaluator | None = None,
    log: Callable[[str], None] = print,
) -> dict:
    """Fit the model; returns the final parameters. Everything else goes to ``run_dir``."""
    run_dir = pathlib.Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if demos.max_actions > model_config.max_actions:
        raise ValueError(
            f"demonstrations hold up to {demos.max_actions} moves but the model emits at most "
            f"{model_config.max_actions}; a route that does not fit is silently truncated"
        )

    model = RoutePrefixLM(model_config, dtype=getattr(jnp, train_config.dtype))
    key = jax.random.PRNGKey(train_config.seed)
    params = initial_params(model, key)
    if train_config.init_from is not None:
        _, loaded = load_checkpoint(pathlib.Path(train_config.init_from))
        if jax.tree_util.tree_structure(loaded) != jax.tree_util.tree_structure(params) or any(
            a.shape != b.shape for a, b in zip(jax.tree_util.tree_leaves(loaded), jax.tree_util.tree_leaves(params))
        ):
            raise ValueError(f"{train_config.init_from} holds a different model shape from {model_config}")
        params = loaded
    if train_config.schedule == "cosine":
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=train_config.learning_rate,
            warmup_steps=train_config.warmup_steps,
            decay_steps=train_config.total_steps,
            end_value=0.0,
        )
    else:
        schedule = optax.join_schedules(
            [
                optax.linear_schedule(0.0, train_config.learning_rate, max(train_config.warmup_steps, 1)),
                optax.constant_schedule(train_config.learning_rate),
            ],
            [max(train_config.warmup_steps, 1)],
        )
    optimiser = optax.chain(
        optax.clip_by_global_norm(train_config.clip_norm),
        optax.adamw(schedule, weight_decay=train_config.weight_decay),
    )
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimiser)

    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "model": model_config.to_dict(),
                "train": train_config.to_dict(),
                "demos": {
                    **demos.meta,
                    "path": None if demos.path is None else str(demos.path),
                    "hide_values": bool(demos.hide_values),
                },
                "parameters": parameter_count(params),
                "checkpoints": checkpoint_schedule(
                    train_config.total_steps, train_config.checkpoint_first, train_config.checkpoint_ratio
                ),
            },
            indent=2,
        )
    )
    log(f"{parameter_count(params):,} parameters; {len(demos):,} demonstrations; {train_config.total_steps:,} steps")

    eos = model_config.eos

    @jax.jit
    def train_step(state, observations, actions, lengths):
        targets = targets_from_routes(actions, lengths, eos)

        def loss_fn(p):
            logits, _ = state.apply_fn(p, observations, actions)
            return cross_entropy(logits, targets), token_accuracy(logits, targets)

        (loss, accuracy), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        return state.apply_gradients(grads=grads), loss, accuracy

    metrics = _Csv(run_dir / "metrics.csv")
    evaluations = _Csv(run_dir / "eval.csv")
    checkpoints = set(
        checkpoint_schedule(train_config.total_steps, train_config.checkpoint_first, train_config.checkpoint_ratio)
    )

    stream = batches(demos, train_config.batch_size, np.random.default_rng(train_config.seed))
    started = time.perf_counter()
    recent_loss, recent_accuracy, recent_count = 0.0, 0.0, 0
    for step in range(train_config.total_steps + 1):
        if step in checkpoints:
            save_checkpoint(run_dir, step, state.params)
            if evaluate is not None:
                row = {"step": step, **evaluate(state.params, step)}
                evaluations.write(row)
        if step == train_config.total_steps:
            break

        observations, actions, lengths = next(stream)
        state, loss, accuracy = train_step(state, jnp.asarray(observations), jnp.asarray(actions), jnp.asarray(lengths))
        recent_loss += float(loss)
        recent_accuracy += float(accuracy)
        recent_count += 1
        if (step + 1) % train_config.log_every == 0:
            elapsed = time.perf_counter() - started
            row = {
                "step": step + 1,
                "loss": recent_loss / recent_count,
                "token_accuracy": recent_accuracy / recent_count,
                "learning_rate": float(schedule(step)),
                "elapsed_s": elapsed,
            }
            metrics.write(row)
            log(
                f"step {step + 1:>7,}  loss {row['loss']:.4f}  acc {row['token_accuracy']:.3f}  "
                f"lr {row['learning_rate']:.2e}  {(step + 1) / elapsed:.1f} steps/s"
            )
            recent_loss, recent_accuracy, recent_count = 0.0, 0.0, 0

    return state.params
