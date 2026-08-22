"""Writing a route into the route model's residual stream.

The offline twin of the machinery ``012_rewrite_the_plan.py`` uses on the DRC's
cell state. The concept written is identical - five classes per cell, four moves
or ``NEVER``, from :mod:`goalmisgen.analysis.plans` - and so is the arithmetic
that turns a fitted probe into a direction, :func:`~goalmisgen.analysis.probes.
class_directions`. What differs is the site, and two properties of the site
decide how the experiment has to be read.

**A write persists for free.** The DRC's edit had to be re-applied before every
forward pass, because ``c`` is rebuilt from the gates each tick. Here the prefix
is recomputed identically at every decoded token and can never see an action
token, so one edit inside the forward pass is present for the whole route.

**A write has a depth budget, and it is short.** The maze-token residual reaches
the action logits only through attention in *later* blocks. An edit after the
last block cannot change a logit at all - the head reads from SEP onward - so
the writable depths are ``0 .. n_layers - 1`` while the probe reads best at the
top. On a four-block model the usable band is blocks 2 and 3, which leaves one
or two attention layers to consume the edit against the DRC's three ticks in
every remaining step. A small effect at depth 3 is therefore ambiguous between
"the plan is not used" and "the edit had one block to be read in", which is why
:func:`propagation` exists and why depth ``n_layers`` is run as a control whose
null is guaranteed in advance.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from goalmisgen.analysis import geometry, plans
from goalmisgen.analysis.probes import apply_multinomial, fit_multinomial
from goalmisgen.offline.demos import DemoSet

N_FEATURES = 2


# ----------------------------------------------------------------------
# The probe the write comes from
# ----------------------------------------------------------------------


def plan_rows(rollouts) -> tuple[np.ndarray, np.ndarray]:
    """``(cells, d_model)`` features and their class label, over scoreable cells.

    The label is the route the model's own greedy decode walked, which is what
    makes the probe a readout of this network's plan rather than of the maze:
    ``UNSCOREABLE`` cells - walls, and the last cell of a route, which has no
    next move - are dropped rather than called ``NEVER``.
    """
    features, labels = [], []
    for rollout in rollouts:
        grid = plans.observed_directions(rollout)
        keep = grid >= 0
        features.append(rollout.features[keep])
        labels.append(grid[keep])
    return np.concatenate(features).astype(np.float64), np.concatenate(labels)


def fit_plan_probe(rollouts, n_classes: int = plans.N_CLASSES):
    """Softmax readout of the per-cell direction. Returns ``(w, mean, std)``."""
    x, y = plan_rows(rollouts)
    return fit_multinomial(x, y, n_classes)


def probe_accuracy(rollouts, weights, mean, std) -> tuple[float, float, float]:
    """Overall accuracy, accuracy on route cells, and the majority-class floor.

    Route-cell accuracy is over the four directions only, where the floor is
    25% rather than the ``NEVER`` rate - the number ``012`` reports, for the
    same reason: a probe that predicts ``NEVER`` everywhere scores well on the
    first column and knows nothing about a route.
    """
    x, y = plan_rows(rollouts)
    predicted = apply_multinomial(x, weights, mean, std).argmax(1)
    on_route = y < plans.NEVER
    majority = max(float((y == k).mean()) for k in range(plans.N_CLASSES))
    return float((predicted == y).mean()), float((predicted[on_route] == y[on_route]).mean()), majority


def typical_cell_norm(rollouts) -> float:
    """Mean residual norm at a free cell, the unit the write's alpha is in.

    Writing in absolute units would make an alpha meaningless across depths: a
    pre-LN stream grows with depth, so the same vector is a shove at the
    embedding and a nudge at block 4.
    """
    norms = [
        np.linalg.norm(rollout.features[geometry.free_cells(rollout.observation)], axis=-1).mean()
        for rollout in rollouts
    ]
    return float(np.mean(norms))


# ----------------------------------------------------------------------
# What to write, and where
# ----------------------------------------------------------------------


def plan_edit(
    observation: np.ndarray,
    target: int,
    replaced: int,
    write_route: bool = True,
    erase_old: bool = True,
) -> dict[tuple[int, int], int] | None:
    """Which class to write at which cell: the new route, and ``NEVER`` on the old.

    ``target`` and ``replaced`` are *feature ids*, as
    :func:`~goalmisgen.analysis.plans.planned_directions` takes them. ``None``
    if the target cannot be reached - the other objective blocks the only
    corridor, which happens often enough on 11x11 levels to matter.

    A copy of ``012.plan_edit`` rather than a shared import, because the DRC's
    version lives in an experiment script and experiments are not importable.
    The two must agree; ``tests/test_offline_rewrite.py`` checks the properties
    that matter rather than the text.

    Only cells that carry information are touched - the route to write and the
    route to erase. Writing ``NEVER`` at every off-route cell instead would
    perturb most of the maze, which measures how much damage the model tolerates
    rather than whether it follows a plan.
    """
    edit: dict[tuple[int, int], int] = {}

    if write_route:
        wanted = plans.planned_directions(observation, target, N_FEATURES)
        if wanted is None:
            return None
        for row, col in np.argwhere((wanted >= 0) & (wanted < plans.NEVER)):
            edit[(int(row), int(col))] = int(wanted[row, col])

    if erase_old:
        old = plans.planned_directions(observation, replaced, N_FEATURES)
        if old is None:
            return None if not edit else edit
        for row, col in np.argwhere((old >= 0) & (old < plans.NEVER)):
            # A cell on both routes keeps its new direction: the model walks it
            # either way, so erasing it would contradict the plan being written
            # rather than the one being replaced.
            edit.setdefault((int(row), int(col)), plans.NEVER)
    return edit or None


def erase_only(
    observation: np.ndarray,
    route_to: int,
    keep: str = "all",
    seed: int = 0,
) -> dict[tuple[int, int], int] | None:
    """``NEVER`` on the cells of one route, or on a decoy set the same size.

    ``030`` found the whole of its effect in the erasing half of the edit, which
    leaves three deflationary readings that the arms it had cannot separate.
    ``keep`` picks between them:

    ``all``     every cell of the route to ``route_to`` - the original arm.
    ``start``   only the cell the agent is standing on. The erase set contains
                it, so "we corrupted the position token" predicts this alone
                reproduces the effect.
    ``tail``    every route cell *except* the agent's. The complement of
                ``start``: between them they partition the original edit, so
                whichever carries it is where the effect lives.
    ``sham``    the same number of free cells drawn at random from *off* both
                routes. Tests "writing NEVER at plausible cells breaks it"
                against "writing NEVER at the cells of the route it is walking
                tells it something". The shuffled arm is a weaker version of
                this - it lands a different maze's route here, so its cells are
                arbitrary rather than merely off-route.
    """
    field = plans.planned_directions(observation, route_to, N_FEATURES)
    if field is None:
        return None
    cells = [(int(r), int(c)) for r, c in np.argwhere((field >= 0) & (field < plans.NEVER))]
    if not cells:
        return None
    start = geometry.agent_cell(observation)

    if keep == "start":
        chosen = [start] if start in cells else []
    elif keep == "tail":
        chosen = [cell for cell in cells if cell != start]
    elif keep == "sham":
        other = plans.planned_directions(observation, 1 - route_to, N_FEATURES)
        on_route = set(cells) | {start}
        if other is not None:
            on_route |= {(int(r), int(c)) for r, c in np.argwhere((other >= 0) & (other < plans.NEVER))}
        free = [
            (int(r), int(c))
            for r, c in np.argwhere(geometry.free_cells(observation))
            if (int(r), int(c)) not in on_route
        ]
        if len(free) < len(cells):
            return None
        picked = np.random.default_rng(seed).choice(len(free), size=len(cells), replace=False)
        chosen = [free[index] for index in picked]
    elif keep == "all":
        chosen = cells
    else:
        raise ValueError(f"unknown erase subset {keep!r}")

    return {cell: plans.NEVER for cell in chosen} or None


def derange(count: int, seed: int) -> np.ndarray:
    """A permutation with no fixed point, so no plan lands on its own maze."""
    order = np.random.default_rng(seed).permutation(count)
    for position, destination in enumerate(order):
        if position == destination:
            swap = (position + 1) % count
            order[position], order[swap] = order[swap], order[position]
    return order


def delta_grid(
    edits: list[dict[tuple[int, int], int] | None],
    directions: np.ndarray,
    size: int,
    magnitude: float,
) -> np.ndarray:
    """``(B, n_cells, d_model)`` to add to the stream at the maze tokens.

    Raster order, matching the model's own ``observations.reshape(batch,
    n_cells, n_channels)``: cell ``(r, c)`` is position ``r * size + c``. An
    off-by-one here writes a coherent plan onto the wrong maze and produces a
    plausible null, so :func:`~goalmisgen.offline.rewrite.written_classes`
    reads it back rather than trusting it.
    """
    grid = np.zeros((len(edits), size * size, directions.shape[1]), dtype=np.float32)
    for index, edit in enumerate(edits):
        if edit is None:
            continue
        for (row, col), label in edit.items():
            grid[index, row * size + col] = magnitude * directions[label]
    return grid


def written_classes(
    rollouts,
    edits: list[dict[tuple[int, int], int] | None],
    weights,
    mean,
    std,
    directions: np.ndarray,
    magnitude: float,
) -> float:
    """Fraction of edited cells where the probe reads back the class written.

    :func:`~goalmisgen.analysis.probes.class_write_accuracy` asks this of a
    class at a time on pooled cells; this asks it of the actual edit, cell for
    cell, so an error in the raster ordering or in the level-to-row alignment
    shows up as a low number rather than as a clean behavioural null.
    """
    hits, total = 0, 0
    for rollout, edit in zip(rollouts, edits):
        if not edit:
            continue
        cells = list(edit)
        features = np.stack([rollout.features[cell] for cell in cells]).astype(np.float64)
        wanted = np.array([edit[cell] for cell in cells])
        written = features + magnitude * directions[wanted]
        hits += int((apply_multinomial(written, weights, mean, std).argmax(1) == wanted).sum())
        total += len(cells)
    return hits / max(total, 1)


def propagation(
    residuals_before: np.ndarray,
    residuals_after: np.ndarray,
    edits: list[dict[tuple[int, int], int] | None],
    weights,
    mean,
    std,
) -> tuple[float, float]:
    """Does a probe at a *later* depth read the written class, or the old one?

    ``residuals_*`` are that later depth's per-cell grids, ``(B, size, size,
    d_model)``, without and with the edit applied further down; the probe is
    the one fitted at that later depth. Returns the fraction of edited cells
    reading the written class before and after.

    This is the check that makes a behavioural null interpretable. If an edit
    at block 2 does not survive to block 4, the experiment measured the depth
    budget rather than whether the plan is used.
    """
    def rate(grids: np.ndarray) -> float:
        hits, total = 0, 0
        for grid, edit in zip(grids, edits):
            if not edit:
                continue
            cells = list(edit)
            features = np.stack([grid[cell] for cell in cells]).astype(np.float64)
            wanted = np.array([edit[cell] for cell in cells])
            hits += int((apply_multinomial(features, weights, mean, std).argmax(1) == wanted).sum())
            total += len(cells)
        return hits / max(total, 1)

    return rate(residuals_before), rate(residuals_after)


# ----------------------------------------------------------------------
# The counterfactual, which the DRC cannot do cheaply and this can
# ----------------------------------------------------------------------


def swapped_values(demos: DemoSet) -> DemoSet:
    """The same levels with the objectives' values exchanged.

    Same maze, same start, same objective cells, same colours - only which
    objective is worth more changes. The demonstrations it carries are now the
    *wrong* expert for the observation, which does not matter: nothing here
    trains, and every route is replayed against the level it was decoded from.

    Patching this level's residual into the real one, rather than adding a
    fitted direction, is a ceiling on what any per-cell write at that depth can
    do. It is available because the prefix is a pure function of the maze; the
    DRC's equivalent would need a second rollout interleaved step by step.
    """
    if demos.values.shape[1] != 2:
        raise ValueError(f"swapping values is defined for two objectives, not {demos.values.shape[1]}")
    return dataclasses.replace(demos, values=np.asarray(demos.values)[:, ::-1].copy())


def swapped_features(demos: DemoSet) -> DemoSet:
    """The same levels with the objectives' *colours* exchanged.

    The counterfactual for a model trained without the value channel. There,
    swapping values is a no-op the model cannot see - the values are learned
    constants, which is the whole point of ``bcnv11`` - so the only way to tell
    it that the other objective is the valuable one is to move the colour.

    Which also makes the two counterfactuals ask different questions, and the
    difference is worth stating. Swapping values on a model that reads them asks
    "if this objective were worth more, would the plan change". Swapping colours
    asks "if the cue said so", which on a proxy-trained model is the
    misgeneralisation itself rather than a control for it.
    """
    return dataclasses.replace(demos, feature_ids=np.asarray(demos.feature_ids)[:, ::-1].copy())


def patch_edit(
    baseline: np.ndarray,
    counterfactual: np.ndarray,
    cells: list[np.ndarray] | None = None,
) -> np.ndarray:
    """``(B, n_cells, d_model)`` turning the real residual into the other one.

    A difference, because the hook adds: adding ``cf - real`` at a cell replaces
    that cell. ``cells`` restricts the patch to a boolean ``(size, size)`` mask
    per level - patching only the two objective cells asks where the value
    evidence enters, which a whole-grid patch cannot, since patching everything
    is barely distinguishable from feeding the counterfactual observation.
    """
    if baseline.shape != counterfactual.shape:
        raise ValueError(f"residuals differ in shape: {baseline.shape} against {counterfactual.shape}")
    batch, size = baseline.shape[0], baseline.shape[1]
    delta = (counterfactual - baseline).reshape(batch, size * size, baseline.shape[-1])
    if cells is None:
        return delta
    mask = np.stack([cell.reshape(-1) for cell in cells])[..., None]
    return delta * mask


def objective_cells(observation: np.ndarray) -> np.ndarray:
    """``(size, size)`` bool marking the cells the two objectives sit on."""
    mask = np.zeros(observation.shape[:2], dtype=bool)
    for feature in range(N_FEATURES):
        mask[geometry.objective_cell(observation, feature)] = True
    return mask
