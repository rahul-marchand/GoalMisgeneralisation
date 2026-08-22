"""Splitting a fitted axis by the module it lands in, and writing back a part of it.

``017`` asked where the DRC's value axis sits and answered by module, with two
cautions that carry over unchanged and are repeated here because this module
makes them easier to forget:

*Localisation is not storage.* Where a diff lands is where gradient descent found
it cheapest to write, which is not where a quantity is represented. Nothing here
licenses "the value lives in block 7".

*A fitted axis is mostly noise unless shown otherwise.* Every per-group number
this module produces is available with a split-half reliability beside it, and a
profile that does not replicate across disjoint halves of the arms is not a
finding.

What is new is the second half of the question. A share of ``||axis||^2`` says
how much of the *direction* sits in a module; it does not say whether writing
that much of it does anything. Those come apart exactly where the width/depth
campaign expects them to: a quantity recomputed at every layer can have its axis
spread thinly and still be uncontrollable one layer at a time, which is the
reported behaviour of steering vectors in language models and the thing the
campaign exists to reproduce somewhere the true answer is known. So the axis is
also *restrictable* — :func:`restrict` zeroes it outside a chosen set of groups
so the caller can write the part and measure the behaviour, rather than reading
a norm and inferring one.
"""

from __future__ import annotations

import numpy as np
from jax import tree_util

from goalmisgen.analysis.weights import _as_float, fit_axis_and_drift, split_half_reliability


def _component(entry) -> str:
    """One step of a pytree key path as a plain string, whatever flavour it is."""
    for attribute in ("key", "name"):
        value = getattr(entry, attribute, None)
        if isinstance(value, str):
            return value
    index = getattr(entry, "idx", None)
    return str(index) if index is not None else str(entry)


def group_of_path(path) -> str:
    """The module a parameter belongs to: the outermost name that is not ``params``.

    For :class:`~goalmisgen.offline.model.RoutePrefixLM` that yields ``block_0``
    ... ``block_{L-1}`` for the transformer stack and one group per top-level
    module beside it -- ``cell_in``, ``action_embedding``, ``ln_final``,
    ``head``. No name is special-cased, so a different architecture groups by
    whatever its own top level is called rather than by a pattern written for
    this one.
    """
    for entry in path:
        name = _component(entry)
        if name != "params":
            return name
    raise ValueError("this parameter has no name outside 'params' to group it by")


def parameter_groups(params, group_of=group_of_path) -> dict[str, np.ndarray]:
    """Indices into the flattened parameter vector, keyed by module.

    Ordered to match :func:`jax.flatten_util.ravel_pytree`, which concatenates
    ``leaf.ravel()`` in :func:`jax.tree_util.tree_flatten` order -- the same
    order :func:`goalmisgen.offline.axis.load_base` flattens a checkpoint in, so
    an index array here addresses the same parameter a diff does.
    """
    groups: dict[str, list[np.ndarray]] = {}
    start = 0
    for path, leaf in tree_util.tree_flatten_with_path(params)[0]:
        size = int(np.prod(np.shape(leaf)))
        groups.setdefault(group_of(path), []).append(np.arange(start, start + size))
        start += size
    return {name: np.concatenate(indices) for name, indices in groups.items()}


def group_spans(groups: dict[str, np.ndarray]) -> dict[str, slice]:
    """The same groups as slices, for callers that cannot afford a fancy-index copy.

    A 50M-parameter model with 24 arms is 9.6 GB of diffs in float64, and
    ``diffs[:, indices]`` allocates a second copy of whatever it selects, where
    ``diffs[:, a:b]`` is a view. Flax parameter dictionaries are sorted, so a
    module's leaves are adjacent and every group is in fact one span.

    **Raises rather than falling back** when a group is not contiguous. Silently
    returning to fancy indexing would turn a structural surprise into an
    out-of-memory hours into a run, which is the failure mode this project keeps
    paying for; a named error at the start is cheaper.
    """
    spans = {}
    for name, indices in groups.items():
        if len(indices) and indices[-1] - indices[0] + 1 != len(indices):
            raise ValueError(f"group {name!r} is not one contiguous span, so it has no slice")
        spans[name] = slice(int(indices[0]), int(indices[-1]) + 1) if len(indices) else slice(0, 0)
    return spans


def restrict(vector: np.ndarray, groups: dict[str, np.ndarray], keep) -> np.ndarray:
    """``vector`` with everything outside the named groups set to zero.

    Full length, so the result can be added straight onto a base's parameters and
    decoded. This is the object the behavioural half of the question needs:
    ``base + offset * restrict(axis, groups, ["block_3"])`` asks whether writing
    the value *there alone* moves the exchange rate.
    """
    vector = np.asarray(vector, dtype=np.float64)
    keep = [keep] if isinstance(keep, str) else list(keep)
    unknown = [name for name in keep if name not in groups]
    if unknown:
        raise ValueError(f"no such group(s): {', '.join(sorted(unknown))}")
    out = np.zeros_like(vector)
    for name in keep:
        out[groups[name]] = vector[groups[name]]
    return out


def group_shares(axis: np.ndarray, groups: dict[str, np.ndarray]) -> dict[str, float]:
    """Each group's fraction of ``||axis||^2``. Sums to one over all groups."""
    axis = np.asarray(axis, dtype=np.float64)
    total = float(axis @ axis)
    if total < 1e-30:
        raise ValueError("a zero axis has no length to divide between groups")
    return {name: float(axis[indices] @ axis[indices]) / total for name, indices in groups.items()}


def parameter_shares(groups: dict[str, np.ndarray], size: int) -> dict[str, float]:
    """Each group's fraction of the parameters -- the baseline a share is read against.

    A group holding 40% of the weights takes about 40% of a *random* direction,
    so an unnormalised share mostly reports how big a module is. Enrichment is
    the ratio of the two.
    """
    if size < 1:
        raise ValueError(f"a network with {size} parameters has no shares")
    return {name: len(indices) / size for name, indices in groups.items()}


def blocks_to_cover(shares: dict[str, float], fraction: float = 0.9) -> int:
    """How many groups, taken largest first, hold ``fraction`` of the axis.

    One scalar for "is this localised", comparable across depths in a way the
    largest group's share is not: at 16 blocks every share is smaller than at 4
    whether or not anything delocalised.
    """
    if not 0 < fraction <= 1:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    running = 0.0
    for count, value in enumerate(sorted(shares.values(), reverse=True), start=1):
        running += value
        if running >= fraction:
            return count
    return len(shares)


def axis_by_group(
    offsets: np.ndarray,
    diffs: np.ndarray,
    groups: dict[str, np.ndarray],
    splits: int = 200,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """Refit the axis inside each group alone, and say what each fit is worth.

    Per group: ``share`` of ``||axis||^2`` from the whole-network fit,
    ``parameter_share``, their ratio as ``enrichment``, and ``reliability`` --
    the split-half agreement of an axis fitted from that group's coordinates
    only. The last is the one that decides whether a group's profile is real,
    and it is computed from an independent fit rather than by slicing the global
    axis, so a group whose share is entirely noise reports a reliability near
    zero instead of inheriting the global one.
    """
    offsets = np.asarray(offsets, dtype=np.float64)
    diffs = _as_float(diffs)
    if offsets.ndim != 1 or diffs.ndim != 2 or len(offsets) != len(diffs):
        raise ValueError(f"need one offset per diff, got {offsets.shape} and {diffs.shape}")
    axis, _ = fit_axis_and_drift(offsets, diffs)
    shares = group_shares(axis, groups)
    sizes = parameter_shares(groups, diffs.shape[1])
    out = {}
    for name, indices in groups.items():
        restricted = diffs[:, indices]
        group_axis, _ = fit_axis_and_drift(offsets, restricted)
        out[name] = {
            "parameters": float(len(indices)),
            "share": shares[name],
            "parameter_share": sizes[name],
            "enrichment": shares[name] / sizes[name] if sizes[name] else float("nan"),
            "axis_norm": float(np.linalg.norm(group_axis)),
            "reliability": split_half_reliability(offsets, restricted, splits=splits, seed=seed),
        }
    return out
