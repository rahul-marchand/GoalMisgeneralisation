"""Compatibility shims for Sokoban-specific assumptions in the training stack.

The training stack is used unmodified, but parts of its *evaluation* path were
written for Sokoban specifically and do not hold for our environment. Where that
happens we adapt from our side rather than editing ``third_party``.

Currently one shim is needed.

``cleanba.evaluate.get_cycles`` computes the "agent paces in cycles" statistic
from the thinking-time work. It rests on two Sokoban assumptions:

1. Observations are 3-channel square RGB — it asserts
   ``all_obs.shape[1] == 3`` outright.
2. A positive reward means a box was pushed onto a target. The caller uses that
   to decide whether the analysis applies at all.

Neither holds here: our observations are symbolic multi-channel stacks, and
*every* objective yields a positive reward, so the analysis triggers on every
successful episode and then fails its own assertion. Training runs fine and then
dies at the first evaluation — and because the failure happens on the evaluation
thread, the main loop blocks rather than exiting, so a run appears to hang.

The shim makes ``get_cycles`` return no cycles for observations it was not
written for, leaving Sokoban behaviour untouched. The statistic itself is not
lost forever: a maze-appropriate version — detecting revisited *positions*
rather than repeated RGB frames — belongs in our analysis code when we come to
the thinking-time experiments.
"""

from __future__ import annotations

from typing import Any

import cleanba.evaluate as _evaluate

_PATCH_FLAG = "_goalmisgen_compat"


def _get_cycles_for_any_observation(all_obs: Any, last_box_time_step: int) -> list:
    """Delegate to upstream for Sokoban-shaped input; return no cycles otherwise."""
    shaped_like_sokoban = getattr(all_obs, "ndim", 0) == 4 and all_obs.shape[1] == 3 and all_obs.shape[2] == all_obs.shape[3]
    if not shaped_like_sokoban:
        return []
    return _original_get_cycles(all_obs, last_box_time_step=last_box_time_step)


_original_get_cycles = _evaluate.get_cycles


def install() -> None:
    """Idempotently apply the shims. Called on importing :mod:`goalmisgen.configs`."""
    if getattr(_evaluate.get_cycles, _PATCH_FLAG, False):
        return
    setattr(_get_cycles_for_any_observation, _PATCH_FLAG, True)
    _evaluate.get_cycles = _get_cycles_for_any_observation
