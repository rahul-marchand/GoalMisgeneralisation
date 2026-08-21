"""Worker pools that do not inherit a forked JAX runtime.

Level and demonstration generation are both CPU-bound, embarrassingly parallel,
and JAX-free: ``envs.dataset`` and ``offline.demos`` import numpy and nothing
else heavy. Their *entry points* are not. ``scripts/generate_levels.py`` and
``scripts/generate_demos.py`` both import ``configs.env``, which reaches cleanba
and therefore JAX, so by the time either reaches its pool the parent process has
a multithreaded JAX runtime in it.

Forking that is the documented deadlock -- ``os.fork()`` copies one thread and
leaves every lock the others held closed forever. Python warns about it, and
``configs/presets.py`` already records the same hazard on the evaluation path.
It has not bitten yet, which is the worst way for a hazard like this to sit:
when it does, generation *hangs* rather than failing, so it reads as a slow job
while still billing for the machine, exactly as the evaluation stall did.

``spawn`` starts each worker from a fresh interpreter, so there is no inherited
lock state to deadlock on. The cost is that the child re-imports the parent's
``__main__``, and both entry points guard theirs with ``if __name__ ==
"__main__"``, so what it re-imports is the module body and not another run.
"""

from __future__ import annotations

import multiprocessing as mp
from multiprocessing.pool import Pool


def worker_pool(workers: int) -> Pool:
    """A pool of ``workers`` processes, started clean rather than forked.

    Used as a context manager, like ``mp.Pool``. Everything passed to it has to
    pickle, which was already true under fork-with-arguments.
    """
    return mp.get_context("spawn").Pool(workers)
