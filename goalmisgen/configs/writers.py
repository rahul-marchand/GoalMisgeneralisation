"""Metrics writers for training runs.

cleanba logs through a ``WandbWriter``. On a machine without Weights & Biases
credentials — a fresh cloud pod, CI — we need the same interface backed by a
local file instead.

The interface is larger than it looks, and getting it wrong fails *late*:

===================  ======================================================
``add_scalar``       called constantly during training
``_save_dir``        used by the inherited ``save_dir`` context manager
``step_digits``      used to zero-pad checkpoint directory names
``named_save_dir``   listed by the checkpoint-resume path at startup
===================  ======================================================

Only ``add_scalar`` runs early. A writer missing ``step_digits`` trains happily
for hundreds of thousands of steps and then dies at the first checkpoint; one
missing ``named_save_dir`` dies at startup. Both have cost us a run, which is
why this lives in the package with a test that saves a checkpoint rather than
being redefined ad hoc per script.
"""

from __future__ import annotations

import math
import threading
from pathlib import Path
from typing import Any

import pandas as pd
from cleanba.cleanba_impala import WandbWriter


class CsvWriter(WandbWriter):
    """Accumulates metrics in a DataFrame and periodically writes them to CSV.

    Deliberately does not call ``super().__init__``, which would require a live
    Weights & Biases run.
    """

    def __init__(self, cfg: Any, save_dir: Path, flush_every: int = 25) -> None:
        self.metrics = pd.DataFrame()
        self.flush_every = flush_every
        self._writes = 0

        # The rollout and learner threads both log, and enlarging a DataFrame
        # through .loc is not atomic - concurrent bursts can leave the frame in
        # a state that raises on the next read. Rare per update, but a run makes
        # tens of thousands of them, and the failure surfaces on the rollout
        # thread, where an exception stalls the learner until its queue times
        # out five minutes later.
        self._lock = threading.Lock()

        save_dir = Path(save_dir)
        self._save_dir = save_dir / "local-files"
        self._save_dir.mkdir(parents=True, exist_ok=True)
        self.named_save_dir = save_dir
        self.step_digits = math.ceil(math.log10(max(10, cfg.total_timesteps)))

        self.csv_path = save_dir / "metrics.csv"

    def add_scalar(self, name: str, value: Any, global_step: int) -> None:
        try:
            values = list(value)
        except TypeError:
            values = None

        with self._lock:
            if values is None:
                self.metrics.loc[global_step, name] = value
            else:
                for offset, item in enumerate(values):
                    try:
                        self.metrics.loc[global_step + 640 * offset, name] = item.item()
                    except (TypeError, AttributeError, ValueError):
                        continue  # non-numeric metrics have no place in a CSV of scalars
            self._writes += 1
            due = self._writes % self.flush_every == 0

        # Outside the lock: flush() takes it again, and a plain Lock is not
        # reentrant.
        if due:
            self.flush()

    def flush(self) -> None:
        """Snapshot under the lock, write outside it.

        Holding the lock across the file write would block both logging threads
        for the duration of the write.
        """
        with self._lock:
            snapshot = self.metrics.sort_index().copy()
        snapshot.to_csv(self.csv_path)
