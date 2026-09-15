"""The maze experiment inside the Craftax-Classic engine.

Craftax is used unmodified, as an engine: we build its states and step them.
What this package adds is everything the experiment needs that the engine does
not have - a controlled world (``world``), an expert that plans on it
(``planner``), a reduced full-map observation in the maze's channel layout
(``observation``) and the bridge that turns a world into an engine state and
walks a route through it (``engine``).

Only ``engine`` imports Craftax, and therefore JAX. The rest is numpy, for the
same reason ``envs.dataset`` and ``offline.demos`` are: generation runs in a
spawned worker pool, and a JAX-free payload is what keeps the spawn cheap.
"""
