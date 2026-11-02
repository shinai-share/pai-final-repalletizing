"""What the camera cannot see: how full each case is and where its contents sit.

Drawn once per load (seed) for the physics only. The cognitive core never reads these values;
it estimates mass and centre of mass with a test lift."""
from __future__ import annotations

import dataclasses

import numpy as np

from .config import Variation


def vary_load(spawns, variation: Variation, seed: int) -> list:
    rng = np.random.default_rng(10_000 + seed)
    out = []
    for s in spawns:
        com = (float(rng.uniform(-variation.com_xy, variation.com_xy)),
               float(rng.uniform(-variation.com_xy, variation.com_xy)),
               float(rng.uniform(-variation.com_z, variation.com_z)))
        mass = float(rng.uniform(*variation.mass))
        out.append(dataclasses.replace(s, com=com, mass=mass))
    return out
