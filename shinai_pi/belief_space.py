"""The region of space the robot must account for around the source pallet: the pallet, the
height of a full load, and a margin of floor where cases can spill."""
from __future__ import annotations

from .config import Cell

SPILL_MARGIN = 0.9


def pile_region(cell: Cell):
    p = cell.pallet
    h = p.size[2] + cell.layout.layers * max(s.size[2] for s in cell.skus) + 0.25
    lo = (p.a_center[0] - p.size[0] / 2 - SPILL_MARGIN, p.a_center[1] - p.size[1] / 2 - SPILL_MARGIN, 0.0)
    hi = (min(p.a_center[0] + p.size[0] / 2 + SPILL_MARGIN, p.b_center[0] - p.size[0] / 2), p.a_center[1] + p.size[1] / 2 + SPILL_MARGIN, h)
    return lo, hi
