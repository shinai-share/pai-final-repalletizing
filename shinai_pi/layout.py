"""Palletising pattern: columns x rows per layer, alternate layers turned 90 deg (interlock)."""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .config import Cell, Sku


@dataclass(frozen=True)
class Slot:
    index: int
    layer: int
    center: tuple
    yaw_deg: float
    footprint: tuple       # extent along world x, y


def pallet_slots(cell: Cell, sku: Sku, pallet_center) -> list:
    lay, p = cell.layout, cell.pallet
    sx, sy, sz = sku.size
    out = []
    for layer in range(lay.layers):
        turned = lay.interlock and layer % 2 == 1
        nx, ny = (lay.rows, lay.cols) if turned else (lay.cols, lay.rows)
        fx, fy = (sy, sx) if turned else (sx, sy)
        if nx * fx > p.size[0] + 1e-9 or ny * fy > p.size[1] + 1e-9:
            raise ValueError(f'sku {sku.id} does not fit the pattern on layer {layer}')
        # Each case sits centred in an equal share of the pallet, so neighbours keep a small gap
        # and a few millimetres of placing error do not land a case on its neighbour's edge.
        px, py = p.size[0] / nx, p.size[1] / ny
        x0 = pallet_center[0] - p.size[0] / 2 + px / 2
        y0 = pallet_center[1] - p.size[1] / 2 + py / 2
        z = p.size[2] + sz * (layer + 0.5)
        for j in range(ny):
            for i in range(nx):
                out.append(Slot(len(out), layer, (x0 + i * px, y0 + j * py, z), 90.0 if turned else 0.0, (fx, fy)))
    return out


def open_long_side(slots, index: int, reach: float = 0.30):
    """World direction (x, y) of a long side of slot `index` with no earlier slot of its layer next to
    it: the side the tines are pulled out on (requirements v05 5.3). None if both are taken."""
    s = slots[index]
    fx, fy = s.footprint
    long_along_x = fx >= fy
    normals = ((0.0, 1.0), (0.0, -1.0)) if long_along_x else ((1.0, 0.0), (-1.0, 0.0))
    half_long = max(fx, fy) / 2
    half_across = min(fx, fy) / 2
    c = np.asarray(s.center[:2], float)
    for n in normals:
        n = np.asarray(n)
        t = np.array([-n[1], n[0]])
        taken = False
        for e in slots[:index]:
            if e.layer != s.layer:
                continue
            d = np.asarray(e.center[:2], float) - c
            ex, ey = e.footprint
            e_half_along_n = abs(ex * n[0]) / 2 + abs(ey * n[1]) / 2
            e_half_across = abs(ex * t[0]) / 2 + abs(ey * t[1]) / 2
            if 0 < d @ n < half_across + reach + e_half_along_n and abs(d @ t) < half_long + e_half_across - 0.01:
                taken = True
                break
        if not taken:
            return n
    return None


@dataclass(frozen=True)
class CaseSpawn:
    case_id: str
    sku_id: str
    pos: tuple
    yaw_deg: float
    layer: int
    com: tuple = (0.0, 0.0, 0.0)      # centre of mass in the case's own frame (m), physics only
    mass: float | None = None         # kg; None means the SKU's nominal mass
    quat: tuple | None = None         # (w, x, y, z) full orientation for staged scenes; None means yaw_deg only


def initial_cases(cell: Cell, sku: Sku) -> list:
    """Full, intact unit load on pallet A before transport."""
    return [CaseSpawn(f'case_{s.index:02d}', sku.id, s.center, s.yaw_deg, s.layer)
            for s in pallet_slots(cell, sku, cell.pallet.a_center)]


def with_grid(cell: Cell, cols: int, rows: int, layers: int) -> Cell:
    """The cell loaded with a smaller block of cases (cols x rows x layers), layers not turned, so each upper case
    stands on the one below (a short demonstration load)."""
    if min(cols, rows, layers) < 1:
        raise ValueError('a load needs at least one case along each side')
    return replace(cell, layout=replace(cell.layout, cols=cols, rows=rows, layers=layers, interlock=False))
