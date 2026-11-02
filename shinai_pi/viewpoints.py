"""Where the gantry can stand to look at a pile, and where it parks so fixed cameras see clearly.

The tool camera is mounted at the front edge of the suction plate and pitched forward, so it looks along the tool's
+y axis rotated by the yaw joint. A viewpoint is a gantry joint vector (x, y, z, yaw)."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import Cell

VIEW_TOOL_HEIGHT = 1.75        # tool face height (m) used for side views, or the top of the z range if lower (1.60 m
                               # from 2026-10-31: higher, the drawn-up fork bars meet the trolley)
VIEW_STANDOFF = 1.0            # horizontal distance from the pile centre to the tool


@dataclass(frozen=True)
class Viewpoint:
    name: str
    q: tuple                   # gantry joints
    looks_toward: tuple        # unit vector (x, y) the camera looks along


def look_yaw(direction_xy) -> float:
    """Yaw that turns the tool's +y axis onto direction_xy."""
    dx, dy = direction_xy
    return math.atan2(-dx, dy)


def side_viewpoints(cell: Cell, center_xy, standoff: float = VIEW_STANDOFF, height: float = VIEW_TOOL_HEIGHT):
    g = cell.gantry
    height = min(height, g.origin_height + g.z_range[1])
    out = []
    for name, (ux, uy) in (('south', (0, 1)), ('north', (0, -1)), ('west', (1, 0)), ('east', (-1, 0))):
        x = float(np.clip(center_xy[0] - ux * standoff, *g.x_range))
        y = float(np.clip(center_xy[1] - uy * standoff, *g.y_range))
        yaw = look_yaw((ux, uy))
        out.append(Viewpoint(name, (x, y, height - g.origin_height, yaw), (ux, uy)))
    return out


def park_pose(cell: Cell) -> np.ndarray:
    """Between the pallets, high: outside both overhead camera cones."""
    g = cell.gantry
    x = (cell.pallet.a_center[0] + cell.pallet.b_center[0]) / 2
    return np.array([x, 0.0, g.park_z, 0.0])
