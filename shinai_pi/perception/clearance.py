"""Room left along the cups' approach, from the downward camera at the plate centre (requirements
v05 R4, 5.2). Whatever is not the case being reached for, inside the plate's footprint plus a
margin and below the cup face, is in the path; the nearest such point is how far the cups can
still go. The camera cannot see past a case held on the cups: it watches approaches, not carries."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FOOTPRINT_MARGIN = 0.02        # m around the plate
OWN_HARDWARE = 0.015           # m below the cup face: the cups themselves show there
MIN_POINTS = 3                 # fewer stray points are not an object
REGION_GAP = 0.005             # m: what a region's bottom passes this close over counts as struck


@dataclass(frozen=True)
class Clearance:
    blocked_at: float | None   # m along the approach to the nearest foreign point; None if the path is clear
    what: str | None           # which object it belongs to
    points: int                # foreign points in the path
    valid: bool                # the camera returned depth at all
    stamp: float


def path_clearance(frame, tool_pos, tool_R, plate_half, ignore=(), reach=np.inf, regions=(), direction=None) -> Clearance:
    """`tool_R` columns are the tool axes in the world. The tool travels along `direction` (world; by default
    -tool z, the cups' approach). Only what lies within `reach` of travel (the distance to the face being
    reached, and a little) can be in the path: what shows beyond the face, past a turned case's corners,
    cannot be struck. `regions` are other parts that travel with the plate, as boxes in the tool frame
    (x0, x1, y0, y1, bottom above the cup face): what one's bottom meets within the same travel is struck."""
    pts, inst, _ = frame.unproject(step=1)
    if len(pts) == 0:
        return Clearance(None, None, 0, False, frame.stamp)
    R = np.asarray(tool_R, float)
    local = (pts - np.asarray(tool_pos, float)) @ R
    d = -R[:, 2] if direction is None else np.asarray(direction, float) / np.linalg.norm(direction)
    v = -(R.T @ d)                     # how a still point moves in the tool frame per metre the tool travels
    if v[2] <= 1e-6:
        return Clearance(None, None, 0, True, frame.stamp)
    keys = np.array([frame.instance_map.get(int(i), '') for i in inst])
    hx, hy = plate_half[0], plate_half[1]
    boxes = [(-hx, hx, -hy, hy, 0.0, OWN_HARDWARE)] + [(*r, REGION_GAP) for r in regions]
    path = np.zeros(len(pts), bool)
    travel = np.full(len(pts), np.inf)
    for x0, x1, y0, y1, bottom, gap in boxes:
        s_hit = (bottom - local[:, 2]) / v[2]            # travel until the part's bottom reaches the point's height
        x = local[:, 0] + s_hit * v[0]
        y = local[:, 1] + s_hit * v[1]
        hit = ((x >= x0 - FOOTPRINT_MARGIN) & (x <= x1 + FOOTPRINT_MARGIN) & (y >= y0 - FOOTPRINT_MARGIN)
               & (y <= y1 + FOOTPRINT_MARGIN) & (s_hit > gap) & (s_hit <= reach))
        travel = np.where(hit, np.minimum(travel, s_hit), travel)
        path |= hit
    foreign = path & ~np.isin(keys, list(ignore))
    if foreign.sum() < MIN_POINTS:
        return Clearance(None, None, int(foreign.sum()), True, frame.stamp)
    depth = travel[foreign]
    i = int(np.argmin(depth))
    return Clearance(float(depth[i]), str(keys[foreign][i]), int(foreign.sum()), True, frame.stamp)
