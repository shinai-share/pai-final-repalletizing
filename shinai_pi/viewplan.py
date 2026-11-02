"""Where to look next, and whether looking is worth it.

The value of a viewpoint is the decision-relevant uncertainty it would resolve: unknown space and
poorly-known cases around the cases we would pick next. Points the camera cannot see (outside the
image, or behind an estimated case) do not count. Value is divided by the time it costs."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import Cell
from .motion import plan_move
from .perception.obb import yaw_matrix
from .pile import PileBelief
from .viewpoints import Viewpoint

CAPTURE_COST_S = 0.5
TOP_K = 3


@dataclass(frozen=True)
class ViewScore:
    viewpoint: Viewpoint
    info: float
    seconds: float

    @property
    def value(self) -> float:
        return self.info / self.seconds if self.seconds > 0 else self.info


def tool_camera_world(cell: Cell, q) -> tuple:
    """Camera position and camera-to-world rotation (columns = camera x, y, z axes) for joints q.
    Matches the USD camera: pitched about tool x, mounted at the plate's front edge."""
    tc = cell.tool_camera
    p = math.radians(tc.pitch_deg)
    Rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    Rt = yaw_matrix(q[3])
    origin = np.array([q[0], q[1], cell.gantry.origin_height + q[2]])
    return origin + Rt @ np.asarray(tc.offset, float), Rt @ Rx


def _intrinsics(cell: Cell):
    tc = cell.tool_camera
    w, h = tc.resolution
    f = tc.focal_mm / tc.aperture_mm * w
    return f, f, w, h


def visible(cell: Cell, q, points, belief: PileBelief) -> np.ndarray:
    """Which points lie in the tool camera's image and are not hidden behind an estimated case."""
    if len(points) == 0:
        return np.zeros(0, bool)
    pos, R = tool_camera_world(cell, q)
    fx, fy, w, h = _intrinsics(cell)
    local = (points - pos) @ R
    z = -local[:, 2]
    with np.errstate(divide='ignore', invalid='ignore'):
        u = local[:, 0] / z * fx + w / 2
        v = -local[:, 1] / z * fy + h / 2
    ok = (z > 0.05) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    if not np.any(ok):
        return ok
    d = points[ok] - pos
    dist = np.linalg.norm(d, axis=1)
    dirs = d / dist[:, None]
    blocked = np.zeros(len(dirs), bool)
    for k, b in belief.boxes.items():
        if k in belief.removed:
            continue
        o = b.obb
        t = o.ray_hit(np.broadcast_to(pos, dirs.shape), dirs)
        inside = o.contains(points[ok], margin=0.0)
        blocked |= (t < dist - 0.03) & ~inside
    for st in belief.static:
        t = st.ray_hit(np.broadcast_to(pos, dirs.shape), dirs)
        blocked |= t < dist - 0.03
    out = ok.copy()
    out[np.nonzero(ok)[0]] = ~blocked
    return out


def decision_targets(belief: PileBelief, k: int = TOP_K):
    """Points whose state matters for the next pick, with weights."""
    cands = [c for c in belief.candidates() if c.pickable][:k] or belief.candidates()[:k]
    pts, w = [], []
    unexplained = belief.unexplained_points()
    for c in cands:
        o = belief.boxes[c.box_id].obb
        reach = 0.5 * float(np.max(o.size))
        if len(unexplained):
            near = unexplained[o.signed_distance(unexplained) <= reach]
            pts.append(near)
            w.append(np.ones(len(near)))
        g = belief.graph()
        for other in g.supported_by(c.box_id) | {c.box_id}:
            b = belief.boxes[other]
            if b.confidence < 0.95:
                s = b.obb.surface_samples(6)
                pts.append(s)
                w.append(np.full(len(s), (1.0 - b.confidence) * 3.0 / len(s) * 20))
    if not pts:
        return np.empty((0, 3)), np.empty(0)
    return np.vstack(pts), np.concatenate(w)


def rank_viewpoints(cell: Cell, belief: PileBelief, viewpoints, q_now) -> list:
    pts, w = decision_targets(belief)
    out = []
    for vp in viewpoints:
        info = float(np.sum(w[visible(cell, vp.q, pts, belief)])) if len(pts) else 0.0
        # Viewpoints are gantry poses (x, y, z, yaw); the wrist does not move to look.
        secs = plan_move(np.asarray(q_now, float)[:4], np.asarray(vp.q, float)[:4], cell.gantry.v_max[:4],
                         cell.gantry.a_max[:4]).duration + CAPTURE_COST_S
        out.append(ViewScore(vp, info, secs))
    return sorted(out, key=lambda s: -s.value)
