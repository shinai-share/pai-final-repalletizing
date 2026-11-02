"""Measure a placed case from its own top face.

A box fitted to all of a case's visible surface is pulled by the neighbours packed against it
(their faces hide its sides, and the fit fills the gap). The top face of a case just placed is
not hidden: its centre, height, long-edge direction and tilt come from that face alone, with the
same extent method the floor camera uses for the bottom face."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

TOP_BAND = 0.03          # m below the highest point that still belongs to the top face (a 3 degree lean across 0.36 m is 19 mm)
MIN_POINTS = 60


@dataclass(frozen=True)
class TopFace:
    centre: np.ndarray           # (x, y) of the face centre, m
    top_z: float                 # m
    long_edge_yaw_deg: float     # direction of the face's long edge
    tilt_deg: float              # angle between the face normal and vertical
    normal: np.ndarray           # unit normal of the face, pointing up
    extents: tuple               # (long, short) edge lengths seen, m
    n_points: int


def measure_top_face(frame, key) -> TopFace | None:
    ids = [i for i, k in frame.instance_map.items() if k == key]
    if not ids:
        return None
    pts, inst, _ = frame.unproject(step=1)
    mine = pts[np.isin(inst, ids)]
    if len(mine) < MIN_POINTS:
        return None
    top = mine[mine[:, 2] > mine[:, 2].max() - TOP_BAND]
    if len(top) < MIN_POINTS:
        return None
    centred = top - top.mean(0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    normal = vt[2] if vt[2][2] >= 0 else -vt[2]
    tilt = math.degrees(math.acos(min(1.0, abs(float(normal[2])))))
    xy = top[:, :2] - top[:, :2].mean(0)
    _, v = np.linalg.eigh(xy.T @ xy)
    a, b = xy @ v[:, 1], xy @ v[:, 0]
    centre = top[:, :2].mean(0) + v[:, 1] * (a.max() + a.min()) / 2 + v[:, 0] * (b.max() + b.min()) / 2
    yaw = math.degrees(math.atan2(v[1, 1], v[0, 1]))
    return TopFace(centre, float(np.median(top[:, 2])), yaw, tilt, normal,
                   (float(a.max() - a.min()), float(b.max() - b.min())), int(len(top)))
