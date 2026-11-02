"""Analytic depth/instance renderer for oriented boxes and a floor plane.

Used by tests and by the fast logic backend; it produces the same CameraFrame as Isaac."""
from __future__ import annotations

import numpy as np

from .geometry import CameraFrame
from .obb import OBB


def render(view: np.ndarray, boxes: dict, width=320, height=240, fx=260.0, fy=260.0, floor_z=0.0,
           noise_std: float = 0.0, seed: int = 0, stamp: float = 0.0, name: str = '') -> CameraFrame:
    frame = CameraFrame(np.zeros((height, width)), np.zeros((height, width), dtype=np.int32), {}, view, fx, fy, width, height, stamp, name)
    dirs, d_cam, v, u = frame.pixel_rays(1)
    origin = frame.position
    best = np.full(len(dirs), np.inf)
    ids = np.zeros(len(dirs), dtype=np.int32)
    imap = {}
    with np.errstate(divide='ignore', invalid='ignore'):
        t_floor = (floor_z - origin[2]) / dirs[:, 2]
    t_floor = np.where(t_floor > 0, t_floor, np.inf)
    best, ids = np.where(t_floor < best, t_floor, best), np.where(t_floor < best, 1, ids)
    imap[1] = 'floor'
    for i, (key, box) in enumerate(boxes.items(), start=2):
        t = box.ray_hit(origin[None, :], dirs)
        closer = t < best
        best[closer], ids[closer] = t[closer], i
        imap[i] = key
    cos = -(d_cam / np.linalg.norm(d_cam, axis=1, keepdims=True))[:, 2]   # range -> image-plane depth
    depth = best * cos
    if noise_std > 0:
        depth = depth + np.random.default_rng(seed).normal(0, noise_std, depth.shape) * np.isfinite(depth)
    frame.depth = depth.reshape(height, width)
    frame.instance = np.where(np.isfinite(best), ids, 0).reshape(height, width)
    frame.instance_map = imap
    return frame
