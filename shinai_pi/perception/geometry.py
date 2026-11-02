"""Camera model shared by the Isaac adapter, the synthetic renderer and perception.

Convention (verified against Isaac Sim 6.2 annotators): the view matrix maps world to camera
for ROW vectors (p_cam = p_world @ view); the camera looks along -Z with +Y up; depth is the
distance to the image plane."""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np


@dataclass
class CameraFrame:
    depth: np.ndarray            # (H, W) metres to image plane, inf/nan = no return
    instance: np.ndarray         # (H, W) integer ids, 0 = none
    instance_map: dict           # id -> object key (e.g. "case_07" or "floor")
    view: np.ndarray             # (4, 4) world->camera, row-vector convention
    fx: float
    fy: float
    width: int
    height: int
    stamp: float
    name: str = ''
    extra: dict = field(default_factory=dict)

    @property
    def cam_to_world(self) -> np.ndarray:
        return np.linalg.inv(self.view)

    @property
    def position(self) -> np.ndarray:
        return self.cam_to_world[3, :3].copy()

    def pixel_rays(self, step: int = 1):
        """Unit ray directions in world for a pixel grid, plus the (v, u) indices."""
        v, u = np.mgrid[0:self.height:step, 0:self.width:step]
        x = (u + 0.5 - self.width / 2) / self.fx
        y = -(v + 0.5 - self.height / 2) / self.fy
        d_cam = np.stack([x, y, -np.ones_like(x)], -1).reshape(-1, 3)
        R = self.cam_to_world[:3, :3]
        d_w = d_cam @ R
        return d_w / np.linalg.norm(d_w, axis=1, keepdims=True), d_cam, v.reshape(-1), u.reshape(-1)

    def unproject(self, step: int = 1):
        """World points, instance ids and range per valid pixel."""
        _, d_cam, v, u = self.pixel_rays(step)
        z = self.depth[v, u].astype(float)
        ok = np.isfinite(z) & (z > 0)
        pc = d_cam[ok] * z[ok, None]
        pts = np.hstack([pc, np.ones((len(pc), 1))]) @ self.cam_to_world
        return pts[:, :3], self.instance[v[ok], u[ok]], np.linalg.norm(pc, axis=1)


def intrinsics_from_projection(projection: np.ndarray, width: int, height: int):
    return float(projection[0, 0] * width / 2), float(projection[1, 1] * height / 2)


def frame_from_npz(path, name: str = '') -> CameraFrame:
    d = np.load(path)
    w, h = (int(v) for v in d['resolution'])
    fx, fy = intrinsics_from_projection(d['projection'], w, h)
    raw = json.loads(str(d['instance_map']))
    return CameraFrame(depth=d['depth'], instance=d['instance'], instance_map=normalise_instance_map(raw),
                       view=d['view'], fx=fx, fy=fy, width=w, height=h, stamp=float(d['stamp']), name=name or str(path))


def normalise_instance_map(raw: dict) -> dict:
    """Map Isaac prim paths to object keys: cases by id, everything else by its top-level name."""
    out = {}
    for k, path in raw.items():
        parts = str(path).strip('/').split('/')
        if 'cases' in parts:
            out[int(k)] = parts[parts.index('cases') + 1]
        elif len(parts) >= 3:
            out[int(k)] = parts[2]
        else:
            out[int(k)] = parts[-1]
    return out


def look_at_view(eye, target, up=(0, 0, 1)) -> np.ndarray:
    """World->camera matrix (row-vector convention) for a camera at eye looking at target."""
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    zc = eye - target
    zc /= np.linalg.norm(zc)
    xc = np.cross(up, zc)
    if np.linalg.norm(xc) < 1e-9:
        xc = np.array([1.0, 0, 0])
    xc /= np.linalg.norm(xc)
    yc = np.cross(zc, xc)
    c2w = np.eye(4)
    c2w[0, :3], c2w[1, :3], c2w[2, :3], c2w[3, :3] = xc, yc, zc, eye
    return np.linalg.inv(c2w)
