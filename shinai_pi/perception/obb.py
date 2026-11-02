"""Oriented boxes: the common geometric currency of perception, belief and evaluation."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def quat_to_matrix(q_wxyz) -> np.ndarray:
    w, x, y, z = (float(v) for v in q_wxyz)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def matrix_to_quat(R) -> tuple:
    R = np.asarray(R, float)
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = (0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s)
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = ((R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s)
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = ((R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s)
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = ((R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s)
    return tuple(float(v) for v in q)


def yaw_matrix(yaw_rad: float) -> np.ndarray:
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


@dataclass(frozen=True)
class OBB:
    center: np.ndarray        # (3,)
    rotation: np.ndarray      # (3, 3) columns are the box axes in world
    size: np.ndarray          # (3,) full extents along the box axes

    @staticmethod
    def make(center, size, rotation=None):
        R = np.eye(3) if rotation is None else np.asarray(rotation, float)
        return OBB(np.asarray(center, float), R, np.asarray(size, float))

    @property
    def half(self) -> np.ndarray:
        return self.size / 2

    def corners(self) -> np.ndarray:
        s = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)], float)
        return self.center + (s * self.half) @ self.rotation.T

    def to_local(self, pts) -> np.ndarray:
        return (np.asarray(pts, float) - self.center) @ self.rotation

    def contains(self, pts, margin: float = 0.0) -> np.ndarray:
        return np.all(np.abs(self.to_local(pts)) <= self.half + margin, axis=-1)

    def signed_distance(self, pts) -> np.ndarray:
        q = np.abs(self.to_local(pts)) - self.half
        outside = np.linalg.norm(np.maximum(q, 0), axis=-1)
        inside = np.minimum(np.max(q, axis=-1), 0)
        return outside + inside

    def aabb(self):
        c = self.corners()
        return c.min(0), c.max(0)

    @property
    def up_axis(self) -> int:
        """Index of the box axis closest to world vertical."""
        return int(np.argmax(np.abs(self.rotation[2, :])))

    def top_face(self):
        """Centre, outward normal and in-plane half extents of the face pointing most upward."""
        k = self.up_axis
        sign = 1.0 if self.rotation[2, k] >= 0 else -1.0
        n = self.rotation[:, k] * sign
        c = self.center + n * self.half[k]
        others = [i for i in range(3) if i != k]
        return c, n, (self.half[others[0]], self.half[others[1]]), (self.rotation[:, others[0]], self.rotation[:, others[1]])

    def tilt_deg(self) -> float:
        _, n, _, _ = self.top_face()
        return math.degrees(math.acos(max(-1.0, min(1.0, n[2]))))

    def surface_samples(self, per_edge: int = 5) -> np.ndarray:
        g = np.linspace(-1, 1, per_edge)
        pts = []
        for axis in range(3):
            o = [i for i in range(3) if i != axis]
            a, b = np.meshgrid(g, g)
            for sgn in (-1, 1):
                local = np.zeros((a.size, 3))
                local[:, axis] = sgn
                local[:, o[0]] = a.reshape(-1)
                local[:, o[1]] = b.reshape(-1)
                pts.append(local)
        local = np.unique(np.vstack(pts), axis=0) * self.half
        return self.center + local @ self.rotation.T

    def ray_hit(self, origins, dirs):
        """Slab test. Returns entry distance per ray (inf for miss)."""
        o = (origins - self.center) @ self.rotation
        d = dirs @ self.rotation
        with np.errstate(divide='ignore', invalid='ignore'):
            inv = 1.0 / d
            t1 = (-self.half - o) * inv
            t2 = (self.half - o) * inv
        tmin = np.nanmax(np.minimum(t1, t2), axis=-1)
        tmax = np.nanmin(np.maximum(t1, t2), axis=-1)
        hit = (tmax >= np.maximum(tmin, 0))
        return np.where(hit, np.maximum(tmin, 0), np.inf)
