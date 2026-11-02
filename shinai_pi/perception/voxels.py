"""Known-space map: every voxel is FREE (a ray passed through), OCCUPIED (a ray ended in it)
or UNKNOWN (no ray has reached it). UNKNOWN is never treated as empty."""
from __future__ import annotations

import numpy as np

from .geometry import CameraFrame

UNKNOWN, FREE, OCCUPIED = 0, 1, 2


class VoxelMap:
    def __init__(self, lo, hi, res: float):
        self.lo = np.asarray(lo, float)
        self.hi = np.asarray(hi, float)
        self.res = float(res)
        self.shape = tuple(np.ceil((self.hi - self.lo) / self.res).astype(int))
        self.free_n = np.zeros(self.shape, dtype=np.uint16)
        self.occ_n = np.zeros(self.shape, dtype=np.uint16)

    def copy(self) -> 'VoxelMap':
        m = VoxelMap(self.lo, self.hi, self.res)
        m.free_n, m.occ_n = self.free_n.copy(), self.occ_n.copy()
        return m

    def _index(self, pts):
        idx = np.floor((np.asarray(pts, float) - self.lo) / self.res).astype(int)
        ok = np.all((idx >= 0) & (idx < np.array(self.shape)), axis=-1)
        return idx, ok

    def centers(self) -> np.ndarray:
        g = np.indices(self.shape).reshape(3, -1).T
        return (self.lo + (g + 0.5) * self.res).reshape(*self.shape, 3)

    def state(self) -> np.ndarray:
        s = np.full(self.shape, UNKNOWN, dtype=np.uint8)
        s[self.free_n > 0] = FREE
        s[(self.occ_n > 0) & (self.occ_n * 3 >= self.free_n)] = OCCUPIED
        return s

    def state_at(self, pts) -> np.ndarray:
        idx, ok = self._index(pts)
        out = np.full(len(idx), UNKNOWN, dtype=np.uint8)
        s = self.state()
        out[ok] = s[tuple(idx[ok].T)]
        return out

    def _box_slice(self, lo, hi):
        a = np.clip(np.floor((np.asarray(lo) - self.lo) / self.res).astype(int), 0, self.shape)
        b = np.clip(np.ceil((np.asarray(hi) - self.lo) / self.res).astype(int), 0, self.shape)
        return tuple(slice(i, j) for i, j in zip(a, b))

    def unknown_fraction(self, lo, hi) -> float:
        sl = self._box_slice(lo, hi)
        s = self.state()[sl]
        return float(np.mean(s == UNKNOWN)) if s.size else 0.0

    def unknown_volume(self, lo, hi) -> float:
        return float(np.sum(self.state()[self._box_slice(lo, hi)] == UNKNOWN)) * self.res ** 3

    def integrate(self, frame: CameraFrame, step: int = 2, max_range: float = 8.0):
        """Ray-march every `step`-th pixel through the grid."""
        dirs, d_cam, v, u = frame.pixel_rays(step)
        depth = frame.depth[v, u].astype(float)
        cos = -(d_cam / np.linalg.norm(d_cam, axis=1, keepdims=True))[:, 2]
        hit = np.isfinite(depth) & (depth > 0)
        rng = np.where(hit, depth / cos, max_range)
        origin = frame.position
        # Clip each ray to the grid box.
        with np.errstate(divide='ignore', invalid='ignore'):
            t1 = (self.lo - origin) / dirs
            t2 = (self.hi - origin) / dirs
        t_in = np.clip(np.nanmax(np.minimum(t1, t2), axis=1), 0, None)
        t_out = np.nanmin(np.maximum(t1, t2), axis=1)
        valid = t_out > t_in
        step_len = self.res * 0.5
        free_seen = np.zeros(self.free_n.size, dtype=bool)     # a voxel counts once per frame
        occ_seen = np.zeros(self.occ_n.size, dtype=bool)
        dims = np.array(self.shape)
        lo32 = self.lo.astype(np.float32)
        for chunk in np.array_split(np.nonzero(valid)[0], max(1, int(valid.sum()) // 4000)):
            if not len(chunk):
                continue
            a, b = t_in[chunk], np.minimum(t_out[chunk], rng[chunk])
            n = int(np.ceil(np.max(b - a) / step_len)) + 1
            ts = (a[:, None] + np.arange(n)[None, :] * step_len).astype(np.float32)
            keep = ts < (b - self.res)[:, None]               # stop one voxel short of the surface
            rows = np.nonzero(keep)[0]
            pts = origin.astype(np.float32) + ts[keep][:, None] * dirs[chunk].astype(np.float32)[rows]
            idx = np.floor((pts - lo32) / self.res).astype(np.int32)
            ok = np.all((idx >= 0) & (idx < dims), axis=1)
            free_seen[np.ravel_multi_index(idx[ok].T, self.shape)] = True
            ends = hit[chunk] & (rng[chunk] <= t_out[chunk])
            if np.any(ends):
                surf = origin + (rng[chunk][ends] + self.res * 0.25)[:, None] * dirs[chunk][ends]
                i2, ok2 = self._index(surf)
                occ_seen[np.ravel_multi_index(i2[ok2].T, self.shape)] = True
        for seen, target in ((free_seen, self.free_n), (occ_seen, self.occ_n)):
            flat = target.reshape(-1)
            flat[seen] = np.minimum(flat[seen].astype(np.int32) + 1, 65535).astype(np.uint16)
