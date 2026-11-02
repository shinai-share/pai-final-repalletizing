"""Fit known-size boxes to segmented depth observations from one or more cameras.

Per object: estimate surface normals, find up to three mutually orthogonal faces, assign the
known case dimensions to the axes, and fix the position. Axes whose ends were not seen are
placed where the box intrudes least into space known to be empty; whatever cannot be resolved
is reported as positional uncertainty instead of being guessed away.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np

from .geometry import CameraFrame
from .obb import OBB
from .voxels import FREE, VoxelMap

FACE_ANGLE_DEG = 12.0
MIN_FACE_POINTS = 25
DIM_TOL = 0.015


@dataclass
class BoxEstimate:
    key: str
    obb: OBB
    confidence: float                 # 0..1, how well the pose is pinned down
    top_visible: float                # fraction of the upward face that was seen
    slack: np.ndarray                 # unresolved positional freedom per box axis (m)
    n_points: int
    faces_seen: int
    views: list = field(default_factory=list)
    stamp: float = 0.0


def frame_points_normals(frame: CameraFrame, step: int = 1):
    """Per-pixel world points and camera-facing normals (finite differences on the depth grid)."""
    H, W = frame.height, frame.width
    _, d_cam, v, u = frame.pixel_rays(1)
    z = frame.depth.reshape(-1).astype(float)
    ok = np.isfinite(z) & (z > 0)
    pc = d_cam * np.where(ok, z, np.nan)[:, None]
    pw = np.hstack([pc, np.ones((len(pc), 1))]) @ frame.cam_to_world
    P = pw[:, :3].reshape(H, W, 3)
    inst = frame.instance
    dx = P[1:-1, 2:] - P[1:-1, :-2]
    dy = P[2:, 1:-1] - P[:-2, 1:-1]
    n = np.cross(dx, dy)
    nn = np.linalg.norm(n, axis=-1, keepdims=True)
    n = np.where(nn > 1e-12, n / np.where(nn > 0, nn, 1), np.nan)
    c = inst[1:-1, 1:-1]
    same = (inst[1:-1, 2:] == c) & (inst[1:-1, :-2] == c) & (inst[2:, 1:-1] == c) & (inst[:-2, 1:-1] == c)
    N = np.full((H, W, 3), np.nan)
    N[1:-1, 1:-1] = np.where(same[..., None], n, np.nan)
    to_cam = frame.position - P
    flip = np.sum(N * to_cam, axis=-1) < 0
    N[flip] *= -1
    sl = (slice(None, None, step), slice(None, None, step))
    return P[sl].reshape(-1, 3), N[sl].reshape(-1, 3), inst[sl].reshape(-1)


def _dominant_direction(normals, exclude=None):
    if len(normals) < MIN_FACE_POINTS:
        return None, None
    cos_t = math.cos(math.radians(FACE_ANGLE_DEG))
    cand = normals[np.linspace(0, len(normals) - 1, min(len(normals), 200)).astype(int)]
    if exclude is not None:
        cand = cand[np.all(np.abs(cand @ np.atleast_2d(exclude).T) < 0.25, axis=1)]
        if not len(cand):
            return None, None
    scores = (cand @ normals.T > cos_t).sum(1)
    best = cand[int(np.argmax(scores))]
    inl = normals @ best > cos_t
    if inl.sum() < MIN_FACE_POINTS:
        return None, None
    m = normals[inl].mean(0)
    return m / np.linalg.norm(m), inl


def _orthonormalise(a, b):
    a = a / np.linalg.norm(a)
    b = b - a * (a @ b)
    b /= np.linalg.norm(b)
    return a, b, np.cross(a, b)


def _min_area_rect_axes(pts2):
    """Unit direction of one side of the minimum-area rectangle around 2-D points."""
    from scipy.spatial import ConvexHull
    if len(pts2) < 3:
        return np.array([1.0, 0.0])
    try:
        hull = pts2[ConvexHull(pts2).vertices]
    except Exception:
        return np.array([1.0, 0.0])
    best, best_dir = np.inf, np.array([1.0, 0.0])
    for i in range(len(hull)):
        e = hull[(i + 1) % len(hull)] - hull[i]
        if np.linalg.norm(e) < 1e-9:
            continue
        e = e / np.linalg.norm(e)
        p = np.array([-e[1], e[0]])
        area = np.ptp(hull @ e) * np.ptp(hull @ p)
        if area < best:
            best, best_dir = area, e
    return best_dir


def _assign_dims(extents, dims):
    """Match the known case dimensions to the box axes. Slack on an axis that was seen edge to
    edge is evidence against the assignment; slack along an unseen depth is not."""
    seen = np.where(extents > 0.05, 1.0, 0.1)
    best, best_cost = None, np.inf
    for perm in set(itertools.permutations(range(3))):
        d = np.array([dims[i] for i in perm])
        violation = np.clip(extents - d - DIM_TOL, 0, None).sum()
        cost = violation * 100 + (np.clip(d - extents, 0, None) * seen).sum()
        if cost < best_cost:
            best, best_cost = d, cost
    return best


def fit_box(key: str, P, N, dims, free_map: VoxelMap | None = None, face_cams=None) -> BoxEstimate | None:
    """P, N: world points and camera-facing normals of one object (NaN normals allowed)."""
    good = np.all(np.isfinite(N), axis=1)
    if len(P) < MIN_FACE_POINTS:
        return None
    normals = N[good]
    n1, inl1 = _dominant_direction(normals)
    if n1 is None:
        return None
    faces = [n1]
    n2, inl2 = _dominant_direction(normals[~inl1], exclude=n1) if (~inl1).sum() >= MIN_FACE_POINTS else (None, None)
    if n2 is not None:
        faces.append(n2)
        a, b, c = _orthonormalise(n1, n2)
        rest = normals[~inl1][~inl2] if inl2 is not None else np.empty((0, 3))
        if len(rest) and np.sum(np.abs(rest @ c) > math.cos(math.radians(FACE_ANGLE_DEG))) >= MIN_FACE_POINTS:
            faces.append(c * np.sign(np.mean(rest[np.abs(rest @ c) > 0.9] @ c)))
    else:
        a = n1
        helper = np.array([0, 0, 1.0]) if abs(a[2]) < 0.9 else np.array([1.0, 0, 0])
        e1 = np.cross(a, helper); e1 /= np.linalg.norm(e1)
        e2 = np.cross(a, e1)
        face_pts = P[good][inl1]
        d2 = _min_area_rect_axes(np.c_[face_pts @ e1, face_pts @ e2])
        b = d2[0] * e1 + d2[1] * e2
        a, b, c = _orthonormalise(a, b)
    R = np.column_stack([a, b, c])
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1
    local = P @ R
    lo, hi = local.min(0), local.max(0)
    extents = hi - lo
    d = _assign_dims(extents, dims)
    slack = np.clip(d - extents, 0, None)
    start = lo.copy()                         # box occupies [start, start + d] along each axis
    pinned = np.zeros(3, bool)
    for f in faces:
        k = int(np.argmax(np.abs(R.T @ f)))
        pinned[k] = True
        if (R[:, k] @ f) > 0:                 # seen face is the +k side, box extends toward -k
            start[k] = hi[k] - d[k]
        else:
            start[k] = lo[k]
    unresolved = np.where(pinned, 0.0, slack)
    if free_map is not None and np.any(~pinned & (slack > free_map.res)):
        start, unresolved = _place_in_known_space(R, start, d, slack, pinned, lo, free_map)
    center_local = start + d / 2
    obb = OBB(R @ center_local, R, d)
    top_vis = _top_visibility(obb, P)
    conf = float(np.clip(0.6 * top_vis + 0.4 * (1.0 - np.sum(unresolved) / np.sum(d)), 0.0, 1.0))
    return BoxEstimate(key, obb, conf, top_vis, unresolved, int(len(P)), len(faces))


def _place_in_known_space(R, start, d, slack, pinned, lo, free_map: VoxelMap):
    """Slide along unpinned axes inside the slack; minimise overlap with known-free voxels."""
    axes = [k for k in range(3) if not pinned[k] and slack[k] > free_map.res]
    grids = [np.linspace(0, slack[k], max(2, int(slack[k] / free_map.res) + 1)) for k in axes]
    state = free_map.state()
    best_score, best = np.inf, start.copy()
    unit = np.stack(np.meshgrid(*[np.linspace(0.1, 0.9, 5)] * 3, indexing='ij'), -1).reshape(-1, 3)
    scores = []
    for offs in itertools.product(*grids):
        s = start.copy()
        for k, o in zip(axes, offs):
            s[k] = lo[k] - o
        samples = (s + unit * d) @ R.T
        idx, ok = free_map._index(samples)
        score = int(np.sum(state[tuple(idx[ok].T)] == FREE))
        scores.append(score)
        if score < best_score:
            best_score, best = score, s
    # Positions that are equally consistent with what was seen remain uncertain.
    ties = sum(1 for sc in scores if sc <= best_score)
    unresolved = np.zeros(3)
    for k, g in zip(axes, grids):
        unresolved[k] = slack[k] * min(1.0, (ties - 1) / max(1, len(g) - 1)) if ties > 1 else 0.0
    return best, unresolved


def _top_visibility(obb: OBB, P, cell: float = 0.02) -> float:
    c, n, half, (u, v) = obb.top_face()
    near = np.abs((P - c) @ n) < 0.015
    if not np.any(near):
        return 0.0
    q = P[near] - c
    a = np.floor((q @ u + half[0]) / cell).astype(int)
    b = np.floor((q @ v + half[1]) / cell).astype(int)
    na, nb = max(1, int(round(2 * half[0] / cell))), max(1, int(round(2 * half[1] / cell)))
    ok = (a >= 0) & (a < na) & (b >= 0) & (b < nb)
    return float(len(set(zip(a[ok], b[ok]))) / (na * nb))


def fit_all(frames, dims_for, free_map: VoxelMap | None = None, step: int = 2) -> dict:
    """dims_for(key) -> case dimensions, or None for objects that are not cases."""
    per_key = {}
    for f in frames:
        P, N, inst = frame_points_normals(f, step)
        for iid, key in f.instance_map.items():
            if dims_for(key) is None:
                continue
            m = inst == iid
            if m.sum() < MIN_FACE_POINTS:
                continue
            acc = per_key.setdefault(key, ([], [], [], 0.0))
            acc[0].append(P[m]); acc[1].append(N[m]); acc[2].append(f.name)
            per_key[key] = (acc[0], acc[1], acc[2], max(acc[3], f.stamp))
    out = {}
    for key, (Ps, Ns, views, stamp) in per_key.items():
        P, N = np.vstack(Ps), np.vstack(Ns)
        ok = np.all(np.isfinite(P), axis=1)
        est = fit_box(key, P[ok], N[ok], dims_for(key), free_map)
        if est is not None:
            est.views, est.stamp = views, stamp
            out[key] = est
    return out
