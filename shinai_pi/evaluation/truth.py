"""Ground-truth bookkeeping for evaluation only. The cognitive core must never import this."""
from __future__ import annotations

import math

import numpy as np

SHIFT_M = 0.03
TILT_DEG = 15.0


def quat_tilt_deg(q_wxyz) -> float:
    """Angle between the body z axis and world vertical. A box upside down has the same shape as
    one the right way up, so the axis direction does not matter (0 to 90 deg)."""
    w, x, y, z = q_wxyz
    zz = abs(1 - 2 * (x * x + y * y))
    return math.degrees(math.acos(max(-1.0, min(1.0, zz))))


def quat_yaw_error_deg(q_wxyz, slot_yaw_deg) -> float:
    """Angle between the body x axis, projected on the floor, and the slot heading. A box turned
    half round has the same footprint, so the error folds into 0 to 90 deg."""
    w, x, y, z = q_wxyz
    yaw = math.degrees(math.atan2(2 * (x * y + w * z), 1 - 2 * (y * y + z * z)))
    d = (yaw - slot_yaw_deg) % 180.0
    return min(d, 180.0 - d)


def classify(spawns, final_poses: dict, pallet_center, pallet_size) -> dict:
    """Per-case state after transport: intact / shifted / tilted / fallen (left its layer) / floor."""
    out = {}
    top = pallet_size[2]
    hx, hy = pallet_size[0] / 2, pallet_size[1] / 2
    for s in spawns:
        pos, quat = final_poses[s.case_id]
        pos = np.asarray(pos, dtype=float)
        tilt = quat_tilt_deg(quat)
        moved = float(np.linalg.norm(pos[:2] - np.asarray(s.pos[:2])))
        drop = float(s.pos[2] - pos[2])
        on_pallet = abs(pos[0] - pallet_center[0]) <= hx and abs(pos[1] - pallet_center[1]) <= hy
        if pos[2] < top and not on_pallet:
            state = 'floor'
        elif drop > 0.05:
            state = 'fallen'
        elif tilt > TILT_DEG:
            state = 'tilted'
        elif moved > SHIFT_M:
            state = 'shifted'
        else:
            state = 'intact'
        out[s.case_id] = {'state': state, 'moved_m': round(moved, 3), 'drop_m': round(drop, 3), 'tilt_deg': round(tilt, 1)}
    return out


def summarize(classes: dict) -> dict:
    counts = {k: 0 for k in ('intact', 'shifted', 'tilted', 'fallen', 'floor')}
    for v in classes.values():
        counts[v['state']] += 1
    return counts
