"""Pose error between an estimated box and the ground truth, matching axes by edge length."""
from __future__ import annotations

import math

import numpy as np


def box_axis_error_deg(est_rotation, est_size, true_rotation, true_size) -> float:
    """Largest angle between corresponding edges, where edges correspond by length.
    Sign of an axis is irrelevant (a box is symmetric under 180-degree flips)."""
    est_rotation, true_rotation = np.asarray(est_rotation), np.asarray(true_rotation)
    worst = 0.0
    for i, d in enumerate(est_size):
        j = int(np.argmin(np.abs(np.asarray(true_size) - d)))
        c = abs(float(est_rotation[:, i] @ true_rotation[:, j]))
        worst = max(worst, math.degrees(math.acos(min(1.0, c))))
    return worst


def box_position_error(est_center, true_center) -> float:
    return float(np.linalg.norm(np.asarray(est_center) - np.asarray(true_center)))
