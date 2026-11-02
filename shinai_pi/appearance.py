"""Colour arithmetic for how the cell looks to people (CIELAB, D65). Material colours in USD are
linear; pictures saved from the cameras are 8-bit sRGB."""
from __future__ import annotations

import numpy as np

_M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
_WHITE = _M @ np.ones(3)


def lab_from_linear(rgb) -> np.ndarray:
    xyz = (_M @ np.asarray(rgb, float)) / _WHITE
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def linear_from_srgb8(rgb) -> np.ndarray:
    c = np.asarray(rgb, float) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def lab_from_srgb8(rgb) -> np.ndarray:
    return lab_from_linear(linear_from_srgb8(rgb))


def delta_e(a_linear, b_linear) -> float:
    """CIE76 difference between two linear colours: about 2 is just visible, 25 is obvious."""
    return float(np.linalg.norm(lab_from_linear(a_linear) - lab_from_linear(b_linear)))


def saturation(rgb) -> float:
    c = np.asarray(rgb, float)
    return float((c.max() - c.min()) / c.max()) if c.max() > 0 else 0.0
