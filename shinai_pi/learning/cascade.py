"""Learning where the geometric cascade rule is wrong.

Training pairs come from physics: in a collapsed pile, one case is removed and the simulator says
which neighbours actually moved. Features describe the pair as the robot's own belief sees it.
The model is a logistic regression stored as plain JSON (no pickled code), so it can be shipped
and audited."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..pile import SupportGraph

FEATURES = (
    'geo_falls', 'geo_severity', 'vertical_share_on_removed', 'leans_on_removed', 'margin_after',
    'margin_before', 'tilt', 'bottom_height', 'distance', 'unexplained', 'n_supporters',
)
NEIGHBOUR_RADIUS = 0.6


def pair_features(g: SupportGraph, removed: str, other: str, severity: dict, fallen: set) -> np.ndarray:
    present = set(g.boxes)
    after = present - {removed} - (fallen - {other})
    o = g.obbs[other]
    lo, _ = o.aabb()
    return np.array([
        1.0 if other in fallen else 0.0,
        severity.get(other, 0.0),
        g.vertical_share(other, removed),
        1.0 if removed in g._leaners(other, present) else 0.0,
        float(np.clip(g.support_margin(other, after), -0.3, 0.3)),
        float(np.clip(g.support_margin(other, present), -0.3, 0.3)),
        o.tilt_deg() / 90.0,
        float(lo[2]),
        float(np.linalg.norm(o.center - g.obbs[removed].center)),
        1.0 if other in g.unexplained() else 0.0,
        float(len(g.supporters(other))),
    ])


def neighbours(g: SupportGraph, removed: str):
    c = g.obbs[removed].center
    return [k for k in g.boxes if k != removed and np.linalg.norm(g.obbs[k].center - c) <= NEIGHBOUR_RADIUS]


def removal_features(g: SupportGraph, removed: str):
    sev = g.fall_severity(removed)
    fallen = set(sev)
    keys = neighbours(g, removed)
    X = np.array([pair_features(g, removed, k, sev, fallen) for k in keys]) if keys else np.empty((0, len(FEATURES)))
    return keys, X


@dataclass
class CascadeModel:
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    bias: float
    meta: dict

    def predict(self, X: np.ndarray) -> np.ndarray:
        if len(X) == 0:
            return np.zeros(0)
        z = ((X - self.mean) / self.scale) @ self.weights + self.bias
        return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))

    def to_json(self) -> str:
        return json.dumps({'features': FEATURES, 'mean': self.mean.tolist(), 'scale': self.scale.tolist(),
                           'weights': self.weights.tolist(), 'bias': self.bias, 'meta': self.meta}, indent=1, ensure_ascii=False)

    @staticmethod
    def load(path: Path) -> 'CascadeModel':
        d = json.loads(Path(path).read_text(encoding='utf-8'))
        if tuple(d['features']) != FEATURES:
            raise ValueError('model was trained on a different feature set')
        return CascadeModel(np.array(d['mean']), np.array(d['scale']), np.array(d['weights']), float(d['bias']), d['meta'])


def fit(X: np.ndarray, y: np.ndarray, meta: dict, C: float = 1.0) -> CascadeModel:
    from sklearn.linear_model import LogisticRegression
    mean, scale = X.mean(0), X.std(0) + 1e-6
    # No class re-weighting: the probabilities feed an expected count, so they must stay calibrated.
    lr = LogisticRegression(C=C, max_iter=2000).fit((X - mean) / scale, y)
    return CascadeModel(mean, scale, lr.coef_[0], float(lr.intercept_[0]), meta)


def expected_falls(model: CascadeModel, g: SupportGraph, removed: str) -> float:
    """Expected number of cases that move when `removed` is taken, weighted by drop severity."""
    keys, X = removal_features(g, removed)
    if not keys:
        return 0.0
    p = model.predict(X)
    sev = np.array([max(x[1], 0.3) for x in X])        # unknown drop: count at least a small move
    return float(np.sum(p * sev))


def brier(p, y):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def calibration_bins(p, y, n=5):
    p, y = np.asarray(p), np.asarray(y)
    edges = np.linspace(0, 1, n + 1)
    out = []
    for a, b in zip(edges, edges[1:]):
        m = (p >= a) & (p < b) if b < 1 else (p >= a) & (p <= b)
        if m.any():
            out.append({'range': [round(a, 2), round(b, 2)], 'n': int(m.sum()), 'predicted': round(float(p[m].mean()), 3),
                        'observed': round(float(y[m].mean()), 3)})
    return out


def auc(p, y):
    p, y = np.asarray(p, float), np.asarray(y, int)
    pos, neg = p[y == 1], p[y == 0]
    if not len(pos) or not len(neg):
        return math.nan
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
