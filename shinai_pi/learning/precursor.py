"""L1 of the experiment plan (chapter 7): from what the robot observed up to a moment, the chance the
cups let go within the next `horizon` seconds. Offline only: the model never changes a safety limit,
and while it is a candidate the robot keeps its verified rules.

Samples are taken every `every` seconds while the case is held; the label looks ahead, the features
look back (a window ending at the sample time) and use only trace columns the robot observes. Scene
families are kept whole on one side of the train / validation / test split (chapter 8)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

WINDOW = 0.3                   # s of history behind each sample
FEATURES = ('tine_sum', 'tine_asym', 'tine_slope', 'head_fz', 'head_torque', 'axis_speed', 'cmd_accel', 'fork_ext')


def _episodes(campaign_dir: Path):
    for d in sorted(Path(campaign_dir).iterdir()):
        m = d / 'episode_manifest.json'
        if not m.exists():
            continue
        man = json.loads(m.read_text(encoding='utf-8'))
        if man.get('status') != 'complete':
            continue
        yield man, np.load(d / 'trace.npz', allow_pickle=False)


def _features(tr, i0, i1):
    t = tr['t'][i0:i1]
    tl, trr = tr['obs_tine_l'][i0:i1], tr['obs_tine_r'][i0:i1]
    s = tl + trr
    q = np.stack([tr[f'obs_q{k}'][i0:i1] for k in range(6)], 1)
    c = np.stack([tr[f'cmd_q{k}'][i0:i1] for k in range(6)], 1)
    dt = max(float(t[-1] - t[0]), 1e-6)
    v = np.abs(np.diff(q, axis=0)).sum(1) / max(np.median(np.diff(t)), 1e-6) if len(t) > 1 else np.zeros(1)
    cv = np.diff(c, axis=0) / max(np.median(np.diff(t)), 1e-6) if len(t) > 1 else np.zeros((1, 6))
    ca = np.abs(np.diff(cv, axis=0)).sum(1) / max(np.median(np.diff(t)), 1e-6) if len(cv) > 1 else np.zeros(1)
    return [float(s[-1]), float(abs(tl[-1] - trr[-1]) / max(abs(s[-1]), 1.0)), float((s[-1] - s[0]) / dt),
            float(tr['obs_head_fz'][i1 - 1]), float(np.hypot(tr['obs_head_tx'][i1 - 1], tr['obs_head_ty'][i1 - 1])),
            float(v.max()), float(ca.max() if len(ca) else 0.0), float(tr['obs_fork_ext'][i1 - 1])]


def samples(campaign_dir, horizon: float, every: float):
    """(X, y, family, t): one sample every `every` s while held, label = the cups let go within `horizon`."""
    X, y, fam, ts = [], [], [], []
    for man, tr in _episodes(campaign_dir):
        t = tr['t']
        held = tr['obs_gripped'] > 0.5
        drops = t[1:][held[:-1] & ~held[1:]]
        next_t = t[0]
        for i in range(len(t)):
            if t[i] < next_t:
                continue
            next_t = t[i] + every
            if not held[i]:
                continue
            i0 = int(np.searchsorted(t, t[i] - WINDOW))
            if i - i0 < 2:
                continue
            X.append(_features(tr, i0, i + 1))
            y.append(int(np.any((drops > t[i]) & (drops <= t[i] + horizon))))
            fam.append(man['family'])
            ts.append(float(t[i]))
    return np.array(X, float).reshape(-1, len(FEATURES)), np.array(y, int), np.array(fam), np.array(ts)


def split_by_family(families, seed: int = 0, shares=(0.6, 0.2, 0.2)):
    """'train' | 'val' | 'test' per sample, decided per family by a hash: a family never straddles."""
    out = np.empty(len(families), dtype=object)
    for f in set(families):
        h = int(hashlib.sha256(f'{seed}:{f}'.encode()).hexdigest(), 16) % 1000 / 1000
        side = 'train' if h < shares[0] else ('val' if h < shares[0] + shares[1] else 'test')
        out[np.asarray(families) == f] = side
    return out


def pr_auc(p, y):
    p, y = np.asarray(p, float), np.asarray(y, int)
    if y.sum() == 0:
        return float('nan')
    order = np.argsort(-p)
    tp = np.cumsum(y[order])
    precision = tp / np.arange(1, len(y) + 1)
    return float(np.sum(precision[y[order] == 1]) / y.sum())


def _metrics(p, y, threshold):
    alarm = p >= threshold
    pos, neg = y == 1, y == 0
    return {'pr_auc': round(pr_auc(p, y), 3), 'recall': round(float(alarm[pos].mean()) if pos.any() else float('nan'), 3),
            'false_alarm_rate': round(float(alarm[neg].mean()) if neg.any() else float('nan'), 3),
            'positives': int(pos.sum()), 'samples': int(len(y))}


def train_and_compare(campaign_dir, horizon=0.5, every=0.1, seed=0):
    """Fit on train families, pick the alarm threshold on validation families, report on test families,
    next to a simple rule (the tines' load turning uneven)."""
    from sklearn.linear_model import LogisticRegression
    X, y, fam, _ = samples(campaign_dir, horizon, every)
    side = split_by_family(fam, seed)
    tr, va, te = side == 'train', side == 'val', side == 'test'
    if y[tr].sum() == 0 or (~y[tr].astype(bool)).sum() == 0:
        raise ValueError('the training families need both outcomes')
    mean, scale = X[tr].mean(0), X[tr].std(0) + 1e-6
    lr = LogisticRegression(C=1.0, max_iter=2000).fit((X[tr] - mean) / scale, y[tr])

    def learned(Z):
        return lr.predict_proba((Z - mean) / scale)[:, 1]
    rule = X[:, FEATURES.index('tine_asym')]
    # Thresholds from validation families only.
    th_l = float(np.quantile(learned(X[va]), 0.9)) if va.any() else 0.5
    th_r = float(np.quantile(rule[va], 0.9)) if va.any() else 0.5
    return {'learned': _metrics(learned(X[te]), y[te], th_l), 'rule': _metrics(rule[te], y[te], th_r),
            'split': {'families': {k: int(len(set(fam[side == k]))) for k in ('train', 'val', 'test')}},
            'samples': int(len(y)), 'horizon_s': horizon,
            'model': {'features': list(FEATURES), 'mean': mean.tolist(), 'scale': scale.tolist(),
                      'weights': lr.coef_[0].tolist(), 'bias': float(lr.intercept_[0]), 'threshold': th_l}}
