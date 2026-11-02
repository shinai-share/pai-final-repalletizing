"""Contacts between bodies, read from the simulator for scoring only: the robot never sees them, and they
are written to their own files beside the run's events. Each touch between two bodies is one record:
when it began and ended, its peak force, its impulse, and how far each case in it moved meanwhile.
A touch that goes off for less than `gap_s` and comes back is one record that still says how many times
it was on, for how long in all and the gaps between; every raw on-period is kept in its own file, so a
short hit is never hidden by the merging."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class ContactLog:
    def __init__(self, on_N: float = 1.0, path: Path | None = None, gap_s: float = 0.0, raw_path: Path | None = None):
        self.on_N = on_N                     # below this a pair is not touching (solver noise)
        self.gap_s = gap_s
        self.open = {}                       # (a, b) -> the record being built
        self.ending = {}                     # (a, b) -> (record, poses when it went off): kept for gap_s
        self.records = []
        self._fh = Path(path).open('w', encoding='utf-8') if path else None
        self._raw = Path(raw_path).open('w', encoding='utf-8') if raw_path else None

    def step(self, t, dt, rows, cols, forces, poses):
        """`forces[i, j]` is the force on body rows[i] from body cols[j] over the last step (world, N)."""
        mag = np.linalg.norm(np.asarray(forces, float), axis=2)
        now = {}
        for i, a in enumerate(rows):
            for j, b in enumerate(cols):
                if a != b and mag[i, j] >= self.on_N:
                    key = tuple(sorted((a, b)))
                    now[key] = max(now.get(key, 0.0), float(mag[i, j]))
        for key, f in now.items():
            c = self.open.get(key)
            if c is None and key in self.ending:
                c = self.open[key] = self.ending.pop(key)[0]
                gap = t - c['_t1']
                c['gaps_s'] += gap
                c['longest_gap_s'] = max(c['longest_gap_s'], round(gap, 4))
                c['_on'], c['_t0'] = {'start_s': round(t, 4), 'peak_N': 0.0, 'impulse_Ns': 0.0}, t
            if c is None:
                c = self.open[key] = {'a': key[0], 'b': key[1], 'start_s': round(t, 4), 'peak_N': 0.0, 'peak_s': None,
                                      'impulse_Ns': 0.0, 'touches': 0, 'contact_s': 0.0, 'gaps_s': 0.0, 'longest_gap_s': 0.0,
                                      '_from': {k: np.asarray(poses[k], float) for k in key if k in poses},
                                      '_on': {'start_s': round(t, 4), 'peak_N': 0.0, 'impulse_Ns': 0.0}, '_t0': t, '_t1': t}
            c['impulse_Ns'] += f * dt
            c['_on']['impulse_Ns'] += f * dt
            c['_on']['peak_N'] = max(c['_on']['peak_N'], round(f, 1))
            if f > c['peak_N']:
                c['peak_N'], c['peak_s'] = round(f, 1), round(t, 4)
        for key in [k for k in self.open if k not in now]:
            c = self.open.pop(key)
            self._off(c, t)
            self.ending[key] = (c, {k: np.asarray(v, float).copy() for k, v in poses.items() if k in key})
        for key in [k for k, (c, _) in self.ending.items() if t - c['end_s'] >= self.gap_s]:
            c, at = self.ending.pop(key)
            self._finish(c, at, open_at_end=False)

    def close(self, t, poses=None):
        for c, at in list(self.ending.values()):
            self._finish(c, at, open_at_end=False)
        self.ending = {}
        for key in list(self.open):
            c = self.open.pop(key)
            self._off(c, t)
            self._finish(c, poses or {}, open_at_end=True)
        for fh in (self._fh, self._raw):
            if fh:
                fh.close()
        self._fh = self._raw = None

    def _off(self, c, t):
        """One raw on-period of a touch ends at `t`."""
        on = c.pop('_on')
        c['end_s'] = round(t, 4)
        c['touches'] += 1
        c['contact_s'] += t - c['_t0']
        c['_t1'] = t
        if self._raw:
            self._raw.write(json.dumps({'a': c['a'], 'b': c['b'], **on, 'end_s': round(t, 4)}, ensure_ascii=False) + '\n')

    def _finish(self, c, poses, open_at_end):
        start = c.pop('_from')
        c.pop('_t0'), c.pop('_t1')
        c['contact_s'], c['gaps_s'] = round(c['contact_s'], 4), round(c['gaps_s'], 4)
        c['moved_mm'] = {k: round(float(np.linalg.norm(np.asarray(poses[k], float) - p0)) * 1000, 1)
                         for k, p0 in start.items() if k in poses}
        c['open_at_end'] = open_at_end
        self.records.append(c)
        if self._fh:
            self._fh.write(json.dumps(c, ensure_ascii=False) + '\n')
            self._fh.flush()
