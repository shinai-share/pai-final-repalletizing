"""Every physics step, read from the simulator for scoring only (the robot never sees it): where each case and
tool part is and how fast it moves, the tool's own state, and every touching pair's force as a vector with its
contact points. Cutting it by the work phases in events.jsonl shows what held a case at each moment, which the
per-touch records (contacts.py) cannot: a touch that lasts while the support under it changes has one peak.

Steps are kept in memory and written every `flush_every` steps to numbered .npz files beside an index."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

VECTORS = ('pair_F', 'pt_p', 'pt_n', 'pt_ft')


class StepTrace:
    def __init__(self, folder, rows, cols, on_N: float = 0.5, flush_every: int = 1200, meta=None):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.rows, self.cols = list(rows), list(cols)
        self.on_N = on_N                    # a pair under this is solver noise, not a touch
        self.flush_every = flush_every
        self.steps, self.chunks, self.tool_names = 0, 0, None
        self.closed = False
        self.meta = dict(meta or {})
        self._clear()

    def _clear(self):
        self._t, self._pose, self._vel, self._tool = [], [], [], {}
        self._pairs = {k: [] for k in ('pair_step', 'pair_row', 'pair_col', 'pair_F')}
        self._pts = {k: [] for k in ('pt_step', 'pt_row', 'pt_col', 'pt_fn', 'pt_p', 'pt_n', 'pt_ft')}

    def step(self, t, pose, vel, forces, detail=None, tool=None):
        """`forces[i, j]` is the force on rows[i] from cols[j] (N, world); `detail` is the simulator's per-point
        data (normal force, point, normal, pair counts, pair starts, friction force or None)."""
        if self.closed:
            raise RuntimeError('the trace is closed')
        s = self.steps
        self._t.append(float(t))
        self._pose.append(np.asarray(pose, np.float32))
        self._vel.append(np.asarray(vel, np.float32))
        tool = tool or {}
        if self.tool_names is None:
            self.tool_names = list(tool)
        for k in self.tool_names:
            self._tool.setdefault(k, []).append(np.asarray(tool[k], np.float32))
        f = np.asarray(forces, float)
        for i, j in zip(*np.nonzero(np.linalg.norm(f, axis=2) >= self.on_N)):
            self._pairs['pair_step'].append(s)
            self._pairs['pair_row'].append(i)
            self._pairs['pair_col'].append(j)
            self._pairs['pair_F'].append(f[i, j])
        if detail is not None:
            fn, pts, nrm, counts, starts, ft = detail
            fn, pts, nrm = np.asarray(fn, float).reshape(-1), np.asarray(pts, float), np.asarray(nrm, float)
            ft = np.zeros_like(pts) if ft is None else np.asarray(ft, float)
            counts, starts = np.asarray(counts), np.asarray(starts)
            for i, j in zip(*np.nonzero(counts)):
                for p in range(int(starts[i, j]), int(starts[i, j]) + int(counts[i, j])):
                    for k, v in (('pt_step', s), ('pt_row', i), ('pt_col', j), ('pt_fn', fn[p]), ('pt_p', pts[p]),
                                 ('pt_n', nrm[p]), ('pt_ft', ft[p])):
                        self._pts[k].append(v)
        self.steps += 1
        if len(self._t) >= self.flush_every:
            self._flush()

    def _flush(self):
        if not self._t:
            return
        out = {'t': np.asarray(self._t), 'pose': np.stack(self._pose), 'vel': np.stack(self._vel)}
        out.update({f'tool_{k}': np.stack(v) for k, v in self._tool.items()})
        for k, v in {**self._pairs, **self._pts}.items():
            a = np.asarray(v, np.float32 if k in ('pair_F', 'pt_fn', 'pt_p', 'pt_n', 'pt_ft') else np.int32)
            out[k] = a.reshape(-1, 3) if k in VECTORS else a
        np.savez_compressed(self.folder / f'trace_{self.chunks:05d}.npz', **out)
        self.chunks += 1
        self._clear()

    def close(self):
        if self.closed:
            return
        self._flush()
        self.closed = True
        (self.folder / 'trace_index.json').write_text(json.dumps({
            'rows': self.rows, 'cols': self.cols, 'tool': self.tool_names or [], 'steps': self.steps, 'chunks': self.chunks,
            'on_N': self.on_N, 'pose': 'x y z qw qx qy qz (m, world)', 'vel': 'vx vy vz wx wy wz (m/s, rad/s, world)',
            'pair_F': 'force on row from col (N, world)', **self.meta}, ensure_ascii=False, indent=1), encoding='utf-8')


def load_trace(folder):
    """The whole trace as arrays; pair_step and pt_step count steps from the start of the run in every file."""
    folder = Path(folder)
    idx = json.loads((folder / 'trace_index.json').read_text(encoding='utf-8'))
    parts = [dict(np.load(p)) for p in sorted(folder.glob('trace_*.npz'))]
    n_rows = len(idx['rows'])
    shapes = {'t': (0,), 'pose': (0, n_rows, 7), 'vel': (0, n_rows, 6), 'pair_F': (0, 3), 'pt_p': (0, 3), 'pt_n': (0, 3), 'pt_ft': (0, 3)}
    keys = ['t', 'pose', 'vel', 'pair_step', 'pair_row', 'pair_col', 'pair_F', 'pt_step', 'pt_row', 'pt_col', 'pt_fn', 'pt_p', 'pt_n', 'pt_ft']
    d = {k: (np.concatenate([p[k] for p in parts]) if parts else np.zeros(shapes.get(k, (0,)))) for k in keys}
    d['tool'] = {k: (np.concatenate([p[f'tool_{k}'] for p in parts]) if parts else np.zeros((0,))) for k in idx['tool']}
    return d
