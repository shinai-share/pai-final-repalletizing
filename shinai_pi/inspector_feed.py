"""What the Inspector shows, derived only from the robot's own belief and events.

A snapshot is a plain dict: case footprints seen from above with their state, the floor area
the robot has not seen (unexplained unknown space projected down), and progress. Writing it to
disk lets live view and replay use the same screen."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

UNKNOWN_CELL = 0.08            # m, grid of the "not seen" overlay


def case_label(key: str) -> str:
    """Internal id -> label people read ("case_07" -> "ケース 7")."""
    if key.startswith('case_'):
        return f'ケース {int(key.split("_")[1])}'
    return key


def belief_snapshot(belief, t: float, next_pick=None, predicted=(), cell=None, progress=None, exclude=()) -> dict:
    boxes = []
    graph = belief.graph()
    everyone = set(graph.boxes)
    cand = {c.box_id: c for c in belief.candidates()}
    for k, b in belief.boxes.items():
        if k in belief.removed or k in exclude:
            continue
        o = b.obb
        c, n, half, (u, v) = o.top_face()
        corners = [c + su * half[0] * u + sv * half[1] * v for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        ca = cand.get(k)
        state = 'next' if k == next_pick else ('falls' if k in predicted else
                ('safe' if ca and ca.pickable and ca.expected_cascade <= belief.max_cascade_risk else
                 ('risky' if ca and ca.pickable else 'blocked')))
        boxes.append({'id': k, 'label': case_label(k), 'poly': [[round(float(p[0]), 3), round(float(p[1]), 3)] for p in corners],
                      'top': round(float(c[2]), 3), 'confidence': round(b.confidence, 2), 'tilt': round(o.tilt_deg(), 1),
                      'risk': None if ca is None else round(ca.expected_cascade, 2), 'reason': '' if ca is None else ca.reason,
                      'state': state, 'unexplained': k in graph.unexplained(),
                      # Where the mass may sit (box centre, +- what the contents allow) and what holds it.
                      'com': [round(float(o.center[0]), 3), round(float(o.center[1]), 3)], 'com_margin': belief.com_margin,
                      'stability': graph.stability(k, everyone) if k in everyone else None,
                      'rests_on': sorted(graph.supporters(k)) if k in everyone else [],
                      'leans_on': sorted(graph.leans_on(k)) if k in everyone else []})
    pts = belief.unexplained_points()
    cells = {}
    for x, y, z in pts:
        key = (int(np.floor(x / UNKNOWN_CELL)), int(np.floor(y / UNKNOWN_CELL)))
        cells[key] = cells.get(key, 0) + 1
    unknown = [[round(i * UNKNOWN_CELL, 3), round(j * UNKNOWN_CELL, 3), n] for (i, j), n in cells.items()]
    snap = {'t': round(float(t), 3), 'boxes': boxes, 'unknown_cells': unknown, 'unknown_cell': UNKNOWN_CELL,
            'unknown_litres': None if belief.unexplained_litres() is None else round(belief.unexplained_litres(), 1),
            'hidden_probability': round(belief.hidden_box_probability(), 3),
            'next_pick': next_pick, 'predicted': sorted(predicted), 'next_hidden': None}
    if next_pick in belief.boxes:
        o = belief.boxes[next_pick].obb
        snap['next_hidden'] = round(belief.hidden_box_probability(near=o, reach=0.5 * float(np.max(o.size)),
                                                                  above=float(o.aabb()[0][2])), 3)
    if cell is not None:
        p = cell.pallet
        snap['pallets'] = {'A': [p.a_center[0], p.a_center[1], p.size[0], p.size[1]], 'B': [p.b_center[0], p.b_center[1], p.size[0], p.size[1]]}
        snap['handover'] = [list(s) for s in cell.handover.spots]
        snap['staging'] = [list(t) for t in cell.staging.tables]
    if progress is not None:
        snap['progress'] = progress
    return snap


class RunRecorder:
    """Appends belief snapshots for the Inspector; frames are written by the backend."""

    def __init__(self, run_dir: Path, executive, cell):
        self.dir = Path(run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.ex = executive
        self.cell = cell
        self._fh = (self.dir / 'belief.jsonl').open('w', encoding='utf-8')
        self._next, self._pred = None, ()
        self._gone = set()

    def on_event(self, e):
        if e.type == 'PICK_PLANNED':
            self._next, self._pred = e.data['case'], tuple(e.data.get('predicted_fall', []))
        elif e.type in ('PLACED', 'HANDOVER', 'PILE_CHANGED'):
            self._next, self._pred = None, ()
            if e.data.get('case'):
                self._gone.add(e.data['case'])
        if e.type in ('BELIEF_UPDATED', 'PICK_PLANNED', 'PLACED', 'HANDOVER') and self.ex.belief is not None:
            r = self.ex.report
            progress = {'placed': len(r.placed), 'handed_over': len(r.handed_over), 'slots': len(self.ex.slots)}
            snap = belief_snapshot(self.ex.belief, e.t, self._next, self._pred, self.cell, progress, exclude=self._gone)
            snap['seq'] = e.seq
            self._fh.write(json.dumps(snap, ensure_ascii=False) + '\n')
            self._fh.flush()

    def close(self):
        self._fh.close()
