"""Scores an episode against ground truth. Lives outside the cognitive core: it listens to the
event log and reads truth from the simulator at the moments those events mark."""
from __future__ import annotations

from collections import Counter

import numpy as np

from ..executive import POS_TOL, TILT_TOL_DEG, YAW_TOL_DEG
from .truth import quat_tilt_deg, quat_yaw_error_deg

DISTURB_M = 0.05
DROP_M = 0.10
COLLAPSE_DROP_M = 0.25
PUSHED_M = 0.003                 # a placed case moved this far after it settled was pushed, not settling
CASE_STATES = ('placed_on_b', 'held_by_robot', 'handed_over', 'outside_cell', 'left_in_cell')


def on_pallet_a(cell, p) -> bool:
    a = cell.pallet
    return abs(p[0] - a.a_center[0]) <= a.size[0] / 2 and abs(p[1] - a.a_center[1]) <= a.size[1] / 2 and p[2] > a.size[2]


class PalletExitClock:
    """The first simulated second each case was off pallet A during the transport (before the robot starts)."""

    def __init__(self, cell):
        self.cell = cell
        self.left = {}

    def tick(self, t, poses):
        for k, (p, _) in poses.items():
            if k not in self.left and not on_pallet_a(self.cell, p):
                self.left[k] = round(float(t), 2)


class TruthRecorder:
    def __init__(self, truth_fn, cell, slots, left_in_transport=None):
        self.truth_fn = truth_fn            # () -> {case: (pos, quat)}
        self.cell = cell
        self.slots = slots
        self.picks = []
        self._pre = None
        self.drops = 0
        self.slot_of = {}                    # case -> slot index named by its PLACED event
        self.initial = truth_fn()
        self.left_in_transport = left_in_transport     # case -> transport second it left pallet A (None: not clocked)
        self.settled_at = {}                 # placed case -> where it was when it last settled
        self.pushed = []

    def on_event(self, e):
        if e.type == 'PICK_PLANNED':
            self._pre = {'case': e.data['case'], 'truth': self.truth_fn(), 'predicted': set(e.data.get('predicted_fall', [])),
                         'risk': e.data.get('risk'), 'fallback': e.data.get('fallback')}
        elif e.type == 'PILE_CHANGED' and self._pre is not None:
            post = self.truth_fn()
            target = self._pre['case']
            disturbed, collapsed = set(), set()
            for k, (p0, _) in self._pre['truth'].items():
                if k == target or k not in post:
                    continue
                p1 = post[k][0]
                if np.linalg.norm(p1[:2] - p0[:2]) > DISTURB_M or p0[2] - p1[2] > DROP_M:
                    disturbed.add(k)
                if p0[2] - p1[2] > COLLAPSE_DROP_M:
                    collapsed.add(k)
            self.picks.append({'case': target, 'risk': self._pre['risk'], 'fallback': self._pre['fallback'],
                               'predicted': sorted(self._pre['predicted']), 'disturbed': sorted(disturbed),
                               'collapsed': sorted(collapsed)})
            self._pre = None
        elif e.type == 'PLACED':
            self.slot_of[e.data['case']] = e.data['slot']
        elif e.type == 'DROP_DETECTED':
            self.drops += 1
        elif e.type == 'PLACEMENT_SETTLED' and e.data.get('ok'):
            now = self.truth_fn()
            self.pushed += self._pushes(now, e.data['case'])
            self.settled_at.update({k: now[k][0][:2] for k in self.settled_at if k in now})
            if e.data['case'] in now:
                self.settled_at[e.data['case']] = now[e.data['case']][0][:2]

    def _pushes(self, now, placing):
        """Placed cases that moved since they last settled, while `placing` was put down (None: after the last)."""
        out = []
        for k, p0 in self.settled_at.items():
            if k == placing or k not in now:
                continue
            d = float(np.linalg.norm(now[k][0][:2] - p0))
            if d > PUSHED_M:
                out.append({'case': k, 'while_placing': placing, 'moved_mm': round(d * 1000, 1)})
        return out

    def _on_pallet_a(self, p) -> bool:
        return on_pallet_a(self.cell, p)

    def _ledger(self, report, belief_keys, final, placed, unscored, held):
        """One final state per case of the load, so the states add up to the load; a placed case carries its
        placement quality apart, and a case that left pallet A says whether it was already off when the
        robot started or which pick moved it."""
        g = self.cell.gantry
        quality = {p['case']: ('ok' if (p['xy_err_m'] <= POS_TOL and abs(p['z_err_m']) <= POS_TOL and p['yaw_deg'] <= YAW_TOL_DEG
                                        and p['tilt_deg'] <= TILT_TOL_DEG) else 'ng') for p in placed}
        quality.update({k: 'unscored' for k in unscored})
        ledger = {}
        for k, (p, _) in final.items():
            in_area = g.x_range[0] <= p[0] <= g.x_range[1] and g.y_range[0] <= p[1] <= g.y_range[1]
            if k in held:
                state, detail = 'held_by_robot', 'ケースを持ったまま人の対応を待つ'
            elif k in report.placed:
                state, detail = 'placed_on_b', ''
            elif k in report.handed_over:
                state, detail = 'handed_over', ''
            elif not in_area:
                state, detail = 'outside_cell', '作業場の外'
            else:
                state = 'left_in_cell'
                detail = report.remaining.get(k) or ('未処理' if k in belief_keys else '未検出')
            p0 = self.initial[k][0] if k in self.initial else None
            if p0 is not None and not self._on_pallet_a(p0):
                left = 'before_mission'
            elif state in ('placed_on_b', 'held_by_robot', 'handed_over') or self._on_pallet_a(p):
                left = None
            else:
                by = next((x['case'] for x in self.picks if k in x['disturbed']), None)
                left = f'during_pick:{by}' if by else 'unknown'
            unplaced = state in ('outside_cell', 'left_in_cell')
            ledger[k] = {'state': state, 'detail': detail, 'quality': quality.get(k) if state == 'placed_on_b' else None,
                         'left_pallet': left, 'detected': k in belief_keys,
                         'left_pallet_at_s': (self.left_in_transport or {}).get(k) if left == 'before_mission' else None,
                         'where': {'off_pallet_a': not self._on_pallet_a(p), 'outside_work_area': not in_area,
                                   'seen_by_robot': k in belief_keys} if unplaced else None}
        return ledger

    def summary(self, report, belief_keys, held=()):
        final = self.truth_fn()
        placed, unscored = [], []
        for k in report.placed:
            if k not in self.slot_of:
                unscored.append(k)
                continue
            slot = self.slots[self.slot_of[k]]
            p, q = final[k]
            placed.append({'case': k, 'slot': slot.index, 'xy_err_m': float(np.linalg.norm(p[:2] - np.array(slot.center[:2]))),
                           'z_err_m': float(p[2] - slot.center[2]), 'tilt_deg': quat_tilt_deg(q),
                           'yaw_deg': quat_yaw_error_deg(q, slot.yaw_deg)})
        a = self.cell.pallet
        moved_away = set(report.placed) | set(report.handed_over)
        on_a_side = [k for k, (p, _) in final.items() if k not in moved_away and p[0] < a.b_center[0] - a.size[0] / 2]
        unknown = sorted(set(on_a_side) - set(belief_keys))
        ledger = self._ledger(report, set(belief_keys), final, placed, unscored, set(held))
        states = Counter(v['state'] for v in ledger.values())
        pq = Counter(v['quality'] for v in ledger.values() if v['quality'])
        tp = sum(1 for p in self.picks for k in p['disturbed'] if k in p['predicted'])
        fn = sum(1 for p in self.picks for k in p['disturbed'] if k not in p['predicted'])
        fp = sum(1 for p in self.picks for k in p['predicted'] if k not in p['disturbed'])
        return {
            'picks': len(self.picks),
            'secondary_collapses': sum(len(p['collapsed']) for p in self.picks),
            'disturbed_cases': sum(len(p['disturbed']) for p in self.picks),
            'placed_cases_pushed': self.pushed + self._pushes(final, None),
            'picks_with_disturbance': sum(1 for p in self.picks if p['disturbed']),
            'drops_in_transit': self.drops,
            'prediction': {'true_positive': tp, 'missed': fn, 'false_alarm': fp},
            'placed_on_b': len(report.placed),
            'placement_xy_err_max_m': round(max([p['xy_err_m'] for p in placed], default=0.0), 4),
            'placement_yaw_max_deg': round(max([p['yaw_deg'] for p in placed], default=0.0), 2),
            'placement_tilt_max_deg': round(max([p['tilt_deg'] for p in placed], default=0.0), 2),
            'placement_errors': {p['case']: {'slot': p['slot'], 'xy_m': round(p['xy_err_m'], 4), 'z_m': round(p['z_err_m'], 4),
                                             'tilt_deg': round(p['tilt_deg'], 2), 'yaw_deg': round(p['yaw_deg'], 2)} for p in placed},
            'placed_unscored': unscored,
            'final_positions': {k: [round(float(v), 3) for v in pq[0]] for k, pq in final.items()},
            'handed_over': len(report.handed_over),
            'left_on_a': len(on_a_side),
            'cases_never_detected': unknown,
            'left_on_a_positions': {k: [round(float(v), 3) for v in final[k][0]] for k in on_a_side},
            'per_pick': self.picks,
            'start_positions': {k: [round(float(v), 3) for v in pq0[0]] for k, pq0 in self.initial.items()},
            'case_states': {s: states.get(s, 0) for s in CASE_STATES},
            'placed_quality': {q: pq.get(q, 0) for q in ('ok', 'ng', 'unscored')},
            'case_ledger': ledger,
        }
