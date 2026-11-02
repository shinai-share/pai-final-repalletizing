"""F0-T: how tipping a case over its bottom edge on the righting table behaves in the physics model
(requirements v05 12.4 (a)). Runs the executive's own righting on the S1 scenes and samples the truth
every physics step while the case is on the table:

  * slip of the pivot edge along the table (stand-in for scuffing the carton),
  * whether it stays on the table and how far it leans after the tip,
  * vertical speed of the case when its new bottom meets the table (landing shock),
  * the most the cups carry while it tips (the wrist reading is the cups' load: no tines are in).

    .venv-isaac/Scripts/python scripts/probe_tip.py [--trials 5]
Writes artifacts/f0_tip/probe_tip.json. A virtual work cell: carton scuffing is not measured, only
the slip that would cause it."""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
import stage_trials as st  # noqa: E402
import shinai_pi.executive as X  # noqa: E402
from shinai_pi.config import load_cell  # noqa: E402
from shinai_pi.perception.obb import quat_to_matrix  # noqa: E402

CRITERIA = {'edge_slip_mm': 5.0, 'lean_after_deg': 2.0, 'landing_speed_mps': 0.1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trials', type=int, default=5)
    ap.add_argument('--out', default=str(ROOT / 'artifacts' / 'f0_tip'))
    args = ap.parse_args()
    from shinai_pi.sim.runtime import IsaacRuntime
    cell = load_cell()
    sku = cell.sku('BEV-500x24')
    rt = IsaacRuntime(cell)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table_top = cell.staging.table_size[2]
    rec = {'on': False}
    orig_step = rt.step
    orig_reorient = X.Executive._reorient

    def step(n=1, render=False):
        orig_step(n, render)
        if not rec['on']:
            return
        pos, quat = rt.truth_case_poses()['case_00']
        R = quat_to_matrix(quat)
        corners = np.array([pos + R @ (np.array([sx, sy, sz]) * np.asarray(sku.size) / 2)
                            for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        low = corners[corners[:, 2] < table_top + 0.004]                # corners on the table
        rec['samples'].append({'t': rt.sim_time, 'pos': pos.copy(), 'low': low.mean(0) if len(low) else None,
                               'n_low': len(low), 'wrist': rt.wrist_load()[0] if rt.gripped() else 0.0,
                               'held': bool(rt.gripped())})

    def reorient(self, key, *a, **kw):
        rec['on'] = True
        rec['samples'] = []
        rec['tare'] = rt.wrist_load()[0]
        try:
            return orig_reorient(self, key, *a, **kw)
        finally:
            rec['on'] = False

    rt.step = step
    X.Executive._reorient = reorient
    rows = []
    for stage in ('S1-side', 'S1-end'):
        for trial in range(args.trials):
            res = st.run_trial(rt, cell, sku, stage, trial, out / f'{stage}_{trial:02d}', False)
            s = rec.get('samples') or []
            row = {'stage': stage, 'trial': trial, 'status': res.get('status'), 'reoriented': res.get('reoriented')}
            # Two corners on the table: the case rests on an edge (from set-down to the end of the tip).
            edge = [x for x in s if x['n_low'] == 2 and x['held']]
            if edge:
                ref = edge[0]['low']
                row['edge_slip_mm'] = round(max(float(np.linalg.norm((x['low'] - ref)[:2])) for x in edge) * 1000, 2)
            on_face = [i for i, x in enumerate(s) if x['n_low'] >= 4]
            if len(s) > 2:
                landing = next((i for i in on_face if i > 0 and s[i - 1]['n_low'] < 4 and edge and s[i]['t'] > edge[0]['t']), None)
                if landing is not None:
                    dt = s[landing]['t'] - s[landing - 1]['t']
                    row['landing_speed_mps'] = round(abs(float(s[landing]['pos'][2] - s[landing - 1]['pos'][2])) / max(dt, 1e-9), 3)
            if s:
                row['cups_peak_N'] = round(max(x['wrist'] for x in s) - rec.get('tare', 0.0), 1)
            pos, quat = rt.truth_case_poses()['case_00']
            R = quat_to_matrix(quat)
            row['lean_after_deg'] = round(math.degrees(math.acos(min(1.0, float(np.max(np.abs(R[2, :])))))), 2)
            row['on_table_or_placed'] = bool(pos[2] > table_top - 0.01)
            row['passed'] = bool(row.get('reoriented') and row.get('edge_slip_mm', 99) <= CRITERIA['edge_slip_mm']
                                 and row.get('landing_speed_mps', 0.0) <= CRITERIA['landing_speed_mps'])
            rows.append(row)
            print('TIP', json.dumps(row, ensure_ascii=False), flush=True)
    report = {'criteria': CRITERIA, 'rows': rows, 'passed': sum(r['passed'] for r in rows), 'runs': len(rows)}
    (out / 'probe_tip.json').write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding='utf-8')
    print('F0T', json.dumps({k: report[k] for k in ('passed', 'runs')}), flush=True)
    rt.close()


if __name__ == '__main__':
    main()
