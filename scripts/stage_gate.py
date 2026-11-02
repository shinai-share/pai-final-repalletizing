"""Judge mission runs against the stage gates of requirements v04 (chapter 8), fixed beforehand.

    S3  6-24 cases in a collapse that holds itself up: no secondary collapse, no drop, every case
        moved and verified (a hand-over counts only when it was for a reason beyond the robot).
    S4  one full pallet (48 cases): no drop, no failed placement check, hand-overs only for reasons
        beyond the robot.

    python scripts/stage_gate.py S3 artifacts/stages/S3
Prints one line per run and the gate, and writes <dir>/gate.json."""
import json
import sys
from pathlib import Path

def judge(stage, r):
    reasons = []
    if r.get('drops_in_transit', 0):
        reasons.append(f"落下 {r['drops_in_transit']} 回")
    if r.get('failed_checks'):
        reasons.append(f"配置の検証に落ちた {len(r['failed_checks'])} 個")
    if stage == 'S3' and r.get('secondary_collapses', 0):
        reasons.append(f"二次崩れ {r['secondary_collapses']} 個")
    if not str(r.get('status', '')).startswith('verified'):
        reasons.append(f"終わり方 {r.get('status')}")
    led = r.get('case_ledger', {}).values()
    ng = sum(1 for v in led if v['state'] == 'placed_on_b' and v['quality'] != 'ok')
    if ng:
        reasons.append(f'置いた後に基準の外にある {ng} 個')
    out = [v['left_pallet'] for v in led if v['state'] == 'outside_cell']
    pushed = sum(1 for x in out if x and x.startswith('during_pick'))
    unknown = sum(1 for x in out if x in ('unknown', 'not_recorded'))
    if pushed:
        reasons.append(f'ロボットの作業中に作業場の外へ出た {pushed} 個')
    if unknown:
        reasons.append(f'作業場の外へ出た時期が分からない {unknown} 個')
    return reasons


def main(stage, run_dir):
    rows = []
    for res in sorted(Path(run_dir).glob('*/result.json')):
        r = json.loads(res.read_text(encoding='utf-8'))
        why = judge(stage, r)
        rows.append({'run': res.parent.name, 'passed': not why, 'why': why, 'placed_on_b': r.get('placed_on_b'),
                     'handed_over': r.get('handed_over'), 'status': r.get('status'),
                     'placement_xy_err_max_mm': round(1000 * r.get('placement_xy_err_max_m', 0.0), 1),
                     'secondary_collapses': r.get('secondary_collapses'), 'sim_seconds': r.get('sim_seconds')})
        print('RUN', json.dumps(rows[-1], ensure_ascii=False))
    gate = {'stage': stage, 'runs': len(rows), 'passed': sum(r['passed'] for r in rows),
            'gate_passed': bool(rows) and all(r['passed'] for r in rows), 'rows': rows}
    (Path(run_dir) / 'gate.json').write_text(json.dumps(gate, indent=1, ensure_ascii=False), encoding='utf-8')
    print('GATE', json.dumps({k: gate[k] for k in ('stage', 'runs', 'passed', 'gate_passed')}))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
