"""Where did every case of the load end? For runs recorded before the harness kept a ledger
(result.json without case_ledger): rebuilt from the final true positions and the events. When a case
left pallet A cannot be told from such a record (no start positions): it is marked not_recorded.

    python scripts/case_states.py <run dir> [<run dir> ...]
Prints one line per run and writes case_ledger.json beside each result.json."""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shinai_pi.config import load_cell  # noqa: E402
from shinai_pi.evaluation.harness import CASE_STATES  # noqa: E402
from shinai_pi.executive import POS_TOL, TILT_TOL_DEG, YAW_TOL_DEG  # noqa: E402

LABELS = {'placed_on_b': 'B に置いた', 'held_by_robot': '持ったまま待機', 'handed_over': '人に回した',
          'outside_cell': '作業場の外', 'left_in_cell': '作業場に残った'}
HELD_MARK = '持ったまま'


def ledger_from_record(result: dict, events: list, cell) -> dict:
    g = cell.gantry
    errors = result['placement_errors']
    placed = set(errors) | set(result.get('placed_unscored', []))
    handed = {e['data']['case'] for e in events if e['type'] == 'HANDOVER' and 'area' in e['data']}
    interrupted = any(e['type'] == 'MISSION_INCOMPLETE' and e['data'].get('status') == 'interrupted' for e in events)
    holds = [e['data']['case'] for e in events if e['type'] == 'HANDOVER' and HELD_MARK in e['data'].get('reason', '')]
    held = {holds[-1]} if interrupted and holds else set()
    never = set(result.get('cases_never_detected', []))
    ledger = {}
    for k, p in result['final_positions'].items():
        quality, detail = None, ''
        if k in held:
            state, detail = 'held_by_robot', 'ケースを持ったまま人の対応を待つ'
        elif k in placed:
            state = 'placed_on_b'
            e = errors.get(k)
            quality = 'unscored' if e is None else ('ok' if e['xy_m'] <= POS_TOL and abs(e['z_m']) <= POS_TOL
                                                    and e['yaw_deg'] <= YAW_TOL_DEG and e['tilt_deg'] <= TILT_TOL_DEG else 'ng')
        elif k in handed:
            state = 'handed_over'
        elif not (g.x_range[0] <= p[0] <= g.x_range[1] and g.y_range[0] <= p[1] <= g.y_range[1]):
            state, detail = 'outside_cell', '作業場の外'
        else:
            state = 'left_in_cell'
            detail = result.get('remaining', {}).get(k) or ('未検出' if k in never else '未処理')
        ledger[k] = {'state': state, 'detail': detail, 'quality': quality,
                     'left_pallet': 'not_recorded' if state in ('outside_cell', 'left_in_cell') else None}
    return ledger


def check(ledger: dict, result: dict) -> list:
    """Disagreements between the rebuilt ledger and the run's own counts."""
    n = sum(1 for v in ledger.values() if v['state'] == 'handed_over')
    return [] if n == result['handed_over'] else [f'人に回した数が記録（{result["handed_over"]}）と一致しない（{n}）']


def main(dirs):
    cell = load_cell()
    for d in map(Path, dirs):
        result = json.loads((d / 'result.json').read_text(encoding='utf-8'))
        if 'case_ledger' in result:
            ledger = result['case_ledger']
        else:
            events = [json.loads(line) for line in (d / 'events.jsonl').read_text(encoding='utf-8').splitlines() if line]
            ledger = ledger_from_record(result, events, cell)
            (d / 'case_ledger.json').write_text(json.dumps(ledger, indent=1, ensure_ascii=False), encoding='utf-8')
        states = Counter(v['state'] for v in ledger.values())
        quality = Counter(v['quality'] for v in ledger.values() if v['quality'])
        left = Counter(v['detail'] for v in ledger.values() if v['state'] == 'left_in_cell')
        line = {'run': d.name, 'cases': len(ledger), **{LABELS[s]: states.get(s, 0) for s in CASE_STATES},
                '置いた品質': {'基準内': quality.get('ok', 0), '基準外': quality.get('ng', 0), '未判定': quality.get('unscored', 0)},
                '残った理由': dict(left), '食い違い': check(ledger, result)}
        print('LEDGER', json.dumps(line, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main(sys.argv[1:])
