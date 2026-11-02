"""Where the time goes: split a run's simulated time into phases from its event log.

    python scripts/cycle_breakdown.py shinai_seed008"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# The phase that *starts* at each event type and lasts until the next event.
PHASE = {
    'MISSION_ACCEPTED': '最初の観測', 'OBSERVE_REQUESTED': '横から見に行く', 'OBSERVED': '判断',
    'BELIEF_UPDATED': '判断', 'PICK_PLANNED': '近づいて吸着', 'GRIP_FAILED': '吸着のやり直し', 'GRIP_OK': '持ち上げ',
    'LIFTED': '運んで置く', 'PLACED': '戻って観測', 'HANDOVER': '戻って観測', 'PILE_CHANGED': '置き場の確認',
    'PLACEMENT_CHECKED': '判断', 'SAFETY_STOP': '安全停止', 'SAFETY_CLEARED': '再開', 'MOVE_REJECTED': '置き場を探す',
    'DROP_DETECTED': '安全停止', 'DECISION_DEFERRED': '判断',
}


def main(run):
    ev = [json.loads(l) for l in (ROOT / 'artifacts' / 'runs' / run / 'events.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
    total = {}
    for a, b in zip(ev, ev[1:]):
        p = PHASE.get(a['type'], 'その他')
        total[p] = total.get(p, 0.0) + (b['t'] - a['t'])
    span = ev[-1]['t'] - ev[0]['t']
    picks = sum(1 for e in ev if e['type'] in ('PLACED',)) + sum(1 for e in ev if e['type'] == 'HANDOVER' and e['data'].get('area'))
    print(f'合計 {span:.1f} 秒 / 移したケース {picks} 個 / 1 個あたり {span / max(1, picks):.1f} 秒')
    for p, s in sorted(total.items(), key=lambda x: -x[1]):
        print(f'{p:10s} {s:7.1f} 秒  {100 * s / span:5.1f}%')


if __name__ == '__main__':
    main(sys.argv[1])
