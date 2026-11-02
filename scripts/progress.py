"""One-line progress of a running mission (for watching long runs).

    python scripts/progress.py artifacts/runs/shinai_seed008"""
import json
import sys
from collections import Counter
from pathlib import Path


def main(run_dir):
    path = Path(run_dir) / 'events.jsonl'
    ev = [json.loads(l) for l in path.read_text(encoding='utf-8').splitlines() if l.strip().endswith('}')]
    if not ev:
        print('no events yet')
        return
    c = Counter(e['type'] for e in ev)
    bad = sum(1 for e in ev if e['type'] == 'PLACEMENT_CHECKED' and not e['data']['ok'])
    print(f"sim {ev[-1]['t']:.0f}s placed {c['PLACED']} handover {c['HANDOVER']} stops {c['SAFETY_STOP']} "
          f"drops {c['DROP_DETECTED']} rejected {c['MOVE_REJECTED']} bad_placements {bad} side_obs {c['OBSERVE_REQUESTED']}")


if __name__ == '__main__':
    main(sys.argv[1])
