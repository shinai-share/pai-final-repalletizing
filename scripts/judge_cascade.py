"""Judge a learned cascade predictor by the criteria fixed before any data (docs/generalisation_plan.md): on 'seen' and
on 'unseen' each, the learned AUC is at least 0.10 above the geometric rule's and its Brier score no higher.

    python scripts/judge_cascade.py artifacts/cascade_v3/eval.json
Writes <eval folder>/verdict.json and exits 1 when the predictor fails."""
import json
import sys
from pathlib import Path

AUC_MARGIN = 0.10
SETS = ('seen', 'unseen')


def judge(report: dict) -> dict:
    missing = [k for k in SETS if k not in report]
    if missing:
        raise ValueError(f'held-out sets missing: {missing}')
    sets = {}
    for k in SETS:
        learned, rule = report[k]['learned'], report[k]['geometric_rule']
        auc_ok = learned['auc'] >= rule['auc'] + AUC_MARGIN - 1e-12
        brier_ok = learned['brier'] <= rule['brier'] + 1e-12
        sets[k] = {'passed': auc_ok and brier_ok, 'auc_learned': learned['auc'], 'auc_rule': rule['auc'],
                   'brier_learned': learned['brier'], 'brier_rule': rule['brier']}
    return {'passed': all(v['passed'] for v in sets.values()), 'criteria': {'auc_margin': AUC_MARGIN, 'brier': 'no higher'},
            'sets': sets}


if __name__ == '__main__':
    path = Path(sys.argv[1])
    verdict = judge(json.loads(path.read_text(encoding='utf-8')))
    (path.parent / 'verdict.json').write_text(json.dumps(verdict, indent=1), encoding='utf-8')
    print(json.dumps(verdict))
    sys.exit(0 if verdict['passed'] else 1)
