"""The table of the planned missions, built from their own records (result.json), and the comparison of the learned
predictor with the rule fixed before the runs (docs/generalisation_plan.md, addendum of 2026-10-31).

    python scripts/mission_table.py artifacts/missions_v3 --seeds 3001 3014 3000 3025
Writes <dir>/table.json and <dir>/table.md."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.stage_gate import judge  # noqa: E402

POLICIES = ('shinai_learned', 'shinai')
POLICY_NAMES = {'shinai_learned': '予測あり', 'shinai': '規則だけ'}
SPLIT_NAMES = {'seen': '学習と同じ種類', 'unseen': '学習に使っていない種類'}
VERDICTS = {'better': '規則より良くした', 'not_shown': '今回の比較では確認できなかった',
            'model_never_changed_a_choice': '予測は判断を変えなかった'}
STATUS_NAMES = {'unresolved': '人の確認を求めて終了', 'verified': '確認済みで終了'}


def row(r, split):
    led = list(r.get('case_ledger', {}).values())
    before = [v for v in led if v.get('left_pallet') == 'before_mission']
    reasons = judge('S3', r)
    return {
        'seed': r['seed'], 'policy': r['policy'], 'split': split, 'status': r['status'],
        's3_passed': not reasons, 's3_reasons': reasons,
        'placed_ok': sum(1 for v in led if v['state'] == 'placed_on_b' and v.get('quality') == 'ok'),
        'placed_ng': sum(1 for v in led if v['state'] == 'placed_on_b' and v.get('quality') != 'ok'),
        'left_in_cell': sum(1 for v in led if v['state'] == 'left_in_cell'),
        'fell_before_start': len(before),
        'fell_before_start_outside_cell': sum(1 for v in before if v['state'] == 'outside_cell'),
        'robot_drops': r.get('drops_in_transit', 0),
        'robot_knocked_off': sum(1 for v in led if str(v.get('left_pallet') or '').startswith('during_pick')),
        'secondary_collapses': r.get('secondary_collapses', 0), 'disturbed_cases': r.get('disturbed_cases', 0),
        'learned_decisions': r.get('learned_decisions', 0), 'learned_changed': r.get('learned_changed', 0),
        'sim_seconds_whole_task': r.get('sim_seconds'), 'wall_seconds_compute': r.get('wall_seconds'),
    }


def compare(rows, seeds):
    """The pre-registered reading: 'better' only when the learned side has fewer secondary collapses plus disturbed
    cases in at least one pair, more in none, and never fails S3 alone."""
    by = {(x['seed'], x['policy']): x for x in rows}
    not_run = [{'seed': s, 'policy': p} for s in seeds for p in POLICIES if (s, p) not in by]
    pairs = []
    for s in seeds:
        if (s, 'shinai_learned') in by and (s, 'shinai') in by:
            l, k = by[s, 'shinai_learned'], by[s, 'shinai']
            harm = lambda x: x['secondary_collapses'] + x['disturbed_cases']  # noqa: E731
            pairs.append({'seed': s, 'learned_harm': harm(l), 'rule_harm': harm(k),
                          'learned_only_failed_s3': (not l['s3_passed']) and k['s3_passed'],
                          'learned_changed': l['learned_changed'], 'learned_decisions': l['learned_decisions']})
    learned_runs = [x for x in rows if x['policy'] == 'shinai_learned']
    if learned_runs and all(x['learned_changed'] == 0 for x in learned_runs):
        verdict = 'model_never_changed_a_choice'
    elif (any(p['learned_harm'] < p['rule_harm'] for p in pairs) and not any(p['learned_harm'] > p['rule_harm'] for p in pairs)
          and not any(p['learned_only_failed_s3'] for p in pairs)):
        verdict = 'better'
    else:
        verdict = 'not_shown'
    return {'verdict': verdict, 'verdict_ja': VERDICTS[verdict], 'pairs': pairs, 'not_run': not_run}


def markdown(rows, cmp):
    head = ('| 番号 | 荷崩れの種類 | 方式 | S3 | B に置いた箱（基準内／基準外） | 運搬中に A から落ちていた箱（うち作業場の外） '
            '| 作業中の落下／作業中に A から落とした箱 | 二次崩れ／動いた箱 | 予測で選ぶ箱が変わった回数 '
            '| 作業全体の時間（シミュレーター内） | 計算にかかった実時間 |')
    out = [head, '|' + '---|' * 11]
    for x in rows:
        why = [w.replace(x['status'], STATUS_NAMES.get(x['status'], x['status'])) for w in x['s3_reasons']]
        s3 = '合格' if x['s3_passed'] else '不合格：' + '、'.join(why)
        changed = f"{x['learned_decisions']} 回中 {x['learned_changed']} 回" if x['policy'] == 'shinai_learned' else '－'
        out.append(f"| {x['seed']} | {SPLIT_NAMES.get(x['split'], x['split'])} | {POLICY_NAMES[x['policy']]} | {s3} "
                   f"| {x['placed_ok']}／{x['placed_ng']} | {x['fell_before_start']}（{x['fell_before_start_outside_cell']}） "
                   f"| {x['robot_drops']}／{x['robot_knocked_off']} | {x['secondary_collapses']}／{x['disturbed_cases']} "
                   f"| {changed} | {x['sim_seconds_whole_task']:,.0f} 秒 | {x['wall_seconds_compute'] / 3600:.1f} 時間 |")
    out += ['', f"予測と規則の比較（走らせる前に決めた読み方）：{cmp['verdict_ja']}"]
    out += [f"未実施：{n['seed']} {POLICY_NAMES[n['policy']]}" for n in cmp['not_run']]
    return '\n'.join(out) + '\n'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dir')
    ap.add_argument('--seeds', type=int, nargs='+', required=True)
    a = ap.parse_args()
    from shinai_pi import scenarios
    from shinai_pi.config import load_cell
    base = load_cell()
    d = Path(a.dir)
    rows = []
    for s in a.seeds:
        sp = scenarios.split(scenarios.sample(base, s).meta)
        for p in POLICIES:
            f = d / f'{p}_seed{s:03d}' / 'result.json'
            if f.exists():
                rows.append(row(json.loads(f.read_text(encoding='utf-8')), sp))
    cmp = compare(rows, a.seeds)
    (d / 'table.json').write_text(json.dumps({'rows': rows, 'comparison': cmp}, indent=1, ensure_ascii=False), encoding='utf-8')
    (d / 'table.md').write_text(markdown(rows, cmp), encoding='utf-8')
    print(markdown(rows, cmp))


if __name__ == '__main__':
    main()
