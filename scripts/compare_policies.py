"""Aggregate mission results by policy over the same seeds (paired comparison).

Reads artifacts/runs/<policy>_seed<NNN>/result.json, writes artifacts/comparison.json and a
Markdown table. Confidence intervals: 95% bootstrap over seeds (2,000 resamples, fixed RNG)."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'artifacts' / 'comparison_runs'
METRICS = [
    ('secondary_collapses', '二次崩れ（25 cm 以上落ちたケース）'),
    ('disturbed_cases', '巻き込み（他のケースが 5 cm 以上動いた）'),
    ('drops_in_transit', '搬送中の落下'),
    ('placed_on_b', 'B に積んだケース'),
    ('handed_over', '人に渡したケース'),
    ('left_on_a', 'A に残ったケース'),
    ('sim_seconds', '所要時間（秒・シミュレーション）'),
    ('side_observations', '横からの観測の回数'),
    ('safety_stops', '安全停止'),
]


def load():
    by = {}
    for f in RUNS.glob('*_seed*/result.json'):
        r = json.loads(f.read_text(encoding='utf-8'))
        by.setdefault(r['policy'], {})[r['seed']] = r
    return by


def boot_ci(values, rng, n=2000):
    v = np.asarray(values, float)
    if len(v) < 2:
        return (float(v.mean()), float(v.mean()))
    means = rng.choice(v, size=(n, len(v)), replace=True).mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    by = load()
    if not by:
        sys.exit('no results')
    seeds = sorted(set.intersection(*(set(v) for v in by.values())))
    rng = np.random.default_rng(0)
    table = {'seeds': seeds, 'policies': {}}
    for pol, runs in sorted(by.items()):
        row = {}
        for key, _ in METRICS:
            vals = [runs[s][key] for s in seeds]
            lo, hi = boot_ci(vals, rng)
            row[key] = {'mean': round(float(np.mean(vals)), 2), 'ci95': [round(lo, 2), round(hi, 2)], 'values': vals}
        row['completed'] = sum(1 for s in seeds if str(runs[s]['status']).startswith('verified'))
        pred = [runs[s]['prediction'] for s in seeds]
        row['prediction'] = {k: int(sum(p[k] for p in pred)) for k in ('true_positive', 'missed', 'false_alarm')}
        table['policies'][pol] = row
    (RUNS / 'comparison.json').write_text(json.dumps(table, indent=1, ensure_ascii=False), encoding='utf-8')
    pols = sorted(table['policies'])
    lines = [f'同じ荷崩れ {len(seeds)} 通り（seed {", ".join(map(str, seeds))}）での平均。括弧は 95% 信頼区間（seed のブートストラップ）。', '',
             '| 指標 | ' + ' | '.join(pols) + ' |', '|---|' + '---|' * len(pols)]
    for key, name in METRICS:
        cells = [f"{table['policies'][p][key]['mean']}（{table['policies'][p][key]['ci95'][0]}〜{table['policies'][p][key]['ci95'][1]}）" for p in pols]
        lines.append(f'| {name} | ' + ' | '.join(cells) + ' |')
    lines.append('| 完了した回数 | ' + ' | '.join(str(table['policies'][p]['completed']) for p in pols) + ' |')
    md = '\n'.join(lines)
    (RUNS / 'comparison.md').write_text(md + '\n', encoding='utf-8')
    print(md)


if __name__ == '__main__':
    main()
