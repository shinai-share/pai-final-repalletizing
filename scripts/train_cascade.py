"""Train the cascade predictor on physics outcomes and compare it with the geometric rule on
held-out piles (different transport seeds, friction and mass).

    python scripts/train_cascade.py [--data artifacts/cascade_v2 --model cascade_v2]
Writes models/<model>.json and <data>/eval.json."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shinai_pi.learning.cascade import FEATURES, auc, brier, calibration_bins, fit  # noqa: E402


def load(data, name):
    """All <name>*.npz in the folder (data is generated in chunks of seeds, one process each)."""
    Xs, ys = [], []
    for f in sorted(data.glob(f'{name}*.npz')):
        d = np.load(f, allow_pickle=False)
        if tuple(d['features']) != FEATURES:
            sys.exit(f'feature set changed in {f.name}; regenerate the data')
        Xs.append(d['X'])
        ys.append(d['y'])
    if not Xs:
        sys.exit(f'no {name}*.npz in {data}')
    return np.concatenate(Xs), np.concatenate(ys)


def compare(model, Xte, yte):
    """The learned predictor against the geometric rule on one held-out set."""
    p_model = model.predict(Xte)
    geo = Xte[:, FEATURES.index('geo_falls')]
    sev = Xte[:, FEATURES.index('geo_severity')]
    return {
        'test_samples': int(len(yte)), 'test_positives': int(yte.sum()),
        'geometric_rule': {'auc': auc(geo + 0.01 * sev, yte), 'brier': brier(geo, yte),
                           'recall': float(geo[yte == 1].mean()) if yte.sum() else None,
                           'false_alarm_rate': float(geo[yte == 0].mean())},
        'learned': {'auc': auc(p_model, yte), 'brier': brier(p_model, yte), 'calibration': calibration_bins(p_model, yte)},
    }


def main(data, name, source):
    Xtr, ytr = load(data, 'train')
    model = fit(Xtr, ytr, {'train_samples': int(len(ytr)), 'train_positives': int(ytr.sum()), 'data': data.name,
                           'source': source})
    held_out = [n for n in ('test', 'seen', 'unseen') if any(data.glob(f'{n}*.npz'))]
    report = {n: compare(model, *load(data, n)) for n in held_out}
    report['weights'] = dict(zip(FEATURES, np.round(model.weights, 3).tolist()))
    (ROOT / 'models').mkdir(exist_ok=True)
    (ROOT / 'models' / f'{name}.json').write_text(model.to_json(), encoding='utf-8')
    (data / 'eval.json').write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='artifacts/cascade', help='folder holding train.npz and test.npz')
    ap.add_argument('--model', default='cascade_v1', help='name of the model file under models/')
    ap.add_argument('--source', default='Isaac Sim 6.2 physics', help='where the data came from, kept in the model file')
    a = ap.parse_args()
    main(ROOT / a.data, a.model, a.source)
