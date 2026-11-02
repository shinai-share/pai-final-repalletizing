"""Training is judged on each held-out set apart: the conditions of training on other seeds ('seen') and the
conditions kept out of it ('unseen'), so adapting to new collapses is measured, not assumed."""
import importlib.util
import json
from pathlib import Path

import numpy as np

from shinai_pi.learning.cascade import FEATURES

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('train_cascade', ROOT / 'scripts' / 'train_cascade.py')
tc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tc)


def _write(folder, name, n, seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, len(FEATURES)))
    y = (X[:, 0] + 0.3 * rng.normal(size=n) > 1.0).astype(int)
    np.savez_compressed(folder / f'{name}.npz', X=X, y=y, features=np.array(FEATURES))


def test_each_held_out_set_is_reported_apart(tmp_path, monkeypatch):
    monkeypatch.setattr(tc, 'ROOT', tmp_path)
    data = tmp_path / 'data'
    data.mkdir()
    _write(data, 'train_a', 400, 1)
    _write(data, 'seen_a', 200, 2)
    _write(data, 'unseen_a', 200, 3)
    tc.main(data, 'm', 'synthetic')
    report = json.loads((data / 'eval.json').read_text(encoding='utf-8'))
    assert {'seen', 'unseen'} <= report.keys() and 'test' not in report
    for k in ('seen', 'unseen'):
        assert report[k]['test_samples'] == 200 and 'auc' in report[k]['learned']
    assert json.loads((tmp_path / 'models' / 'm.json').read_text(encoding='utf-8'))
