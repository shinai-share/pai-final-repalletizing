"""scripts/rerender_trace.py reads a run's physics-step trace whole, in order, and refuses one cut short."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
rr = pytest.importorskip('rerender_trace')


def _write(folder, chunks, steps_in_index):
    folder.mkdir()
    t0 = 0.0
    for k, n in enumerate(chunks):
        t = t0 + np.arange(n) / 120
        t0 = t[-1] + 1 / 120
        np.savez_compressed(folder / f'trace_{k:05d}.npz', t=t, pose=np.zeros((n, 3, 7), np.float32),
                            tool_gantry_q=np.zeros((n, 6), np.float32), tool_fork_state=np.zeros((n, 4), np.float32))
    (folder / 'trace_index.json').write_text(json.dumps({'rows': ['case_00', 'case_01', 'wrist_r'],
                                                         'chunks': len(chunks), 'steps': steps_in_index}))


def test_the_chunks_are_joined_in_order(tmp_path):
    _write(tmp_path / 'trace', [5, 3], 8)
    idx, tr = rr.load_trace(tmp_path / 'trace')
    assert len(tr['t']) == 8 and np.all(np.diff(tr['t']) > 0)
    assert tr['pose'].shape == (8, 3, 7) and tr['tool_fork_state'].shape == (8, 4)


def test_a_trace_shorter_than_its_index_is_refused(tmp_path):
    _write(tmp_path / 'trace', [5, 3], 9)
    with pytest.raises(RuntimeError):
        rr.load_trace(tmp_path / 'trace')


def test_a_scenario_run_is_drawn_with_the_scenario_pile():
    from shinai_pi import scenarios
    from shinai_pi.config import load_cell
    base = load_cell()
    cell, sku, spawns = rr.scene(base, 3001, scenario=True, layers=None)
    want = scenarios.sample(base, 3001)
    assert cell.layout.layers == want.cell.layout.layers == 3
    assert sku.id == want.sku.id and len(spawns) == len(want.spawns)


def test_a_plain_run_keeps_the_layers_given():
    from shinai_pi.config import load_cell
    cell, sku, spawns = rr.scene(load_cell(), 118, scenario=False, layers=2, sku_id='BEV-500x24')
    assert cell.layout.layers == 2 and len(spawns) == 24


def test_a_plain_run_needs_its_layers():
    from shinai_pi.config import load_cell
    with pytest.raises(ValueError):
        rr.scene(load_cell(), 118, scenario=False, layers=None, sku_id='BEV-500x24')


def test_tines_count_as_out_only_when_both_are_past_the_rest():
    assert rr.tines_out([-0.08, -0.08, 0.0, 0.0]) is False
    assert rr.tines_out([0.2, -0.08, 0.0, 0.0]) is False
    assert rr.tines_out([0.2, 0.21, 0.1, 0.1]) is True


def test_a_small_grid_run_is_drawn_with_its_grid():
    from shinai_pi.config import load_cell
    cell, sku, spawns = rr.scene(load_cell(), 7, scenario=False, layers=None, sku_id='BEV-500x24', grid=(3, 1, 2))
    assert len(spawns) == 6 and cell.layout.layers == 2
