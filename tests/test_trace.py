import json

import numpy as np
import pytest

from shinai_pi.evaluation.trace import StepTrace, load_trace

ROWS, COLS = ['case_00', 'case_01', 'wrist_r'], ['case_00', 'case_01', 'wrist_r', 'pallet_b']


def _step(tr, t, f01=0.0, fb=0.0, detail=None, grip=1.0):
    pose = np.zeros((3, 7)); pose[:, 6] = 1.0; pose[0, 2] = t
    vel = np.zeros((3, 6)); vel[0, 2] = -1.0
    forces = np.zeros((3, 4, 3))
    forces[0, 1] = [0.0, 0.0, f01]
    forces[1, 0] = [0.0, 0.0, -f01]
    forces[0, 3] = [0.3, 0.0, fb]
    tr.step(t, pose, vel, forces, detail=detail, tool={'z_load': 2.0 * t, 'gripped': grip, 'fork_e': [t, -t]})


def test_every_step_keeps_the_bodies_motion_the_tool_state_and_each_touching_pair_as_a_vector(tmp_path):
    tr = StepTrace(tmp_path / 'trace', ROWS, COLS, flush_every=2)      # the folder is made when missing
    _step(tr, 0.1, f01=5.0)
    _step(tr, 0.2, fb=120.0)
    _step(tr, 0.3)
    tr.close()
    d = load_trace(tmp_path / 'trace')
    assert np.allclose(d['t'], [0.1, 0.2, 0.3])
    assert d['pose'].shape == (3, 3, 7) and np.allclose(d['pose'][:, 0, 2], [0.1, 0.2, 0.3])
    assert np.allclose(d['vel'][:, 0, 2], -1.0)
    assert np.allclose(d['tool']['z_load'], [0.2, 0.4, 0.6]) and d['tool']['fork_e'].shape == (3, 2)
    pairs = list(zip(d['pair_step'], d['pair_row'], d['pair_col']))
    # case_00/case_01 in both directions at the first step, case_00 on pallet_b at the second, nothing at the third
    assert pairs == [(0, 0, 1), (0, 1, 0), (1, 0, 3)]
    assert np.allclose(d['pair_F'][2], [0.3, 0.0, 120.0])          # the direction is kept, not only the size
    idx = json.loads((tmp_path / 'trace' / 'trace_index.json').read_text(encoding='utf-8'))
    assert idx['rows'] == ROWS and idx['cols'] == COLS and idx['steps'] == 3


def test_a_pair_just_under_the_threshold_is_not_kept_and_one_at_it_is(tmp_path):
    tr = StepTrace(tmp_path / 'tr', ROWS, COLS, on_N=1.0)
    _step(tr, 0.1, f01=0.99)
    _step(tr, 0.2, f01=1.0)
    tr.close()
    d = load_trace(tmp_path / 'tr')
    assert set(d['pair_step'].tolist()) == {1}


def test_contact_points_are_kept_per_pair_with_their_normal_force_and_friction(tmp_path):
    tr = StepTrace(tmp_path / 'tr', ROWS, COLS)
    counts = np.zeros((3, 4), int); starts = np.zeros((3, 4), int)
    counts[0, 3], starts[0, 3] = 2, 1                           # case_00 on pallet_b: points 1 and 2 of the buffer
    fn = np.array([9.0, 40.0, 80.0])
    pts = np.array([[9, 9, 9], [0.1, 0.2, 0.0], [0.3, 0.2, 0.0]], float)
    nrm = np.array([[0, 0, 1]] * 3, float)
    ft = np.array([[0, 0, 0], [1.0, 0, 0], [2.0, 0, 0]], float)
    _step(tr, 0.1, fb=120.0, detail=(fn, pts, nrm, counts, starts, ft))
    tr.close()
    d = load_trace(tmp_path / 'tr')
    assert d['pt_row'].tolist() == [0, 0] and d['pt_col'].tolist() == [3, 3]
    assert np.allclose(d['pt_fn'], [40.0, 80.0]) and np.allclose(d['pt_p'][1], [0.3, 0.2, 0.0])
    assert np.allclose(d['pt_ft'][:, 0], [1.0, 2.0])


def test_closing_twice_is_harmless_and_a_step_after_closing_is_refused(tmp_path):
    tr = StepTrace(tmp_path / 'tr', ROWS, COLS)
    _step(tr, 0.1)
    tr.close()
    tr.close()
    with pytest.raises(RuntimeError):
        _step(tr, 0.2)
    assert len(load_trace(tmp_path / 'tr')['t']) == 1


def test_a_trace_with_no_steps_loads_as_empty(tmp_path):
    StepTrace(tmp_path / 'tr', ROWS, COLS).close()
    d = load_trace(tmp_path / 'tr')
    assert len(d['t']) == 0 and len(d['pair_step']) == 0


def test_steps_in_later_files_keep_their_place_in_the_run(tmp_path):
    tr = StepTrace(tmp_path / 'tr', ROWS, COLS, flush_every=2)
    for k in range(5):
        _step(tr, 0.1 * (k + 1), f01=5.0 if k == 3 else 0.0)
    tr.close()
    d = load_trace(tmp_path / 'tr')
    assert len(list((tmp_path / 'tr').glob('trace_*.npz'))) == 3
    assert set(d['pair_step'].tolist()) == {3} and np.isclose(d['t'][3], 0.4)


def test_the_index_keeps_what_the_run_says_about_the_bodies(tmp_path):
    StepTrace(tmp_path / 'tr', ROWS, COLS, meta={'mass_kg': {'case_00': 12.0}, 'dt': 1 / 120}).close()
    idx = json.loads((tmp_path / 'tr' / 'trace_index.json').read_text(encoding='utf-8'))
    assert idx['mass_kg'] == {'case_00': 12.0} and idx['dt'] == 1 / 120
