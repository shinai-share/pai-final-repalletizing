import json
import math
from pathlib import Path

import numpy as np
import pytest

from shinai_pi.evaluation.metrics import box_axis_error_deg
from shinai_pi.perception.boxfit import fit_all
from shinai_pi.perception.geometry import frame_from_npz, look_at_view
from shinai_pi.perception.obb import OBB, quat_to_matrix, yaw_matrix
from shinai_pi.perception.synthetic import render
from shinai_pi.perception.voxels import VoxelMap

DIMS = (0.365, 0.275, 0.22)
FRAMES = Path(__file__).resolve().parents[1] / 'artifacts' / 'frames'


def dims_for(key):
    return DIMS if key.startswith('case') else None


def axis_angle_error_deg(est, R_true):
    return box_axis_error_deg(est.obb.rotation, est.obb.size, R_true, DIMS)


def test_single_top_view_recovers_flat_box():
    truth = OBB.make((0.1, -0.05, 0.11), DIMS, yaw_matrix(0.35))
    f = render(look_at_view((0.12, -0.05, 2.5), (0.1, -0.05, 0)), {'case_a': truth}, width=400, height=300, fx=500, fy=500)
    est = fit_all([f], dims_for)['case_a']
    assert np.linalg.norm(est.obb.center - truth.center) < 0.01
    assert axis_angle_error_deg(est, truth.rotation) < 2.0
    assert sorted(est.obb.size) == pytest.approx(sorted(DIMS))
    assert est.top_visible > 0.9
    assert est.confidence > 0.85


def test_tilted_box_from_two_views():
    R = yaw_matrix(0.5) @ quat_to_matrix((math.cos(0.25), math.sin(0.25), 0, 0))   # 28.6 deg tilt
    truth = OBB.make((0, 0, 0.25), DIMS, R)
    frames = [render(look_at_view(e, (0, 0, 0.2)), {'case_t': truth}, width=320, height=240, fx=300, fy=300, name=str(i))
              for i, e in enumerate([(1.2, -1.0, 1.4), (-1.0, 1.1, 1.3)])]
    est = fit_all(frames, dims_for)['case_t']
    assert axis_angle_error_deg(est, truth.rotation) < 3.0
    assert np.linalg.norm(est.obb.center - truth.center) < 0.02


def test_occluded_box_reports_uncertainty_and_known_space_resolves_it():
    hidden = OBB.make((0, 0.30, 0.11), DIMS, np.eye(3))
    blocker = OBB.make((0, 0.0, 0.20), (0.6, 0.2, 0.40))
    scene = {'case_h': hidden, 'blocker': blocker}
    side = render(look_at_view((0, -1.5, 0.9), (0, 0.2, 0.1)), scene, width=320, height=240, fx=300, fy=300, name='side')
    one = fit_all([side], dims_for)
    if 'case_h' in one:
        assert one['case_h'].confidence < 0.8
    vm = VoxelMap((-1, -1, 0), (1, 1, 1), 0.03)
    views = [side] + [render(look_at_view(e, (0, 0.3, 0.1)), scene, width=320, height=240, fx=300, fy=300, name=str(i))
                      for i, e in enumerate([(0, 1.6, 1.2), (1.4, 0.3, 1.2)])]
    for v in views:
        vm.integrate(v, step=1)
    est = fit_all(views, dims_for, vm)['case_h']
    assert np.linalg.norm(est.obb.center - hidden.center) < 0.03
    assert est.confidence > 0.8


@pytest.mark.skipif(not (FRAMES / 'seed008').exists(), reason='recorded Isaac frames not present')
def test_isaac_multiview_fit_against_truth():
    d = FRAMES / 'seed008'
    truth = json.loads((d / 'truth.json').read_text())['cases']
    frames = [frame_from_npz(p, p.stem) for p in sorted(d.glob('*.npz'))]
    vm = VoxelMap((-2.1, -1.2, 0.0), (0.2, 1.2, 1.4), 0.04)
    for f in frames:
        vm.integrate(f, step=3)
    est = fit_all(frames, dims_for, vm)
    pos_err, ang_err, confident = [], [], []
    for key, e in est.items():
        t = truth[key]
        R_true = quat_to_matrix(t['quat_wxyz'])
        pe = float(np.linalg.norm(e.obb.center - np.array(t['pos'])))
        ae = axis_angle_error_deg(e, R_true)
        pos_err.append(pe); ang_err.append(ae)
        if e.confidence > 0.8:
            confident.append(pe)
    assert len(est) >= 30
    assert np.median(pos_err) < 0.02
    assert np.median(ang_err) < 3.0
    assert np.percentile(confident, 90) < 0.04     # confident estimates are trustworthy
