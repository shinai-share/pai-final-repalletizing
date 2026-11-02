import math
from pathlib import Path

import numpy as np
import pytest

from shinai_pi.perception.geometry import frame_from_npz, look_at_view
from shinai_pi.perception.obb import OBB, matrix_to_quat, quat_to_matrix, yaw_matrix
from shinai_pi.perception.synthetic import render

FRAMES = Path(__file__).resolve().parents[1] / 'artifacts' / 'frames'


def test_quaternion_roundtrip():
    R = yaw_matrix(0.7) @ quat_to_matrix((math.cos(0.2), math.sin(0.2), 0, 0))
    assert np.allclose(quat_to_matrix(matrix_to_quat(R)), R)


def test_obb_contains_and_distance():
    b = OBB.make((0, 0, 0.5), (0.4, 0.2, 1.0), yaw_matrix(math.pi / 4))
    assert b.contains([[0, 0, 0.5]])[0]
    assert not b.contains([[0.5, 0, 0.5]])[0]
    assert b.signed_distance([[0, 0, 1.2]])[0] == pytest.approx(0.2)


def test_top_face_of_box_lying_on_its_side():
    R = quat_to_matrix((math.cos(math.pi / 4), math.sin(math.pi / 4), 0, 0))   # 90 deg about x
    b = OBB.make((0, 0, 0.15), (0.4, 0.3, 0.2), R)
    c, n, half, _ = b.top_face()
    assert np.allclose(n, [0, 0, 1], atol=1e-9)
    assert c[2] == pytest.approx(0.30)
    assert sorted(half) == pytest.approx([0.1, 0.2])


def test_synthetic_render_unprojects_onto_box_surface():
    box = OBB.make((0, 0, 0.11), (0.365, 0.275, 0.22), yaw_matrix(0.3))
    f = render(look_at_view((1.0, -1.2, 1.5), (0, 0, 0.1)), {'case_00': box})
    pts, ids, _ = f.unproject()
    on_box = pts[ids == 2]
    assert len(on_box) > 300
    assert np.all(np.abs(box.signed_distance(on_box)) < 1e-6)
    floor = pts[ids == 1]
    assert np.allclose(floor[:, 2], 0, atol=1e-6)


@pytest.mark.skipif(not (FRAMES / 'seed008' / 'overhead_a.npz').exists(), reason='recorded Isaac frames not present')
def test_isaac_frame_points_lie_on_true_case_surfaces():
    import json
    truth = json.loads((FRAMES / 'seed008' / 'truth.json').read_text())['cases']
    f = frame_from_npz(FRAMES / 'seed008' / 'overhead_a.npz')
    pts, ids, _ = f.unproject(step=2)
    errs = []
    for iid, key in f.instance_map.items():
        if not key.startswith('case_'):
            continue
        p = pts[ids == iid]
        if len(p) < 30:
            continue
        t = truth[key]
        box = OBB.make(t['pos'], (0.365, 0.275, 0.22), quat_to_matrix(t['quat_wxyz']))
        errs.append(np.median(np.abs(box.signed_distance(p))))
    assert len(errs) >= 10
    assert np.median(errs) < 0.01      # sensor points sit on the true surfaces within 1 cm
