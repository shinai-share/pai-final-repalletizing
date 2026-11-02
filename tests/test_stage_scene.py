"""The leaning case of stage S1-tilt at other angles (a factor of the layer-B experiments): it must
still rest on the floor and on pallet A's edge, or the scene is not the one being varied."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import stage_trials as st  # noqa: E402
from shinai_pi.config import load_cell  # noqa: E402
from shinai_pi.perception.obb import quat_to_matrix  # noqa: E402


def corners(spawn, size):
    R = quat_to_matrix(spawn.quat)
    h = np.asarray(size) / 2
    return np.array([np.asarray(spawn.pos) + R @ (h * s) for s in
                     [(a, b, c) for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)]])


@pytest.mark.parametrize('tilt', [20.0, 30.0, 40.0])
def test_a_leaning_case_rests_on_the_floor_and_the_pallet_edge(tilt):
    cell = load_cell()
    sku = cell.sku('BEV-500x24')
    spawn = st.scene(cell, sku, 'S1-tilt', 0, tilt_deg=tilt)[0]
    R = quat_to_matrix(spawn.quat)
    assert math.degrees(math.acos(R[2, 2])) == pytest.approx(tilt, abs=1e-6)
    c = corners(spawn, sku.size)
    assert c[:, 2].min() == pytest.approx(0.001, abs=1e-6)                 # lower edge just above the floor
    edge_x = cell.pallet.a_center[0] + cell.pallet.size[0] / 2
    # the bottom face rests on the pallet's top edge
    n = R[:, 2]
    bottom = np.asarray(spawn.pos) - n * sku.size[2] / 2
    z_at_edge = bottom[2] - (n[0] * (edge_x - bottom[0])) / n[2]
    assert z_at_edge - cell.pallet.size[2] == pytest.approx(0.0, abs=1e-4)


def test_the_default_tilt_is_the_gate_scene():
    cell = load_cell()
    sku = cell.sku('BEV-500x24')
    assert st.scene(cell, sku, 'S1-tilt', 3) == st.scene(cell, sku, 'S1-tilt', 3, tilt_deg=30.0)


def test_a_tilt_angle_is_only_for_the_leaning_stage():
    cell = load_cell()
    with pytest.raises(ValueError):
        st.scene(cell, cell.sku('BEV-500x24'), 'S0', 0, tilt_deg=40.0)
