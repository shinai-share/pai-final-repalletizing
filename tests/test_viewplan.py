import numpy as np

from shinai_pi.config import load_cell
from shinai_pi.perception.geometry import look_at_view
from shinai_pi.perception.voxels import VoxelMap
from shinai_pi.pile import Box, PileBelief
from shinai_pi.viewplan import rank_viewpoints, tool_camera_world, visible
from shinai_pi.viewpoints import side_viewpoints

CELL = load_cell()
A = CELL.pallet.a_center


def test_tool_camera_looks_forward_and_down_along_tool_y():
    pos, R = tool_camera_world(CELL, np.array([0.0, 0.0, 0.0, 0.0]))
    forward = -R[:, 2]
    assert forward[1] > 0.5 and forward[2] < -0.5
    pos2, R2 = tool_camera_world(CELL, np.array([0.0, 0.0, 0.0, np.pi]))
    assert (-R2[:, 2])[1] < -0.5


def test_side_viewpoints_see_the_pile_centre():
    belief = PileBelief()
    centre = np.array([[A[0], A[1], 0.6]])
    for vp in side_viewpoints(CELL, A):
        assert visible(CELL, np.array(vp.q), centre, belief)[0], vp.name


def test_points_behind_a_case_are_not_visible():
    belief = PileBelief()
    belief.observe([Box('wall', (A[0], A[1] - 0.3, 0.6), (1.0, 0.1, 1.2))], 'x')
    behind = np.array([[A[0], A[1] + 0.2, 0.6]])
    south = [v for v in side_viewpoints(CELL, A) if v.name == 'south'][0]
    north = [v for v in side_viewpoints(CELL, A) if v.name == 'north'][0]
    assert not visible(CELL, np.array(south.q), behind, belief)[0]
    assert visible(CELL, np.array(north.q), behind, belief)[0]


def test_ranking_prefers_the_view_that_resolves_the_decision():
    # A low-confidence case leans against the north side; unknown space only on the north side.
    belief = PileBelief(floor=CELL.pallet.size[2])
    belief.observe([Box('top', (A[0], A[1], 0.144 + 0.11), (0.365, 0.275, 0.22), confidence=0.5)], 'overhead')
    vm = VoxelMap((A[0] - 1, A[1] - 1, 0), (A[0] + 1, A[1] + 1, 1.2), 0.05)
    vm.free_n[:] = 1
    north_strip = (slice(None), slice(int(1.15 / 0.05), int(1.45 / 0.05)), slice(0, 8))   # down to the floor: cases do not float
    vm.free_n[north_strip] = 0
    belief.set_known_space(vm, ((A[0] - 0.8, A[1] - 0.8, 0.0), (A[0] + 0.8, A[1] + 0.8, 1.0)))
    ranked = rank_viewpoints(CELL, belief, side_viewpoints(CELL, A), np.zeros(4))
    assert ranked[0].info > 0
    assert ranked[0].viewpoint.name in ('north', 'east', 'west')
    by_name = {r.viewpoint.name: r for r in ranked}
    assert by_name['north'].info >= by_name['south'].info


def test_tool_camera_model_matches_isaac_recordings():
    from pathlib import Path
    frames = sorted((Path(__file__).resolve().parents[1] / 'artifacts' / 'frames').glob('seed*/tool_*.npz'))
    if not frames:
        import pytest
        pytest.skip('recorded Isaac frames not present')
    from shinai_pi.perception.geometry import frame_from_npz
    for path in frames[:4]:
        f = frame_from_npz(path)
        q = np.load(path)['gantry']
        pos, R = tool_camera_world(CELL, q)
        c2w = f.cam_to_world
        assert np.allclose(pos, c2w[3, :3], atol=0.01), path.name
        assert np.allclose(R.T, c2w[:3, :3], atol=0.02), path.name
