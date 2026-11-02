import numpy as np

from shinai_pi.perception.geometry import look_at_view
from shinai_pi.perception.obb import OBB
from shinai_pi.perception.synthetic import render
from shinai_pi.perception.voxels import FREE, OCCUPIED, UNKNOWN, VoxelMap

BOX = OBB.make((0, 0, 0.25), (0.4, 0.4, 0.5))


def fresh():
    return VoxelMap(lo=(-1, -1, 0.0), hi=(1, 1, 1.0), res=0.05)


def test_everything_is_unknown_before_looking():
    m = fresh()
    assert np.all(m.state() == UNKNOWN)
    assert m.unknown_fraction(lo=(-1, -1, 0), hi=(1, 1, 1)) == 1.0


def test_space_in_front_is_free_surface_occupied_and_behind_unknown():
    m = fresh()
    m.integrate(render(look_at_view((0, -2.0, 0.6), (0, 0, 0.3)), {'b': BOX}, width=200, height=150, fx=160, fy=160), step=1)
    assert m.state_at([[0, -0.6, 0.3]])[0] == FREE
    assert m.state_at([[0, -0.2, 0.25]])[0] == OCCUPIED
    assert m.state_at([[0, 0.5, 0.25]])[0] == UNKNOWN          # hidden behind the box


def test_second_view_reveals_the_hidden_side():
    m = fresh()
    m.integrate(render(look_at_view((0, -2.0, 0.6), (0, 0, 0.3)), {'b': BOX}, width=200, height=150, fx=160, fy=160))
    before = m.unknown_fraction(lo=(-0.3, 0.25, 0.05), hi=(0.3, 0.8, 0.45))
    m.integrate(render(look_at_view((0, 2.0, 0.6), (0, 0, 0.3)), {'b': BOX}, width=200, height=150, fx=160, fy=160))
    after = m.unknown_fraction(lo=(-0.3, 0.25, 0.05), hi=(0.3, 0.8, 0.45))
    assert after < before * 0.3


def test_interior_of_box_stays_unknown_and_is_not_called_free():
    m = fresh()
    for eye in [(0, -2, 0.6), (0, 2, 0.6), (2, 0, 0.6), (-2, 0, 0.6), (0.01, 0, 3)]:
        m.integrate(render(look_at_view(eye, (0, 0, 0.3)), {'b': BOX}, width=200, height=150, fx=160, fy=160))
    assert m.state_at([[0, 0, 0.25]])[0] != FREE


def test_only_the_floor_is_occupied_in_an_empty_scene():
    m = VoxelMap(lo=(-1, -1, -0.1), hi=(1, 1, 1.0), res=0.05)     # floor surface inside the grid
    m.integrate(render(look_at_view((0, -2.0, 0.6), (0, 0, 0.3)), {}, width=120, height=90, fx=100, fy=100))
    occupied_z = m.centers()[m.state() == OCCUPIED][:, 2]
    assert len(occupied_z) > 0
    assert np.all(occupied_z < 0.06)
