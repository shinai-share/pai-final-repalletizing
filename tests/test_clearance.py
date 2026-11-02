"""How far the cups can still go along their approach before they meet something that is not the
case they are reaching for, from the downward camera at the plate centre."""
import math

import numpy as np

from shinai_pi.config import load_cell
from shinai_pi.perception.clearance import path_clearance
from shinai_pi.perception.geometry import look_at_view
from shinai_pi.perception.obb import OBB
from shinai_pi.perception.synthetic import render

CELL = load_cell()
TOOL = np.array([0.0, 0.0, 1.0])            # cup face centre, wrist level, yaw 0
HALF = (CELL.gantry.tool_size[0] / 2, CELL.gantry.tool_size[1] / 2)


def down_frame(objects):
    view = look_at_view(TOOL - np.array([0, 0, 0.003]), TOOL - np.array([0, 0, 1.0]), up=(0, 1, 0))
    w, h = 160, 120
    f = 6.0 / 20.955 * w
    return render(view, objects, width=w, height=h, fx=f, fy=f)


def test_only_the_target_below_leaves_the_path_clear():
    case = OBB.make((0.0, 0.0, 1.0 - 0.06 - 0.11), (0.36, 0.27, 0.22))
    c = path_clearance(down_frame({'case_00': case}), TOOL, np.eye(3), HALF, ignore={'case_00'})
    assert c.blocked_at is None and c.valid


def test_something_between_the_cups_and_the_target_is_found_at_its_depth():
    case = OBB.make((0.0, 0.0, 1.0 - 0.06 - 0.11), (0.36, 0.27, 0.22))
    bar = OBB.make((0.0, 0.0, 1.0 - 0.045), (0.40, 0.03, 0.02))
    c = path_clearance(down_frame({'case_00': case, 'bar': bar}), TOOL, np.eye(3), HALF, ignore={'case_00'})
    assert c.blocked_at is not None and abs(c.blocked_at - 0.035) < 0.004
    assert c.what == 'bar'


def test_something_beside_the_plate_footprint_is_not_in_the_path():
    case = OBB.make((0.0, 0.0, 1.0 - 0.06 - 0.11), (0.36, 0.27, 0.22))
    side = OBB.make((0.30, 0.0, 1.0 - 0.03), (0.10, 0.10, 0.02))      # beside, outside the margin
    c = path_clearance(down_frame({'case_00': case, 'side': side}), TOOL, np.eye(3), HALF, ignore={'case_00'})
    assert c.blocked_at is None


def test_the_stop_margin_grows_with_speed_and_delay():
    from shinai_pi.hand import stop_margin
    slow = stop_margin(0.05, latency=0.15, decel=2.0)
    fast = stop_margin(0.4, latency=0.15, decel=2.0)
    assert 0.015 < slow < 0.02 and fast > 0.09


def test_what_shows_beyond_the_face_being_reached_is_not_in_the_path():
    # A case turned a few degrees under the plate leaves the pallet visible past its corners, inside
    # the footprint's margin. The cups stop at the face: what lies beyond it cannot be struck.
    from shinai_pi.perception.obb import yaw_matrix
    face_z = 1.0 - 0.06
    case = OBB.make((0.0, 0.0, face_z - 0.11), (0.36, 0.27, 0.22), yaw_matrix(math.radians(8)))
    pallet = OBB.make((0.0, 0.0, face_z - 0.22 - 0.072), (1.1, 1.1, 0.144))
    c = path_clearance(down_frame({'case_00': case, 'pallet_a': pallet}), TOOL, np.eye(3), HALF, ignore={'case_00'},
                       reach=0.06 + 0.01)
    assert c.blocked_at is None


# --- the under-supports stand out past the plate on its -y side, above the cup face (S3 102, 2026-10-18:
# lowered beside pallet B, the tines pushed B's second layer while the plate's footprint was clear)

TINES = (-0.15, 0.15, -0.40, -0.17, 0.07)                 # x0, x1, y0, y1, bottom above the cup face


def test_a_stack_under_the_tines_blocks_the_descent_though_the_plate_footprint_is_clear():
    case = OBB.make((0.0, 0.0, 1.0 - 0.70 - 0.11), (0.36, 0.27, 0.22))       # 0.70 below: checked before descending
    stack = OBB.make((0.0, -0.36, 1.0 - 0.40 - 0.22), (0.36, 0.27, 0.44))    # its top 0.40 below the cup face
    plate_only = path_clearance(down_frame({'case_00': case, 'stack': stack}), TOOL, np.eye(3), HALF, ignore={'case_00'},
                                reach=0.73)
    with_tines = path_clearance(down_frame({'case_00': case, 'stack': stack}), TOOL, np.eye(3), HALF, ignore={'case_00'},
                                reach=0.73, regions=[TINES])
    assert plate_only.blocked_at is None
    assert with_tines.what == 'stack' and abs(with_tines.blocked_at - (0.40 + 0.07)) < 0.01


def test_the_same_stack_on_the_side_without_tines_leaves_the_descent_clear():
    case = OBB.make((0.0, 0.0, 1.0 - 0.70 - 0.11), (0.36, 0.27, 0.22))
    stack = OBB.make((0.0, 0.36, 1.0 - 0.40 - 0.22), (0.36, 0.27, 0.44))
    c = path_clearance(down_frame({'case_00': case, 'stack': stack}), TOOL, np.eye(3), HALF, ignore={'case_00'},
                       reach=0.73, regions=[TINES])
    assert c.blocked_at is None


def test_something_under_the_tines_lower_than_they_will_go_is_not_in_the_way():
    case = OBB.make((0.0, 0.0, 1.0 - 0.70 - 0.11), (0.36, 0.27, 0.22))
    low = OBB.make((0.0, -0.30, 1.0 - 0.73 + 0.07 - 0.01 - 0.01), (0.36, 0.10, 0.02))   # 1 cm under the tines' lowest (reach 0.73)
    c = path_clearance(down_frame({'case_00': case, 'low': low}), TOOL, np.eye(3), HALF, ignore={'case_00'},
                       reach=0.73, regions=[TINES])
    assert c.blocked_at is None


def test_a_tilted_tool_lowered_straight_down_sweeps_straight_down():
    # S3 103 follow-up (2026-10-18): the wrist is tilted to the face up high and the tool then goes straight
    # down. Judged along the tilted tool axis instead, a post 1.3 m below slid 0.45 m out of the tines' box.
    from shinai_pi.perception.obb import quat_to_matrix
    roll = math.radians(20)
    R = quat_to_matrix((math.cos(roll / 2), math.sin(roll / 2), 0.0, 0.0))
    case = OBB.make((0.0, 0.0, 1.0 - 0.70 - 0.11), (0.36, 0.27, 0.22))
    post = OBB.make((0.0, -0.30, 1.0 - 0.40 - 0.25), (0.30, 0.10, 0.50))
    f = down_frame({'case_00': case, 'post': post})
    straight = path_clearance(f, TOOL, R, HALF, ignore={'case_00'}, reach=0.73, regions=[TINES], direction=(0.0, 0.0, -1.0))
    assert straight.what == 'post'


def test_the_default_sweep_is_along_the_cups_axis():
    case = OBB.make((0.0, 0.0, 1.0 - 0.70 - 0.11), (0.36, 0.27, 0.22))
    stack = OBB.make((0.0, -0.36, 1.0 - 0.40 - 0.22), (0.36, 0.27, 0.44))
    f = down_frame({'case_00': case, 'stack': stack})
    a = path_clearance(f, TOOL, np.eye(3), HALF, ignore={'case_00'}, reach=0.73, regions=[TINES])
    b = path_clearance(f, TOOL, np.eye(3), HALF, ignore={'case_00'}, reach=0.73, regions=[TINES], direction=(0.0, 0.0, -1.0))
    assert a.what == b.what == 'stack' and abs(a.blocked_at - b.blocked_at) < 1e-9
