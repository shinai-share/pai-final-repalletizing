"""The suction + under-support hand: where the tines go, when they fit, when the transfer of the
weight to them is accepted, and what the cups are left to carry (requirements v05, drawings v01)."""
import math

import numpy as np
import pytest

from shinai_pi.config import load_cell
from shinai_pi import hand

CELL = load_cell()
F = CELL.fork
SKU = CELL.skus[0]
DEPTH = sorted(SKU.size, reverse=True)[1]          # held top down, long edge along tool x: 0.27 across tool y
W = 12.0 * 9.81


def test_tines_go_below_the_bottom_by_the_clearance_and_by_what_a_tilt_lowers_the_far_edge():
    level = hand.tine_extension(0.223, 0.0, DEPTH, F)
    assert level == pytest.approx(0.223 + F.clearance)
    tilted = hand.tine_extension(0.223, math.radians(4.0), DEPTH, F)
    assert tilted == pytest.approx(0.223 + F.clearance + DEPTH / 2 * math.tan(math.radians(4.0)))


def test_a_tine_extension_beyond_the_fork_s_reach_is_refused():
    with pytest.raises(ValueError):
        hand.tine_extension(F.extension_range[1], 0.0, DEPTH, F)


def test_the_fork_fits_a_case_held_near_the_centre():
    ok, why = hand.fork_fits(np.array([0.01, 0.01]), DEPTH, max(SKU.size), F)
    assert ok, why


def test_a_case_held_towards_the_bars_would_be_struck_by_them():
    ok, why = hand.fork_fits(np.array([0.0, -0.03]), DEPTH, max(SKU.size), F)
    assert not ok and '棒' in why


def test_a_case_held_far_along_x_leaves_a_tine_outside_its_bottom():
    ok, why = hand.fork_fits(np.array([0.05, 0.0]), DEPTH, max(SKU.size), F)
    assert not ok and '爪' in why


def test_the_transfer_is_accepted_when_the_tines_take_most_of_the_weight_evenly():
    r = hand.judge_support((0.40 * W, 0.38 * W), W, (0.0, 0.0), DEPTH, F, CELL.gripper, CELL.gantry)
    assert r.accepted and r.max_accel == pytest.approx(CELL.gantry.a_max[0])


def test_a_tine_over_its_load_limit_is_never_accepted():
    r = hand.judge_support((F.tine_load_limit + 5.0, 0.3 * W), W, (0.0, 0.0), DEPTH, F, CELL.gripper, CELL.gantry)
    assert not r.accepted and '片爪' in r.reason


def test_the_tines_push_in_proportion_to_where_the_mass_sits_so_the_cups_are_not_left_the_moment():
    m = 14.0 * 9.81
    left, right = hand.tine_caps(m, 0.06, F)
    assert left + right == pytest.approx(F.support_fraction * m)
    assert (right - left) * F.bar_x == pytest.approx(m * 0.06)          # the tines carry the moment about tool y
    _, _, ty = hand.suction_wrench(m, (0.06, 0.0), (left, right), (-F.bar_x, F.bar_x), 0.0)
    assert abs(ty) < 1e-9


def test_pushes_are_kept_within_one_tine_s_limit():
    left, right = hand.tine_caps(14.0 * 9.81, 0.12, F)
    assert max(left, right) <= F.tine_load_limit and min(left, right) >= 0


def test_contents_in_a_far_corner_are_carried_only_at_a_cut_acceleration():
    # Centre of mass 60 mm off along both edges, 14 kg: even pushes leave the cups more moment than
    # half their limit; split pushes carry it, but the heavy tine then limits the acceleration.
    m = 14.0 * 9.81
    even = hand.judge_support((0.36 * m, 0.36 * m), m, (0.06, 0.06), DEPTH, F, CELL.gripper, CELL.gantry)
    assert not even.accepted
    split = hand.judge_support(hand.tine_caps(m, 0.06, F), m, (0.06, 0.06), DEPTH, F, CELL.gripper, CELL.gantry)
    assert split.accepted and split.mode == 'uneven'
    assert CELL.gantry.slow_accel <= split.max_accel < CELL.gantry.a_max[0]
    k = 1 + split.max_accel / 9.81
    heavy = max(hand.tine_caps(m, 0.06, F)) * k
    worst = hand.cup_utilisation(hand.suction_wrench(m * k, (0.06, 0.06), [t * k for t in hand.tine_caps(m, 0.06, F)],
                                                     (-F.bar_x, F.bar_x), hand.tine_y(DEPTH, F)),
                                 CELL.gantry.cup_offsets, CELL.gripper.planning_coaxial_limit)
    assert heavy <= F.tine_load_limit + 1e-6 and worst <= 0.5 + 1e-6


def test_too_little_on_the_tines_is_not_a_transfer():
    r = hand.judge_support((0.10 * W, 0.10 * W), W, (0.0, 0.0), DEPTH, F, CELL.gripper, CELL.gantry)
    assert not r.accepted


def test_the_cups_left_with_force_and_moment_are_split_cup_by_cup():
    wrench = hand.suction_wrench(W, (0.03, 0.0), (0.0, 0.0), (-F.bar_x, F.bar_x), 0.0)
    loads = hand.cup_loads(wrench, CELL.gantry.cup_offsets)
    assert sum(loads) == pytest.approx(W)
    assert max(loads) > W / 4 > min(loads)                # the cups on the +x side carry more


def test_the_bars_face_the_side_the_tool_yaw_turns_them_to():
    assert np.allclose(hand.bar_direction(0.0), (0.0, -1.0))
    assert np.allclose(hand.bar_direction(math.pi), (0.0, 1.0))
    assert np.allclose(hand.bar_direction(math.pi / 2), (1.0, 0.0))


def test_touchdown_of_the_tines_is_a_lasting_change_either_way():
    # F0: at 120 Hz the tine readings rose on landing, at 240 Hz they fell. The sign is not relied on.
    for sign in (1, -1):
        d = hand.TouchdownDetector(carried=(50.0, 50.0), weight=W)
        hits = [d.update((50.0 + sign * 30.0, 50.0 + sign * 20.0)) for _ in range(3)]
        assert hits == [False, False, True]


def test_a_single_spike_is_not_a_touchdown():
    d = hand.TouchdownDetector(carried=(50.0, 50.0), weight=W)
    assert not d.update((90.0, 90.0)) and not d.update((50.5, 49.8)) and not d.update((90.0, 90.0))


def test_a_jump_far_beyond_noise_is_a_touchdown_at_once():
    d = hand.TouchdownDetector(carried=(50.0, 50.0), weight=W)
    assert d.update((50.0 + 4.5 * d.threshold, 50.0))


# ---------------------------------------------------------------- righting by tipping on the table (stage 2)
def test_the_pivot_is_placed_so_the_table_holds_the_case_before_and_after_it_tips():
    half = max(CELL.staging.table_size[:2]) / 2
    for w0, h0 in ((0.22, 0.27), (0.22, 0.36)):            # lying on its side; standing on its end
        p = hand.pivot_offset(w0, h0, half)
        assert p - w0 / 2 >= -half + hand.TABLE_MARGIN - 1e-9     # centre of mass on the table before
        assert p + h0 / 2 <= half - hand.TABLE_MARGIN + 1e-9      # and after


def test_a_table_too_small_for_the_tip_is_refused():
    with pytest.raises(ValueError):
        hand.pivot_offset(0.22, 0.36, 0.12)


def test_the_pivot_leaves_room_for_a_centre_of_mass_off_the_box_centre():
    # S1-end trial 0 (2026-10-18): the centre of mass 59 mm along the long axis - straight down while the
    # case hung from its end, so the test lift could not see it. Tipped, the case lay as designed with
    # its box centre 35 mm inside the edge, and its centre of mass went over it: it fell off the table.
    half, com = 0.23, 0.06
    p = hand.pivot_offset(0.22, 0.36, half, com=com)
    assert p - 0.22 / 2 - com >= -half + hand.TABLE_MARGIN - 1e-9
    assert p + 0.36 / 2 + com <= half - hand.TABLE_MARGIN + 1e-9
    with pytest.raises(ValueError):
        hand.pivot_offset(0.22, 0.36, 0.18, com=com)                  # the 0.36 m table of before


def test_the_cups_tip_the_case_with_far_less_than_turning_it_in_the_air():
    # Tipping on its edge, the table carries the weight; the cups only supply the moment about the edge.
    m = 14.0
    need = hand.pivot_cup_force(m, 0.22, 0.27)
    assert need == pytest.approx(m * 9.81 * 0.135 / math.hypot(0.11, 0.27), rel=1e-6)
    assert need <= 0.5 * len(CELL.gantry.cup_offsets) * CELL.gripper.planning_shear_limit


def test_the_cup_centre_moves_on_an_arc_about_the_bottom_edge_while_the_wrist_rolls_the_same_angle():
    t = np.array([0.0, 1.0])                                  # tip towards world +y
    pivot = np.array([1.72, 0.30, 0.12])
    c0 = np.array([1.72, 0.30 - 0.11, 0.12 + 0.27])          # centre of the top face of a case lying on its side
    path = hand.pivot_path(pivot, c0, t, steps=6)
    assert len(path) == 7
    for c, roll in path:
        assert np.linalg.norm(c - pivot) == pytest.approx(np.linalg.norm(c0 - pivot))
    end, roll_end = path[-1]
    assert roll_end == pytest.approx(-math.pi / 2)
    assert end[2] == pytest.approx(0.12 + 0.11)               # the face is now vertical at half the old width
    assert end[1] == pytest.approx(0.30 + 0.27)


# ---------------------------------------------------------------- after the cups let go on the tines (v05 12.4 (b))
def test_recovery_needs_the_tines_to_carry_the_weight_and_the_mass_to_sit_between_them():
    ok, why = hand.can_recover((0.5 * W, 0.5 * W), W, (0.0, 0.0), F)
    assert ok, why
    ok, why = hand.can_recover((0.05 * W, 0.05 * W), W, (0.0, 0.0), F)
    assert not ok and '重さ' in why                                   # the tines do not hold it
    ok, why = hand.can_recover((0.2 * W, 0.8 * W), W, (0.10, 0.0), F)
    assert not ok and '重心' in why                                   # too near a tine to stay on them
    ok, why = hand.can_recover((F.tine_load_limit + 1, 0.3 * W), W, (0.0, 0.0), F)
    assert not ok and '片爪' in why


# ---------------------------------------------------------------- contact forces (requirements v05 R1, chapter 3)
def test_a_lasting_force_over_its_limit_is_a_damage_risk_but_a_one_step_spike_is_counted_apart():
    lim = CELL.contact
    v = hand.contact_verdict('touch', static=lim.touch_static + 5, peak=lim.touch_peak - 1, limits=lim)
    assert v['damage_risk'] and not v['spike_over']
    v = hand.contact_verdict('touch', static=lim.touch_static - 5, peak=lim.touch_peak + 500, limits=lim)
    assert not v['damage_risk'] and v['spike_over']                  # spikes wait for the step-size check
    v = hand.contact_verdict('landing', static=lim.landing_static - 1, peak=10.0, limits=lim)
    assert not v['damage_risk']


def test_an_uncalibrated_reading_is_not_judged_in_newtons():
    v = hand.contact_verdict('touch', static=None, peak=None, limits=CELL.contact, raw=(12.0, 30.0))
    assert v['calibrated'] is False and v['damage_risk'] is False and v['raw'] == [12.0, 30.0]


def test_the_retracted_tines_are_a_box_beside_the_plate_above_the_cup_face():
    from shinai_pi.hand import fork_envelope
    f = load_cell().fork
    x0, x1, y0, y1, bottom = fork_envelope(f)
    assert x1 == -x0 == f.bar_x + f.tine_size[0] / 2
    assert abs(y0 - (f.bar_y - f.tine_size[1])) < 1e-12 and abs(y1 - (f.bar_y + f.bar_size[1] / 2)) < 1e-12
    assert abs(bottom - (-f.extension_range[0] - f.tine_size[2])) < 1e-12 and bottom > 0
    assert fork_envelope(f, extension=0.10)[4] < 0                       # lowered, they reach below the face
