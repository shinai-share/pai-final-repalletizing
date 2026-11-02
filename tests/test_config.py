import dataclasses

import pytest

from shinai_pi.config import load_cell, validate


def test_default_cell_loads_and_matches_owner_geometry():
    c = load_cell()
    assert c.pallet.gap == pytest.approx(0.8)
    assert abs(c.pallet.b_center[0] - c.pallet.a_center[0]) - c.pallet.size[0] == pytest.approx(0.8)
    assert c.layout.cols * c.layout.rows * c.layout.layers == 48


def test_wrong_gap_is_rejected():
    c = load_cell()
    bad = dataclasses.replace(c, pallet=dataclasses.replace(c.pallet, gap=0.5))
    with pytest.raises(ValueError, match='gap'):
        validate(bad)


def test_gantry_must_reach_both_pallets():
    c = load_cell()
    bad = dataclasses.replace(c, gantry=dataclasses.replace(c.gantry, x_range=(-1.0, 1.0)))
    with pytest.raises(ValueError, match='x range'):
        validate(bad)


def test_overhanging_layout_is_rejected():
    c = load_cell()
    bad = dataclasses.replace(c, layout=dataclasses.replace(c.layout, cols=4))
    with pytest.raises(ValueError, match='overhang'):
        validate(bad)


def test_unknown_sku_raises():
    with pytest.raises(KeyError):
        load_cell().sku('nope')


def test_gantry_has_a_two_axis_wrist_with_limits_for_every_axis():
    g = load_cell().gantry
    assert len(g.limits) == 6 and len(g.v_max) == 6 and len(g.a_max) == 6
    assert g.roll_range[0] <= -1.6 and g.roll_range[1] >= 1.6          # a full quarter turn either way
    assert g.pitch_range[1] > 0 and g.wrist_torque_limit >= 33.0         # worst 12 kg case, 0.28 m arm


def test_the_robot_plans_on_half_an_ideal_cup_and_the_simulator_never_lets_go_below_that():
    # The robot's decisions use half the pull of an 80 mm cup at 60 kPa, for leakage through corrugated
    # board (requirements v05), and a safety factor on top. Isaac's Surface Gripper lets go on transient
    # constraint values far below its set limit (probe_grip_limit): its limit is set high so the artefact
    # does not stand in for the seal; the seal's capacity is the planner's.
    gr = load_cell().gripper
    ideal = 60e3 * 3.14159 * 0.04 ** 2
    assert gr.planning_coaxial_limit == pytest.approx(ideal / 2, rel=0.02)
    assert gr.coaxial_force_limit >= 4 * ideal and gr.shear_force_limit >= 2 * ideal
    assert gr.cup_twist_stiffness < gr.cup_tilt_stiffness
    assert gr.planning_shear_limit <= 0.5 * gr.planning_coaxial_limit + 1e-9


def test_planning_on_more_than_the_cup_can_hold_is_rejected():
    c = load_cell()
    bad = dataclasses.replace(c, gripper=dataclasses.replace(c.gripper, planning_coaxial_limit=c.gripper.coaxial_force_limit * 1.5))
    with pytest.raises(ValueError, match='plan'):
        validate(bad)


def test_righting_tables_hold_a_case_before_and_after_it_tips_and_leave_room_around_it():
    # A case is righted by tipping it over its bottom edge on the table (requirements v05, stage 2):
    # the table must carry its centre of mass lying and standing, for a side and for an end.
    from shinai_pi import hand
    c = load_cell()
    st = c.staging
    w, d, h = c.skus[0].size
    half = st.table_size[1] / 2                                         # cases tip along y
    assert len(st.tables) >= 1
    for w0, h0 in ((h, d), (h, w)):                                     # lying on its side; standing on its end
        hand.pivot_offset(w0, h0, half, com=st.com_allowance)
    assert st.com_allowance >= c.variation.com_xy                      # the cases' centres of mass may lie this far off
    p = c.pallet
    spot_half = (0.15, 0.21)                                            # handover marks are 0.30 x 0.42
    for tx, ty in st.tables:
        out = 1.0 if ty >= 0 else -1.0                                  # cases tip away from the other table
        # Far side of the tipped case, the plate on it and a margin, from the table centre.
        reach = max(hand.pivot_offset(w0, h0, half, com=st.com_allowance) + h0 for w0, h0 in ((h, d), (h, w))) + c.gantry.tool_size[2] + 0.02
        for hx, hy in c.handover.spots:
            if abs(hx - tx) < max(c.skus[0].size) / 2 + spot_half[0]:
                assert (hy - ty) * out <= 0 or abs(hy - ty) - spot_half[1] > reach, (hx, hy)
        assert tx - st.table_size[0] / 2 > p.b_center[0] + p.size[0] / 2   # beside pallet B, not on it
        assert st.table_size[0] >= max(c.skus[0].size)                  # the long edge lies along x


def test_wrist_limits_must_be_increasing():
    c = load_cell()
    bad = dataclasses.replace(c, gantry=dataclasses.replace(c.gantry, pitch_range=(1.0, -1.0)))
    with pytest.raises(ValueError, match='increasing'):
        validate(bad)


def test_the_camera_over_pallet_b_has_the_finer_resolution_its_placement_check_needs():
    # 640 px over the pallet is 4.3 mm per pixel: the long edge of a case then reads up to 2 degrees
    # off, the same as the placement tolerance. B's camera checks placements, so it gets finer pixels.
    oc = load_cell().overhead_camera
    assert tuple(oc.resolution_b) == (1280, 960)
    assert tuple(oc.resolution) == (640, 480)


def test_the_fork_reaches_under_the_tallest_upright_case_and_retracts_clear_of_the_plate():
    # Only upright cases ride on the tines (executive); the stroke was cut to that on 2026-10-31 (review: the drawn-up
    # bars stood 0.69 m above the cup face)
    c = load_cell()
    f = c.fork
    tallest = max(s.size[2] for s in c.skus)
    assert f.extension_range[1] >= tallest + f.grip_gap + f.clearance
    # Retracted, the tines' underside is above the back of the plate: nothing hangs below the cups.
    assert -f.extension_range[0] - f.tine_size[2] >= c.gantry.tool_size[2]
    assert 0 < f.support_fraction < 1 and f.tine_load_limit > 0


def test_a_fork_that_cannot_reach_under_the_case_is_rejected():
    c = load_cell()
    bad = dataclasses.replace(c, fork=dataclasses.replace(c.fork, extension_range=(-0.08, 0.20)))
    with pytest.raises(ValueError, match='fork'):
        validate(bad)


def test_a_fork_that_hangs_below_the_plate_when_retracted_is_rejected():
    c = load_cell()
    bad = dataclasses.replace(c, fork=dataclasses.replace(c.fork, extension_range=(-0.03, 0.40)))
    with pytest.raises(ValueError, match='retract'):
        validate(bad)


def test_the_bars_stand_clear_of_a_case_held_square_on_the_cups():
    # The bars come down beside the held case's long face: they must not touch it.
    c = load_cell()
    f = c.fork
    depth = sorted(c.skus[0].size, reverse=True)[1]          # held top down, long edge along tool x
    assert abs(f.bar_y) - f.bar_size[1] / 2 > depth / 2 + 0.02
    bad = dataclasses.replace(c, fork=dataclasses.replace(f, bar_y=-0.14))
    with pytest.raises(ValueError, match='bar'):
        validate(bad)


def test_the_empty_cups_feel_a_face_well_below_the_force_that_could_mark_a_carton():
    # probe_zload: a load on the cups reads 1/8, a force on the plate about 1:1. A touch is the plate
    # pressing a face, so it is set in newtons with the plate's scale, under the damage limit.
    c = load_cell().contact
    assert c.link_force_gain > 0 and c.cup_load_gain > c.link_force_gain   # calibrated (artifacts/probe_zload.json)
    assert 0 < c.touch_detect < c.touch_static


def test_the_collision_load_lies_above_what_the_empty_tool_reads_from_its_own_motion_and_below_a_crush():
    # Empty moves read up to 66 raw (about 69 N) from inertia alone (S3 102 replay); a struck stack read 3,275.
    c = load_cell().contact
    assert 100.0 <= c.collision <= 300.0
