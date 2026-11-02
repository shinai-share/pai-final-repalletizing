"""The gantry's sliding parts stay in their guides over all of their travel (review 2026-10-29: in the submission
video the ram hung free of its trolley below 0.6 m of travel; the fork bars had no mount; a fully inserted tine
left its bar). One table (shinai_pi/structure.py) gives the scene its boxes and these tests their geometry."""
import dataclasses

import numpy as np
import pytest

from shinai_pi import structure as st
from shinai_pi.config import load_cell, validate

CELL = load_cell()


@pytest.mark.parametrize('slide', st.slides(CELL), ids=lambda s: s.joint)
def test_every_slide_keeps_its_engagement_over_all_of_its_travel(slide):
    lo, hi = slide.travel
    for q in np.linspace(lo, hi, 81):
        assert st.engagement(slide, q) >= slide.min_engagement - 1e-9, (slide.joint, q)


@pytest.mark.parametrize('slide', st.slides(CELL), ids=lambda s: s.joint)
def test_every_runner_lies_inside_its_guide_across_the_slide(slide):
    for axis in range(3):
        if axis == slide.axis:
            continue
        g0, g1 = st.extent(slide.guide, axis)
        r0, r1 = st.extent(slide.runner, axis)
        assert g0 - 1e-9 <= r0 and r1 <= g1 + 1e-9, (slide.joint, axis)


def test_the_slides_cover_every_prismatic_joint_of_the_tool():
    names = {s.joint for s in st.slides(CELL)}
    assert {'z_joint', 'fork_z_l', 'fork_z_r', 'fork_y_l', 'fork_y_r'} <= names


def test_the_old_ram_would_hang_free_of_its_trolley():
    short = dataclasses.replace(CELL, structure=dataclasses.replace(CELL.structure, ram_length=0.90))
    ram = next(s for s in st.slides(short) if s.joint == 'z_joint')
    assert st.engagement(ram, CELL.gantry.z_range[0]) < 0       # a gap, not an overlap


def test_the_ram_s_mass_follows_from_its_tube():
    s = CELL.structure
    area = s.ram_section ** 2 - (s.ram_section - 2 * s.ram_wall) ** 2
    assert st.ram_mass(CELL) == pytest.approx(area * s.ram_length * s.steel_density)
    assert 30.0 < st.ram_mass(CELL) < 45.0


def test_the_tines_still_reach_where_the_plan_puts_them():
    # the tine was lengthened backwards: its tip at full insert is where it always was
    f = CELL.fork
    tine = next(b for b in st.fork_boxes(CELL, 'l') if b.name == 'tine')
    assert st.extent(tine, 1)[1] + f.insert_range[1] == pytest.approx(f.bar_y + f.insert_range[1])


def test_the_park_height_is_kept_inside_the_travel():
    g = CELL.gantry
    assert g.z_range[0] <= g.park_z <= g.z_range[1]
    bad = dataclasses.replace(CELL, gantry=dataclasses.replace(g, park_z=g.z_range[1] + 0.1))
    with pytest.raises(ValueError, match='park'):
        validate(bad)


def test_a_structure_that_cannot_hold_a_slide_fails_on_loading():
    bad = dataclasses.replace(CELL, structure=dataclasses.replace(CELL.structure, ram_length=0.90))
    with pytest.raises(ValueError, match='z_joint'):
        validate(bad)


def test_the_tine_sleeve_stays_clear_of_a_held_case_s_face():
    # the sleeve reaches forward rather than the tine backwards: a longer tine's retracted tail made the plan refuse
    # the crosswise grip on a steep face (test_executive: 0.24 m and longer)
    held_depth = max(sorted(sk.size, reverse=True)[1] for sk in CELL.skus)
    sleeve = next(b for b in st.fork_boxes(CELL, 'l') if b.name == 'sleeve')
    assert -held_depth / 2 - st.extent(sleeve, 1)[1] >= st.SLEEVE_CLEAR - 1e-9



# ---------------------------------------------------------------- review 2026-10-31: the fork's mount and the wrist
def test_every_fixed_part_of_the_tool_is_fastened_to_the_plate_over_a_face():
    # The fork's arm met the plate along an edge only: no area, no load path. Every part fixed to the wrist must
    # reach the plate through joins over a face (or welded into another part).
    assert st.unfastened(st.tool_frame(CELL)) == []


def test_a_part_meeting_another_along_an_edge_is_not_fastened():
    plate = st.Box('wrist_r', 'plate', (0.30, 0.22, 0.06), (0.0, 0.0, 0.03))
    on_edge = st.Box('wrist_r', 'arm', (0.04, 0.05, 0.04), (0.0, -0.135, 0.08))     # touches along y=-0.11, z=0.06
    on_face = st.Box('wrist_r', 'yoke', (0.30, 0.05, 0.06), (0.0, -0.135, 0.03))    # face to face on y=-0.11
    assert [b.name for b in st.unfastened([plate, on_edge])] == ['arm']
    assert st.unfastened([plate, on_face]) == []


def test_nothing_on_the_tool_reaches_below_the_cup_face_with_the_fork_drawn_up():
    f = CELL.fork
    for b, R, p in st.tool_boxes_at(CELL, extension=f.extension_range[0], insert=0.0):
        assert st.extent(b, 2)[0] + p[2] >= -1e-9, b.name


def test_the_tool_clears_the_ram_trolley_and_wrist_over_every_allowed_pose():
    assert st.interference(CELL) is None


def test_the_wrist_rule_keeps_the_bars_from_swinging_into_the_ram():
    # Rolled one way and pitched, the drawn-up bars lean into the ram (up to 89 mm in the old fork). Turning the tool a
    # half turn gives the same face the opposite roll, so the rule costs no reachable face.
    assert not st.wrist_allowed(0.3, -0.6)
    assert st.wrist_allowed(0.3, 0.6) and st.wrist_allowed(0.0, -1.5) and st.wrist_allowed(-0.4, 0.0)


def test_the_old_fork_met_the_ram_when_the_wrist_was_free():
    old = dataclasses.replace(CELL, fork=dataclasses.replace(CELL.fork, bar_size=(0.04, 0.015, 0.62),
                                                              extension_range=(-0.08, 0.40)))
    assert st.interference(old, wrist_rule=False) is not None
