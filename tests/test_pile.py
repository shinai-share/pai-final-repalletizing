"""Support-relation belief over a collapsed pile; decisions never read ground truth."""
import math

import numpy as np
import pytest

from shinai_pi.pile import Box, PileBelief, Verdict, support_graph

# Boxes are axis-aligned: (center x, y, z) and (size x, y, z); z is up, pallet top at z=0.
FLOOR = 0.0


def box(bid, cx, cy, cz, sx=0.4, sy=0.3, sz=0.25, conf=1.0):
    return Box(bid, (cx, cy, cz), (sx, sy, sz), confidence=conf)


def test_box_on_pallet_has_no_supporters():
    g = support_graph([box('a', 0, 0, 0.125)], floor=FLOOR)
    assert g.supporters('a') == set()
    assert g.on_floor('a')


def test_stacked_box_is_supported_by_lower_box():
    g = support_graph([box('a', 0, 0, 0.125), box('b', 0, 0, 0.375)], floor=FLOOR)
    assert g.supporters('b') == {'a'}
    assert g.supported_by('a') == {'b'}


def test_bridging_box_rests_on_two_supporters():
    g = support_graph([box('a', -0.2, 0, 0.125), box('b', 0.2, 0, 0.125), box('c', 0, 0, 0.375, sx=0.6)], floor=FLOOR)
    assert g.supporters('c') == {'a', 'b'}


def test_gap_above_tolerance_is_not_contact():
    g = support_graph([box('a', 0, 0, 0.125), box('b', 0, 0, 0.40)], floor=FLOOR, tol=0.01)
    assert g.supporters('b') == set()
    assert not g.on_floor('b')
    assert g.floating('b')


def test_removing_sole_supporter_destabilises_box_above():
    g = support_graph([box('a', 0, 0, 0.125), box('b', 0, 0, 0.375)], floor=FLOOR)
    assert g.would_fall_if_removed('a') == {'b'}
    assert g.would_fall_if_removed('b') == set()


def test_removing_one_of_two_supporters_is_safe_when_cog_still_inside():
    g = support_graph([box('a', -0.2, 0, 0.125), box('b', 0.2, 0, 0.125), box('c', 0.15, 0, 0.375, sx=0.6)], floor=FLOOR)
    assert g.would_fall_if_removed('a') == set()
    assert g.would_fall_if_removed('b') == {'c'}


def test_cascade_counts_boxes_that_lose_support_transitively():
    g = support_graph([box('a', 0, 0, 0.125), box('b', 0, 0, 0.375), box('c', 0, 0, 0.625)], floor=FLOOR)
    assert g.would_fall_if_removed('a') == {'b', 'c'}


def test_free_boxes_are_those_nothing_rests_on():
    g = support_graph([box('a', 0, 0, 0.125), box('b', 0, 0, 0.375), box('d', 1.0, 0, 0.125)], floor=FLOOR)
    assert g.free() == {'b', 'd'}


def test_belief_pick_order_prefers_free_boxes_with_zero_cascade():
    b = PileBelief(floor=FLOOR)
    b.observe([box('a', 0, 0, 0.125), box('b', 0, 0, 0.375), box('d', 1.0, 0, 0.125)], viewpoint='front')
    first = b.next_pick()
    assert first.box_id in {'b', 'd'}
    assert first.expected_cascade == 0.0


def test_belief_refuses_to_pick_when_nothing_is_safe_enough():
    b = PileBelief(floor=FLOOR, max_cascade_risk=0.2)
    # Only one visible box, but it was seen with low confidence and may be supporting unseen boxes.
    b.observe([box('a', 0, 0, 0.125, conf=0.3)], viewpoint='front')
    assert b.next_pick() is None
    assert b.verdict() is Verdict.NEED_OBSERVATION


def test_low_confidence_pose_inflates_cascade_risk():
    sure = PileBelief(floor=FLOOR)
    sure.observe([box('a', -0.2, 0, 0.125), box('b', 0.2, 0, 0.125), box('c', 0.15, 0, 0.375, sx=0.6, conf=1.0)], 'front')
    unsure = PileBelief(floor=FLOOR)
    unsure.observe([box('a', -0.2, 0, 0.125), box('b', 0.2, 0, 0.125), box('c', 0.15, 0, 0.375, sx=0.6, conf=0.4)], 'front')
    assert unsure.cascade_risk('a') > sure.cascade_risk('a')


def test_unseen_region_is_not_assumed_empty():
    b = PileBelief(floor=FLOOR)
    b.observe([box('a', 0, 0, 0.125)], viewpoint='front', seen_fraction=0.5)
    assert b.hidden_box_probability() > 0.0
    b.observe([box('a', 0, 0, 0.125)], viewpoint='back', seen_fraction=1.0)
    assert b.hidden_box_probability() == 0.0


def test_observation_gain_is_zero_when_pick_is_already_safe():
    b = PileBelief(floor=FLOOR)
    b.observe([box('a', 0, 0, 0.125), box('b', 0, 0, 0.375)], 'front', seen_fraction=1.0)
    assert b.observation_gain() == 0.0


def test_observation_gain_is_positive_when_order_could_change():
    b = PileBelief(floor=FLOOR, max_cascade_risk=0.2)
    b.observe([box('a', 0, 0, 0.125, conf=0.3)], 'front', seen_fraction=0.5)
    assert b.observation_gain() > 0.0


def test_post_pick_check_flags_unexpected_motion():
    b = PileBelief(floor=FLOOR)
    b.observe([box('a', 0, 0, 0.125), box('b', 0, 0, 0.375), box('d', 1.0, 0, 0.125)], 'front', seen_fraction=1.0)
    pick = b.next_pick()
    b.commit_pick(pick.box_id)
    remaining = [x for x in [box('a', 0, 0, 0.125), box('b', 0, 0, 0.375), box('d', 1.0, 0, 0.125)] if x.box_id != pick.box_id]
    assert b.verify_after_pick(remaining) == []
    moved = [Box(x.box_id, (x.center[0] + 0.1, x.center[1], x.center[2]), x.size, x.confidence) for x in remaining]
    assert set(b.verify_after_pick(moved)) == {x.box_id for x in remaining}


def test_empty_pile_is_complete():
    b = PileBelief(floor=FLOOR)
    b.observe([], 'front', seen_fraction=1.0)
    assert b.verdict() is Verdict.COMPLETE


@pytest.mark.parametrize('bad', [(math.nan, 0, 0), (0, math.inf, 0)])
def test_invalid_box_is_rejected(bad):
    with pytest.raises(ValueError):
        Box('x', bad, (0.1, 0.1, 0.1))


def test_non_positive_size_is_rejected():
    with pytest.raises(ValueError):
        Box('x', (0, 0, 0), (0.1, 0, 0.1))


# ---------------------------------------------------------------- oriented boxes and real geometry
from shinai_pi.perception.obb import OBB, quat_to_matrix  # noqa: E402


def tilted(bid, center, size, deg_about_x, conf=1.0):
    h = math.radians(deg_about_x) / 2
    return Box(bid, center, size, confidence=conf, rotation=quat_to_matrix((math.cos(h), math.sin(h), 0, 0)))


def test_leaning_box_is_supported_by_floor_and_neighbour_and_falls_without_it():
    a = Box('a', (0, 0, 0.25), (0.4, 0.3, 0.5))           # +y face at y = 0.15
    b = tilted('b', (0, 0, 0), (0.4, 0.3, 0.25), -25)     # top tipped toward a
    c = b.obb.corners()
    # translate so the lowest corner is on the floor and the nearest corner just touches a's face
    b = Box('b', (0, 0.1505 - c[:, 1].min(), -c[:, 2].min()), b.size, rotation=b.rotation)
    corners = b.obb.corners()
    floor_edge_y = corners[np.argmin(corners[:, 2]), 1]
    assert b.obb.center[1] < floor_edge_y                  # centre of mass leans toward a
    g = support_graph([a, b], floor=0.0, tol=0.01)
    assert g.on_floor('b')
    assert 'a' in g.supporters('b')
    assert g.would_fall_if_removed('a') == {'b'}


def test_pallet_as_static_support():
    pallet = OBB.make((0, 0, 0.072), (1.1, 1.1, 0.144))
    g = support_graph([box('a', 0, 0, 0.144 + 0.125)], floor=0.0, static=[pallet])
    assert g.on_floor('a') or g.on_static('a')
    assert g.would_fall_if_removed('a') == set()


def test_tilted_top_face_is_not_pickable():
    b = PileBelief(floor=FLOOR, max_face_tilt_deg=15.0)
    b.observe([tilted('t', (0, 0, 0.2), (0.4, 0.3, 0.25), 30)], 'top', seen_fraction=1.0)
    c = b.candidates()
    assert c[0].box_id == 't' and not c[0].pickable
    assert b.next_pick() is None


def test_box_under_an_overhang_is_not_pickable_from_above():
    low = box('low', 0, 0, 0.125)
    over = box('over', 0.25, 0, 0.625, sx=0.4)   # floating overhang above part of 'low'
    b = PileBelief(floor=FLOOR)
    b.observe([low, over], 'top', seen_fraction=1.0)
    cands = {c.box_id: c for c in b.candidates()}
    assert not cands['low'].pickable
    assert 'approach' in cands['low'].reason


def test_unexplained_unknown_space_drives_hidden_probability():
    from shinai_pi.perception.voxels import VoxelMap
    vm = VoxelMap((-1, -1, 0), (1, 1, 1), 0.05)
    b = PileBelief(floor=FLOOR)
    b.observe([box('a', 0, 0, 0.125)], 'top')
    b.set_known_space(vm, region=((-0.5, -0.5, 0.0), (0.5, 0.5, 0.6)))
    assert b.hidden_box_probability() > 0.9                 # nothing observed yet
    vm.free_n[:] = 1                                          # everything observed empty
    b.set_known_space(vm, region=((-0.5, -0.5, 0.0), (0.5, 0.5, 0.6)))
    assert b.hidden_box_probability() < 0.05


def test_fall_severity_weights_by_drop_height():
    # 'b' sits on 'a' (falls 0.25 m when a is removed); 'c' bridges a and low step 'd' (tips a few cm)
    g = support_graph([box('a', 0, 0, 0.125), box('b', 0, 0, 0.375)], floor=FLOOR)
    sev = g.fall_severity('a')
    assert set(sev) == {'b'} and sev['b'] == pytest.approx(1.0)
    low = support_graph([box('p', 0, 0, 0.025, sz=0.05), box('q', 0.15, 0, 0.075, sx=0.3, sz=0.05)], floor=FLOOR)
    sev2 = low.fall_severity('p')
    assert all(v < 0.5 for v in sev2.values())


def test_unreachable_case_is_not_pickable():
    b = PileBelief(floor=FLOOR, reach=((-1.0, 1.0), (-1.0, 1.0)))
    b.observe([box('far', 1.5, 0, 0.125)], 'top', seen_fraction=1.0)
    c = b.candidates()[0]
    assert not c.pickable and 'reach' in c.reason


def test_unknown_space_above_seen_free_space_cannot_hide_a_case():
    from shinai_pi.perception.voxels import VoxelMap
    vm = VoxelMap((-1, -1, 0), (1, 1, 1), 0.05)
    vm.free_n[:, :, :6] = 1                        # floor area seen empty up to 0.3 m, above that unknown
    b = PileBelief(floor=FLOOR)
    b.observe([], 'top')
    b.set_known_space(vm, region=((-0.5, -0.5, 0.0), (0.5, 0.5, 0.9)))
    assert b.hidden_box_probability() < 0.01


def test_safe_candidates_are_taken_from_the_top_down():
    b = PileBelief(floor=FLOOR, max_cascade_risk=0.25)
    b.observe([box('low', 1.0, 0, 0.125, conf=0.99), box('base', 0, 0, 0.125), box('top', 0, 0, 0.375, conf=0.9)], 'top')
    assert b.next_pick().box_id == 'top'


def test_case_the_model_cannot_hold_up_is_not_predicted_to_cascade():
    # 'ghost' floats 2 cm above 'a' (pose error): the pile is at rest, so its support is merely unexplained.
    g = support_graph([box('a', 0, 0, 0.125), box('ghost', 0, 0, 0.395), box('top', 0, 0, 0.645)], floor=FLOOR)
    assert 'ghost' in g.unexplained()
    assert g.would_fall_if_removed('top') == set()
    assert g.possibly_affected('top') == {'ghost'}          # it touches 'top', so it is not trusted
    assert 'ghost' not in g.would_fall_if_removed('a')


def test_unknown_sliver_too_small_for_a_case_cannot_hide_one():
    from shinai_pi.perception.voxels import VoxelMap
    vm = VoxelMap((-1, -1, 0), (1, 1, 1), 0.05)
    vm.free_n[:] = 1
    vm.free_n[0:2, :, :] = 0                     # a 10 cm strip along one edge, floor to top
    b = PileBelief(floor=FLOOR, case_dims=(0.36, 0.27, 0.22))
    b.observe([], 'top')
    b.set_known_space(vm, region=((-1, -1, 0), (1, 1, 1)))
    assert b.hidden_box_probability() == 0.0
    vm.free_n[0:8, 0:8, 0:8] = 0                 # a 40 cm cube: a case fits
    b.set_known_space(vm, region=((-1, -1, 0), (1, 1, 1)))
    assert b.hidden_box_probability() > 0.9


def test_an_estimate_that_overlaps_a_more_confident_neighbour_is_not_picked():
    # Two cases cannot share space: the less confident of two overlapping estimates is wrong
    # somewhere, and the cups aimed at it would land partly on the neighbour.
    pb = PileBelief(floor=FLOOR)
    pb.observe([box('sure', 0.0, 0.0, 0.125, conf=0.9), box('doubt', 0.0, 0.21, 0.125, conf=0.8)], 'top')
    ok, why = pb.pickability('doubt')
    assert not ok and 'sure' in why
    assert pb.pickability('sure')[0]


def test_touching_neighbours_do_not_count_as_overlapping():
    pb = PileBelief(floor=FLOOR)
    pb.observe([box('a', 0.0, 0.0, 0.125, conf=0.8), box('b', 0.0, 0.305, 0.125, conf=0.9)], 'top')
    assert pb.pickability('a')[0] and pb.pickability('b')[0]


def test_with_unknown_contents_a_box_near_the_edge_of_its_support_is_uncertain_not_stable():
    # 0.4 m box on a 0.4 m supporter shifted 0.17 m: its centre is 3 cm inside the edge. Centred
    # contents would hold; contents 6 cm towards the edge would not.
    boxes = [box('low', 0.0, 0.0, 0.125), box('top', 0.17, 0.0, 0.375)]
    certain = support_graph(boxes, floor=FLOOR)
    unsure = support_graph(boxes, floor=FLOOR, com_margin=0.06)
    everyone = {'low', 'top'}
    assert certain.stability('top', everyone) == 'stable'
    assert unsure.stability('top', everyone) == 'uncertain'


def test_removing_a_supporter_that_leaves_a_box_uncertain_carries_half_the_risk_of_a_fall():
    # 'top' bridges 'a' and 'b'; without 'b' its centre still sits over 'a', 2 cm from the edge.
    boxes = [box('a', -0.10, 0.0, 0.125), box('b', 0.32, 0.0, 0.125), box('top', 0.08, 0.0, 0.375)]
    certain = support_graph(boxes, floor=FLOOR)
    unsure = support_graph(boxes, floor=FLOOR, com_margin=0.06)
    assert certain.would_fall_if_removed('b') == set() and certain.fall_severity('b') == {}
    sev = unsure.fall_severity('b')
    assert set(sev) == {'top'} and 0.0 < sev['top'] <= 0.5
