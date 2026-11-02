"""Mission logic on the analytic backend: completion, safety stops, grip failures, tipped cases,
and that the executive never reads ground truth."""
import ast
import math
from pathlib import Path

import numpy as np
import pytest

from shinai_pi.config import load_cell
from shinai_pi.events import EventLog
from shinai_pi.executive import Executive, MissionConfig
from shinai_pi.fake_backend import FakeBackend
from shinai_pi.motion import tool_rotation
from shinai_pi.perception.obb import OBB, quat_to_matrix, yaw_matrix

CELL = load_cell()
SKU = CELL.sku('BEV-500x24')
W, D, H = SKU.size
A = CELL.pallet.a_center
TOP = CELL.pallet.size[2]


def upright(i, j, layer, yaw=0.0):
    x = A[0] - W + i * W
    y = A[1] - D / 2 + j * D
    return OBB.make((x, y, TOP + H * (layer + 0.5)), SKU.size, yaw_matrix(yaw))


def run(boxes, policy='shinai', **kw):
    be = FakeBackend(CELL, boxes, **{k: v for k, v in kw.items() if k in ('curtain_breaks', 'grip_failures')})
    ev = EventLog('test')
    ex = Executive(CELL, be, ev, MissionConfig(SKU.id, policy=policy))
    return ex.run(), ev, be


def small_pile():
    boxes = {f'case_{i:02d}': upright(i % 3, i // 3, 0) for i in range(6)}
    boxes['case_06'] = upright(1, 0, 1)
    return boxes


def test_small_intact_pile_is_completed_on_b():
    rep, ev, be = run(small_pile())
    assert rep.status == 'verified', rep
    assert len(rep.placed) == 7
    assert rep.placed[0] == 'case_06'                    # the case on top comes off first
    checks = ev.of('PLACEMENT_CHECKED')
    assert len(checks) == 7 and all(e.data['ok'] for e in checks)
    assert ev.events[0].type == 'MISSION_ACCEPTED' and ev.events[-1].type == 'MISSION_COMPLETE'
    seqs = [e.seq for e in ev.events]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)


def test_light_curtain_breach_stops_then_resumes_after_operator_reset():
    rep, ev, be = run(small_pile(), curtain_breaks=[(20.0, 24.0)])
    assert rep.safety_stops >= 1
    stop = ev.of('SAFETY_STOP')[0]
    cleared = ev.of('SAFETY_CLEARED')[0]
    assert cleared.t >= 26.0 and cleared.data['operator'] == '作業者'
    assert stop.seq < cleared.seq
    assert rep.status == 'verified'


def test_grip_failures_are_retried_then_handed_over():
    rep, ev, _ = run(small_pile(), grip_failures={'case_06': 2})
    assert len(ev.of('GRIP_FAILED')) == 2
    assert 'case_06' in rep.placed
    rep2, ev2, _ = run(small_pile(), grip_failures={'case_06': 10})
    assert 'case_06' in rep2.remaining
    assert any(e.data.get('case') == 'case_06' for e in ev2.of('HANDOVER'))
    # The case left for a person also hides the space beneath it: nothing can be called verified there.
    assert rep2.status == 'unresolved' and rep2.remaining['case_06'] == '吸着できなかった'


def standing_height(o):
    """Height of a case as it stands: the extent of the box axis closest to vertical."""
    i = int(np.argmax(np.abs(o.rotation[2, :])))
    return float(o.size[i])


@pytest.mark.parametrize('name, quat, height', [
    ('lying on its side', (math.cos(math.pi / 4), math.sin(math.pi / 4), 0, 0), D),
    ('standing on its end', (math.cos(math.pi / 4), 0, math.sin(math.pi / 4), 0), W),
])
def test_a_case_not_upright_is_righted_on_the_staging_table_and_stacked_on_b(name, quat, height):
    boxes = small_pile()
    boxes['case_06'] = OBB.make((A[0], A[1], TOP + H + height / 2), SKU.size, quat_to_matrix(quat))
    rep, ev, be = run(boxes)
    assert [e.data['case'] for e in ev.of('REORIENTED')] == ['case_06'], name
    assert 'case_06' in rep.placed and not rep.handed_over
    assert abs(standing_height(be.truth()['case_06']) - H) < 1e-6           # bottom face down on B
    assert rep.status == 'verified'


def test_supporter_of_an_overhang_is_not_taken_first():
    # case_10 bridges case_00 and case_01 with its centre over case_01; removing case_01 first would drop it.
    boxes = {'case_00': upright(0, 0, 0), 'case_01': upright(1, 0, 0)}
    bridge = OBB.make((A[0] - W / 2 + 0.12, A[1] - D / 2, TOP + 1.5 * H), SKU.size)
    boxes['case_10'] = bridge
    rep, ev, be = run(boxes)
    first = ev.of('PICK_PLANNED')[0].data
    assert first['case'] == 'case_10'
    assert be.collapses == []
    assert rep.status == 'verified'


def test_unknown_policy_is_rejected():
    with pytest.raises(ValueError):
        Executive(CELL, FakeBackend(CELL, {}), EventLog('x'), MissionConfig(SKU.id, policy='guess'))


def test_cognitive_core_never_imports_ground_truth():
    root = Path(__file__).resolve().parents[1] / 'shinai_pi'
    core = ['executive.py', 'pile.py', 'viewplan.py', 'safety.py', 'perception/boxfit.py', 'perception/voxels.py']
    for rel in core:
        tree = ast.parse((root / rel).read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                mod = getattr(node, 'module', '') or ''
                names = [a.name for a in node.names]
                assert 'truth' not in mod and 'evaluation' not in mod, (rel, mod)
                assert not any('truth' in n for n in names), (rel, names)
                assert 'sim' not in mod.split('.'), (rel, mod)


def test_every_carried_case_is_measured_from_below_before_placing():
    rep, ev, _ = run(small_pile())
    lifted = ev.of('LIFTED')
    assert lifted and all(e.data['measured_from_below'] for e in lifted)
    under = [e for e in ev.of('OBSERVED') if e.data['view'] == 'under']
    assert under and all(abs(e.data['height_m'] - H) < 0.015 for e in under)   # includes the cup-to-face gap


def test_handover_spots_are_not_reused_after_a_case_is_put_there():
    # Two tipped cases and no staging table to right them on: the second must go to a different
    # handover spot, because the map is refreshed.
    import dataclasses
    no_tables = dataclasses.replace(CELL, staging=dataclasses.replace(CELL.staging, tables=()))
    boxes = small_pile()
    tip = quat_to_matrix((math.cos(math.pi / 4), math.sin(math.pi / 4), 0, 0))
    boxes['case_06'] = OBB.make((A[0], A[1], TOP + H + D / 2), SKU.size, tip)
    boxes['case_07'] = OBB.make((A[0] + 0.40, A[1] + 0.05, TOP + H + D / 2), SKU.size, tip)
    be = FakeBackend(no_tables, boxes)
    rep = Executive(no_tables, be, EventLog('test'), MissionConfig(SKU.id)).run()
    assert sorted(rep.handed_over) == ['case_06', 'case_07']
    spots = [tuple(np.round(be.truth()[k].center[:2], 1)) for k in rep.handed_over]
    assert spots[0] != spots[1]


def test_more_tipped_cases_than_handover_spots_are_stacked_two_high():
    # Eight cases lying on their side, six handover spots: two spots take a second case of the
    # same lie, verified from B's camera, and no spot is stacked higher than two.
    tip = quat_to_matrix((math.cos(math.pi / 4), math.sin(math.pi / 4), 0, 0))   # height D, footprint W x H
    boxes = {}
    for n in range(8):
        i, j = n % 2, n // 2
        boxes[f'case_{n:02d}'] = OBB.make((A[0] - 0.2 + i * 0.40, A[1] - 0.36 + j * 0.24, TOP + D / 2), SKU.size, tip)
    import dataclasses
    six = dataclasses.replace(CELL, handover=dataclasses.replace(CELL.handover, spots=CELL.handover.spots[:6]),
                              staging=dataclasses.replace(CELL.staging, tables=()))
    be = FakeBackend(six, boxes)
    ev = EventLog('test')
    rep = Executive(six, be, ev, MissionConfig(SKU.id)).run()
    assert sorted(rep.handed_over) == sorted(boxes)
    assert not ev.of('MISSION_INCOMPLETE')
    truth = be.truth()
    heights = sorted(round(truth[k].center[2] / D) for k in rep.handed_over)
    assert heights.count(1) == 6 and heights.count(2) == 2          # centres at D/2 and 3D/2
    spots = [tuple(np.round(truth[k].center[:2], 1)) for k in rep.handed_over]
    assert max(spots.count(s) for s in spots) == 2


def test_a_case_is_not_stacked_on_a_case_lying_another_way():
    # An upright-on-end case (height W) must not go on top of a case lying at height D.
    from shinai_pi.executive import Executive as E
    longest = max(SKU.size)
    assert E._stackable([(D, (W, H))], W, longest) is False
    assert E._stackable([(D, (W, H))], D, longest) is True
    assert E._stackable([(D, (W, H)), (D, (W, H))], D, longest) is False
    assert E._stackable([], D, longest) is True


def test_a_case_standing_on_end_is_never_stacked():
    # Two cases on end make a column three times taller than its base: never stacked.
    from shinai_pi.executive import Executive as E
    assert E._stackable([(W, (D, H))], W, max(SKU.size)) is False
    assert E._stackable([], W, max(SKU.size)) is True


def test_long_edge_yaw_error_is_measured_modulo_half_turn():
    from shinai_pi.executive import long_edge_yaw_error_deg
    assert abs(long_edge_yaw_error_deg(upright(0, 0, 0, yaw=0.0), 90.0) - 90.0) < 1e-6
    assert abs(long_edge_yaw_error_deg(upright(0, 0, 0, yaw=math.radians(88)), 90.0) - 2.0) < 1e-6
    assert abs(long_edge_yaw_error_deg(upright(0, 0, 0, yaw=math.pi), 0.0)) < 1e-6
    assert abs(long_edge_yaw_error_deg(upright(0, 0, 0, yaw=math.radians(-89)), 90.0) - 1.0) < 1e-6


def test_a_case_that_turns_on_the_cups_while_carried_is_squared_up_by_sight_before_release():
    # The case turns 30 degrees on the cups after the floor camera measured it. Held just above its
    # slot, B's camera sees the turn and the tool squares it up before letting go.
    be = FakeBackend(CELL, small_pile(), slips={'case_06': 30.0})
    ev = EventLog('test')
    rep = Executive(CELL, be, ev, MissionConfig(SKU.id)).run()
    checks = [e.data for e in ev.of('PLACEMENT_CHECKED') if e.data['case'] == 'case_06']
    assert checks[0]['ok'] is True and checks[0]['yaw_error_deg'] < 2
    aims = [e.data for e in ev.of('AIMED') if e.data['case'] == 'case_06']
    assert aims and abs(aims[0]['yaw_correction_deg']) > 20
    assert not ev.of('PLACEMENT_REPAIRED') and rep.status == 'verified'


def test_a_turn_smaller_than_the_hanging_view_can_resolve_is_left_to_the_floor_camera():
    # Under the plate only a rim of the top face shows; its edge reads a degree or so off. Turning
    # the case by that reading puts a real error in, so a small turn is not acted on.
    be = FakeBackend(CELL, small_pile(), slips={'case_06': 1.0})
    ev = EventLog('test')
    Executive(CELL, be, ev, MissionConfig(SKU.id)).run()
    aims = [e.data for e in ev.of('AIMED') if e.data['case'] == 'case_06']
    assert all(abs(a['yaw_correction_deg']) < 0.5 for a in aims)


def test_a_case_that_turns_as_it_is_let_go_is_found_and_put_right():
    # The case turns 8 degrees as the vacuum is released, after the last look. The check on B must
    # see it, re-grip it, measure again and place it square.
    be = FakeBackend(CELL, small_pile(), release_slips={'case_06': 8.0})
    ev = EventLog('test')
    rep = Executive(CELL, be, ev, MissionConfig(SKU.id)).run()
    checks = [e.data for e in ev.of('PLACEMENT_CHECKED') if e.data['case'] == 'case_06']
    assert checks[0]['ok'] is False and checks[0]['yaw_error_deg'] > 5
    assert checks[-1]['ok'] is True and checks[-1]['yaw_error_deg'] < 2
    assert [e.data['case'] for e in ev.of('PLACEMENT_REPAIRED')] == ['case_06']
    from shinai_pi.executive import long_edge_yaw_error_deg
    from shinai_pi.layout import pallet_slots
    slot = pallet_slots(CELL, SKU, CELL.pallet.b_center)[[e.data for e in ev.of('PLACED') if e.data['case'] == 'case_06'][0]['slot']]
    assert long_edge_yaw_error_deg(be.truth()['case_06'], slot.yaw_deg) < 2
    assert rep.status == 'verified'


def test_a_case_pushed_by_its_neighbour_landing_is_found_and_put_right():
    # S3 104 (2026-10-18): case_09 landed 32 mm off and was put right, but it had pushed case_17 beside it,
    # already checked, to 10.9 mm off its slot. Only the case just placed was looked at again.
    be = FakeBackend(CELL, small_pile(), release_pushes={'case_00': 0.02})
    ev = EventLog('test')
    rep = Executive(CELL, be, ev, MissionConfig(SKU.id)).run()
    pushed = [e.data['case'] for e in ev.of('PLACEMENT_CHECKED') if not e.data['ok']]
    assert pushed and 'case_00' not in pushed
    assert [e.data['case'] for e in ev.of('PLACEMENT_REPAIRED')] == pushed
    from shinai_pi.layout import pallet_slots
    slots = pallet_slots(CELL, SKU, CELL.pallet.b_center)
    for e in ev.of('PLACED'):
        s = slots[e.data['slot']]
        assert np.linalg.norm(be.truth()[e.data['case']].center[:2] - np.asarray(s.center[:2])) <= 0.010
    assert rep.status == 'verified'


def test_cases_missing_from_the_load_count_are_reported_for_a_person_to_look_for():
    # The load is known to hold nine cases; seven are in the cell. Two may have been thrown
    # outside it in transport: the mission does not call itself complete.
    be = FakeBackend(CELL, small_pile())
    ev = EventLog('test')
    rep = Executive(CELL, be, ev, MissionConfig(SKU.id, expected_cases=9)).run()
    assert len(rep.placed) + len(rep.handed_over) == 7
    assert rep.not_found == 2 and rep.status == 'unresolved'
    end = ev.of('MISSION_INCOMPLETE')[-1].data
    assert end['not_found'] == 2 and '作業場の外' in end['reason']


def test_without_a_load_count_nothing_is_reported_missing():
    rep, ev, _ = run(small_pile())
    assert rep.not_found == 0 and rep.status == 'verified'


def test_a_test_lift_weighs_the_case_and_finds_its_centre_of_mass():
    # Shape alone cannot tell where the contents sit. A low test lift reads the load on the z
    # axis and the torques on the wrist, as a person feels a box tip in the hands.
    be = FakeBackend(CELL, small_pile(), coms={'case_06': (0.05, -0.03, 0.0)}, masses={'case_06': 11.0})
    ev = EventLog('test')
    Executive(CELL, be, ev, MissionConfig(SKU.id)).run()
    m = [e.data for e in ev.of('LOAD_MEASURED') if e.data['case'] == 'case_06'][0]
    assert abs(m['mass_kg'] - 11.0) < 0.05
    assert abs(m['com_offset_mm'][0] - 50.0) < 3 and abs(m['com_offset_mm'][1] + 30.0) < 3


def test_every_placement_is_looked_at_again_after_the_case_has_settled():
    rep, ev, _ = run(small_pile())
    settled = ev.of('PLACEMENT_SETTLED')
    assert sorted(e.data['case'] for e in settled) == sorted(rep.placed)
    assert all(e.data['ok'] and e.data['drift_mm'] <= 5.0 and e.data['waited_s'] >= 3.0 for e in settled)


def test_a_mission_holding_a_case_for_a_person_ends_interrupted():
    from shinai_pi.executive import end_status
    assert end_status(remaining={}, unresolved_remaining=0, not_found=0, holding=True, failed_checks=0, handed=0) == 'interrupted'
    assert end_status(remaining={}, unresolved_remaining=0, not_found=0, holding=False, failed_checks=1, handed=0) == 'unresolved'
    assert end_status(remaining={'a': '吸着できなかった'}, unresolved_remaining=0, not_found=0, holding=False,
                      failed_checks=0, handed=0) == 'verified_with_handover'
    assert end_status(remaining={}, unresolved_remaining=0, not_found=0, holding=False, failed_checks=0, handed=0) == 'verified'


def test_a_case_tipped_over_on_the_table_does_not_hang_on_the_cups_so_their_droop_does_not_tilt_it():
    # Bellows cups bend 12-15 degrees under a case held on its side in the air (Isaac). Tipped over
    # its bottom edge on the righting table, the table carries it: it stands square without a correction.
    boxes = small_pile()
    boxes['case_06'] = OBB.make((A[0], A[1], TOP + H + D / 2), SKU.size,
                                quat_to_matrix((math.cos(math.pi / 4), math.sin(math.pi / 4), 0, 0)))
    be = FakeBackend(CELL, boxes, sag_deg=13.0)
    ev = EventLog('test')
    rep = Executive(CELL, be, ev, MissionConfig(SKU.id)).run()
    r = ev.of('REORIENTED')[0].data
    assert r['method'] == 'tip_on_table' and abs(r['standing_m'] - H) < 0.01
    assert 'case_06' in rep.placed and rep.status == 'verified'


def test_a_move_that_cannot_be_made_while_holding_a_case_ends_with_the_case_held_for_a_person():
    import dataclasses
    # The righting tables lie beyond what this gantry can reach: the move to them is refused with the
    # case on the cups. The robot must not go on picking with a case still held.
    short = dataclasses.replace(CELL, gantry=dataclasses.replace(CELL.gantry, x_range=(-1.95, 1.62)))
    boxes = small_pile()
    boxes['case_06'] = OBB.make((A[0], A[1], TOP + H + D / 2), SKU.size,
                                quat_to_matrix((math.cos(math.pi / 4), math.sin(math.pi / 4), 0, 0)))
    be = FakeBackend(short, boxes)
    ev = EventLog('test')
    rep = Executive(short, be, ev, MissionConfig(SKU.id)).run()
    assert rep.status == 'interrupted'
    assert be.gripped() == ['case_06']
    assert len([e for e in ev.of('PICK_PLANNED') if e.data['case'] == 'case_06']) == 1


def test_a_case_hanging_tilted_on_the_cups_is_aimed_by_its_top_face_not_its_bottom():
    # Off-centre contents tip a held case a few degrees on the compliant cups. Its bottom face, seen
    # by the floor camera, is then displaced by height x tilt; on the pallet it levels again about
    # the cups. Aiming by the bottom face would leave it 10 mm off.
    from shinai_pi.layout import pallet_slots
    be = FakeBackend(CELL, small_pile(), hang_tilt_deg={'case_06': (2.5, -1.5)})
    ev = EventLog('test')
    Executive(CELL, be, ev, MissionConfig(SKU.id)).run()
    slot = pallet_slots(CELL, SKU, CELL.pallet.b_center)[[e.data for e in ev.of('PLACED') if e.data['case'] == 'case_06'][0]['slot']]
    err = np.linalg.norm(be.truth()['case_06'].center[:2] - np.asarray(slot.center[:2]))
    assert err < 0.004


def test_a_partly_hidden_case_is_not_corrected_by_sight():
    # When the gantry hides most of the hanging case, the visible rim gives the wrong size: the
    # look before release must then change nothing rather than act on a wrong measurement.
    import shinai_pi.executive as X
    from shinai_pi.perception.topface import TopFace
    real = X.measure_top_face

    def hidden(frame, key):
        m = real(frame, key)
        if m is None or frame.name != 'overhead_b':
            return m
        return TopFace(m.centre + 0.02, m.top_z, m.long_edge_yaw_deg + 30.0, m.tilt_deg, m.normal, (0.20, 0.12), m.n_points)

    X.measure_top_face = hidden
    try:
        be = FakeBackend(CELL, small_pile())
        ev = EventLog('test')
        Executive(CELL, be, ev, MissionConfig(SKU.id)).run()
    finally:
        X.measure_top_face = real
    assert not ev.of('AIMED')


def test_a_case_tilted_thirty_degrees_off_the_pallet_edge_is_gripped_square_to_its_face_and_set_upright():
    # A case slid half off pallet A: lower edge on the floor, bottom face on the pallet's edge, 30
    # degrees. Beyond what the cups alone can seal on (15 degrees): the wrist tilts the cups square
    # to its face, lifts it, and levels it in the air.
    tilt = math.radians(30)
    low_x = (W / 2) * math.cos(tilt) - (H / 2) * math.sin(tilt)
    low_z = -(W / 2) * math.sin(tilt) - (H / 2) * math.cos(tilt)
    edge_x = A[0] + CELL.pallet.size[0] / 2
    centre = (edge_x + TOP / math.tan(tilt) - low_x, A[1], -low_z)
    boxes = {'case_00': OBB.make(centre, SKU.size, quat_to_matrix((math.cos(tilt / 2), 0, math.sin(tilt / 2), 0)))}
    rep, ev, be = run(boxes)
    grip = ev.of('GRIP_OK')[0].data
    assert abs(grip['wrist_deg'][0]) > 10                      # the wrist met the face tilted
    assert 'case_00' in rep.placed and abs(standing_height(be.truth()['case_00']) - H) < 1e-6
    assert rep.status == 'verified'


def test_a_case_leaning_beyond_the_wrist_pitch_is_gripped_crosswise_and_taken_again_on_a_table():
    # 42 degrees is past the wrist's pitch (35): turned a quarter round, the plate meets the face with
    # the wrist's roll instead. Held across its length, the case is set on a table and taken again
    # along it before the tines go under (stage sweep envelope_B1: at 40-45 degrees every such case
    # was handed to a person without a try).
    tilt = math.radians(42)
    low_x = (W / 2) * math.cos(tilt) - (H / 2) * math.sin(tilt)
    low_z = -(W / 2) * math.sin(tilt) - (H / 2) * math.cos(tilt)
    edge_x = A[0] + CELL.pallet.size[0] / 2
    centre = (edge_x + TOP / math.tan(tilt) - low_x, A[1], -low_z)
    boxes = {'case_00': OBB.make(centre, SKU.size, quat_to_matrix((math.cos(tilt / 2), 0, math.sin(tilt / 2), 0)))}
    rep, ev, be = run(boxes)
    grip = ev.of('GRIP_OK')[0].data
    assert abs(grip['wrist_deg'][0]) < 1 and abs(grip['wrist_deg'][1]) > 35       # the roll met the face
    regrip = [e for e in ev.of('REGRIPPED') if e.data['case'] == 'case_00']
    assert regrip and '横向き' in regrip[0].data['reason']
    assert 'case_00' in rep.placed and abs(standing_height(be.truth()['case_00']) - H) < 1e-6
    assert rep.status == 'verified'


def test_the_tool_turns_a_quarter_when_only_the_wrist_roll_reaches_a_face():
    # 60 degrees across the long edge: the roll reaches it. 120 degrees (facing down): neither axis does.
    from shinai_pi.executive import Executive, MissionConfig
    ex = Executive(CELL, FakeBackend(CELL, {}), EventLog('test'), MissionConfig(SKU.id))
    n = np.array([math.sin(math.radians(60)), 0.0, math.cos(math.radians(60))])
    n2 = np.array([0.0, math.sin(math.radians(120)), math.cos(math.radians(120))])
    assert ex._grip_pose_for(n, 0.0) is not None
    assert ex._grip_pose_for(n2, 0.0) is None


def test_a_case_gripped_near_its_end_is_set_down_and_gripped_again_at_its_centre():
    # The pile estimate put case_06's centre 12 cm off, so the cups took it near one end. The floor
    # camera sees the offset: the case goes onto a staging table and is gripped again at its centre.
    import shinai_pi.executive as X
    real = X.Executive._perceive

    def off_estimate(self):
        bel = real(self)
        if bel is not None and 'case_06' in bel.boxes and not self.removed:
            b = bel.boxes['case_06']
            o = b.obb
            bel.boxes['case_06'] = X.Box('case_06', tuple(o.center + np.array([0.12, 0.0, 0.0])), tuple(o.size),
                                         confidence=b.confidence, rotation=o.rotation, top_visible=b.top_visible)
            bel._invalidate()
        return bel

    X.Executive._perceive = off_estimate
    try:
        rep, ev, be = run(small_pile())
    finally:
        X.Executive._perceive = real
    regrips = [e.data for e in ev.of('REGRIPPED')]
    assert regrips and regrips[0]['case'] == 'case_06' and regrips[0]['offset_mm'] > 40
    assert 'case_06' in rep.placed and rep.status == 'verified'


def test_a_case_beyond_the_gantry_is_left_to_a_person_with_that_reason():
    # One case was thrown beyond what the gantry reaches: the robot moves the rest and names why
    # it leaves that one, so the mission ends with a hand-over rather than unexplained.
    boxes = small_pile()
    boxes['case_09'] = OBB.make((CELL.gantry.x_range[0] + 0.02, A[1], H / 2), SKU.size)   # just past the reach, still in view
    rep, ev, _ = run(boxes, )
    assert rep.remaining.get('case_09', '').startswith('腕が届かない')
    assert rep.status == 'verified_with_handover'


def test_a_floor_camera_half_covered_by_a_fallen_case_gives_no_measurement_rather_than_a_wrong_one():
    # A case from the collapse lies over part of the floor camera: the carried case's bottom shows
    # only as a strip (seen in Isaac, transport seed 102). The strip must not be taken for the face.
    be = FakeBackend(CELL, small_pile())
    real = be.capture

    def covered(camera):
        f = real(camera)
        if camera == 'under':
            f.depth[:, f.width // 2:] = np.inf               # the right half of the picture is blocked
            f.instance[:, f.width // 2:] = 0
        return f

    be.capture = covered
    ev = EventLog('test')
    rep = Executive(CELL, be, ev, MissionConfig(SKU.id)).run()
    under = [e.data for e in ev.of('OBSERVED') if e.data['view'] == 'under']
    assert under and not any(u['measured'] for u in under)
    assert rep.status == 'verified'                        # placed by the pile estimate, checked on B


def test_a_case_that_cannot_be_measured_from_below_is_set_down_and_taken_again_by_its_top_face():
    # The floor camera is covered (a fallen case lies on it). Placing by the pile estimate alone
    # dropped a case off the stack in Isaac; instead the case goes onto a staging table, where its
    # top face is measured and it is gripped at its centre before it goes to B.
    be = FakeBackend(CELL, small_pile())
    real = be.capture

    def covered(camera):
        f = real(camera)
        if camera == 'under':
            f.depth[:] = np.inf
            f.instance[:] = 0
        return f

    be.capture = covered
    ev = EventLog('test')
    rep = Executive(CELL, be, ev, MissionConfig(SKU.id)).run()
    assert len(ev.of('REGRIPPED')) == len(rep.placed) == 7
    assert rep.status == 'verified'


# ---------------------------------------------------------------- the under-support hand (requirements v05)
def _fork_run(**kw):
    be = FakeBackend(CELL, small_pile(), **kw)
    ev = EventLog('test')
    rep = Executive(CELL, be, ev, MissionConfig(SKU.id)).run()
    return rep, ev, be


def test_an_upright_case_rides_on_the_tines_which_come_out_before_it_is_set_down():
    rep, ev, be = _fork_run()
    assert rep.status == 'verified', rep
    for key in rep.placed:
        kinds = [e.type for e in ev.events if e.data.get('case') == key and e.type in ('FORK_IN', 'FORK_LANDED', 'FORK_OUT', 'PLACED')]
        assert kinds[:4] == ['FORK_IN', 'FORK_LANDED', 'FORK_OUT', 'PLACED'], (key, kinds)
        fin = next(e.data for e in ev.events if e.type == 'FORK_IN' and e.data['case'] == key)
        assert fin['accepted'] and fin['share'] >= 0.7
    assert be.fork['ins'] == 0.0 and not be.fork['engaged']          # retracted at the end


def test_each_change_of_what_holds_the_case_is_marked_so_the_simulator_trace_can_be_cut_by_it():
    # The trace of every physics step is read by these phases: the tines take the case, give it back to the
    # cups, come out, and the cups let go once the case is felt on the surface (or are said not to have felt it).
    rep, ev, be = _fork_run()
    assert rep.placed
    for key in rep.placed:
        kinds = [e.type for e in ev.events if e.data.get('case') == key and e.type in (
            'FORK_LANDED', 'FORK_UNLOADING', 'FORK_RETRACTING', 'FORK_OUT', 'RELEASED', 'PLACED')]
        assert kinds[-6:] == ['FORK_LANDED', 'FORK_UNLOADING', 'FORK_RETRACTING', 'FORK_OUT', 'RELEASED', 'PLACED'], (key, kinds)
        rel = [e.data for e in ev.events if e.type == 'RELEASED' and e.data['case'] == key][-1]
        assert rel['felt_landing'] is True and 'how' in rel


def test_the_tines_come_out_on_a_side_no_earlier_case_occupies():
    rep, ev, be = _fork_run()
    from shinai_pi.layout import pallet_slots
    slots = pallet_slots(CELL, SKU, CELL.pallet.b_center)
    placed_slots = []
    for e in ev.events:
        if e.type == 'FORK_OUT':
            d = np.asarray(e.data['bar_direction'])
            s = slots[e.data['slot']]
            for other in placed_slots:
                off = np.asarray(other.center[:2]) - np.asarray(s.center[:2])
                if other.layer == s.layer and np.linalg.norm(off) < 0.45:
                    assert off @ d < 0.05, (e.data, other)               # nothing already placed on the bar side
        if e.type == 'PLACED':
            placed_slots.append(slots[e.data['slot']])


def test_without_the_tines_a_case_is_carried_slowly_on_the_cups():
    # Something stops the tines sliding in under the first case: it is carried on the cups alone,
    # at the slow acceleration (interlock), and still verified on B.
    be = FakeBackend(CELL, small_pile(), fork_blocks={'insert'})
    speeds = []
    orig = be.move

    def spy(goal, monitor, carrying=None, speed=1.0, **kw):
        if carrying:
            speeds.append((carrying, speed))
        return orig(goal, monitor, carrying=carrying, speed=speed, **kw)
    be.move = spy
    ev = EventLog('test')
    rep = Executive(CELL, be, ev, MissionConfig(SKU.id)).run()
    first = rep.placed[0]
    skipped = [e.data for e in ev.of('FORK_SKIPPED') if e.data['case'] == first]
    assert skipped and skipped[0]['reason']
    assert all(sp < 1.0 for k, sp in speeds if k == first)
    assert rep.status == 'verified'


def _carry_log(be):
    """Every move made while holding a case: (case, start q, goal, tines inserted, tines pressed up)."""
    log, orig = [], be.move

    def spy(goal, monitor, carrying=None, speed=1.0, **kw):
        if carrying:
            log.append((carrying, np.array(be.q[:4], float), np.array(goal[:4], float), be.fork['ins'] > 0, be.fork['engaged']))
        return orig(goal, monitor, carrying=carrying, speed=speed, **kw)
    be.move = spy
    return log


def test_the_tines_are_under_the_case_before_it_travels_sideways():
    # Review 2026-10-31: the tines are there to keep a case from falling when the travel shakes it, so they go in
    # right after the lift above A, not after a carry on the cups alone to the floor camera (seed 118: 1.4 m).
    be = FakeBackend(CELL, small_pile())
    log = _carry_log(be)
    ev = EventLog('test')
    rep = Executive(CELL, be, ev, MissionConfig(SKU.id)).run()
    assert rep.status == 'verified', rep
    forked = {e.data['case'] for e in ev.of('FORK_IN') if e.data['accepted']}
    assert forked == set(rep.placed)
    for case, q0, goal, inserted, _ in log:
        if case in forked and float(np.linalg.norm(goal[:2] - q0[:2])) > 0.01 and goal[2] >= q0[2] - 1e-6:
            assert inserted, (case, q0, goal)


def test_a_held_case_is_turned_with_the_tines_under_it_but_not_pressing():
    # The heading turn to the slot is made over A, with the tines under the case as a catch: pressed up, the
    # case is clamped, and a clamped case is not turned (the cups let go as a turn began, stage gates).
    be = FakeBackend(CELL, small_pile())
    log = _carry_log(be)
    ev = EventLog('test')
    rep = Executive(CELL, be, ev, MissionConfig(SKU.id)).run()
    assert rep.status == 'verified', rep
    turns = [(c, inserted, pressed) for c, q0, goal, inserted, pressed in log
             if abs(goal[3] - q0[3]) > math.radians(10) and float(np.linalg.norm(goal[:2] - q0[:2])) <= 0.01]
    assert turns
    for case, inserted, pressed in turns:
        assert inserted and not pressed, case


def test_the_case_is_still_measured_from_below_with_the_tines_in():
    rep, ev, be = _fork_run()
    assert rep.status == 'verified'
    measured = [e for e in ev.of('OBSERVED') if e.data.get('view') == 'under' and e.data.get('measured')]
    assert len(measured) >= len(rep.placed)


def test_tines_that_will_not_come_out_keep_the_case_held_for_a_person():
    rep, ev, be = _fork_run(fork_blocks={'retract': 2})                # still held after the one retry
    assert rep.status != 'verified'
    assert be.attached is not None                                    # never let go with the tines in
    assert any('爪' in e.data.get('reason', '') for e in ev.of('HANDOVER'))


def test_tines_held_back_once_come_out_after_the_cups_lift_the_case_a_little_more():
    # S3 seeds 102 and 105: on the second layer a pull met resistance now and then (in a rerun of 102
    # every pull stayed under 7 N). The cups lift the case 3 mm more and the tines are pulled once again.
    rep, ev, be = _fork_run(fork_blocks={'retract': 1})
    retried = ev.of('FORK_RETRIED')
    assert len(retried) == 1 and len(retried[0].data['residual_N']) == 2
    assert rep.status == 'verified'


# ---------------------------------------------------------------- watching the path of the cups (requirements v05 R4)
def test_an_object_in_the_cups_path_stops_the_approach_before_it_is_touched():
    # A bar hangs 15 mm above the top case, inside the plate's footprint: the downward camera at the
    # plate centre sees it from above, before the tool goes down, and the cups never reach the case under it.
    boxes = small_pile()
    top = boxes['case_06']
    be = FakeBackend(CELL, boxes)
    c = top.center
    be.static['bar'] = OBB.make((c[0], c[1], c[2] + H / 2 + 0.015 + 0.01), (0.40, 0.03, 0.02))
    ev = EventLog('test')
    rep = Executive(CELL, be, ev, MissionConfig(SKU.id, max_cycles=6)).run()
    near = [e.data for e in ev.of('NEAR_MISS') if e.data['case'] == 'case_06']
    assert near and near[0]['what'] == 'bar'
    assert not [e for e in ev.of('GRIP_OK') if e.data['case'] == 'case_06']
    assert 'case_06' not in rep.placed


def test_an_ordinary_pile_raises_no_near_miss():
    rep, ev, be = run(small_pile())
    assert rep.status == 'verified' and not ev.of('NEAR_MISS')


def test_vacuum_lost_while_the_tines_carry_the_case_holds_reseals_and_finishes_slowly():
    # Design 9: a vacuum drop with the tines under the case is not a dropped case. Stop holding,
    # seal again, and carry the rest of the way at the slow acceleration (the pads alone are slippery).
    be = FakeBackend(CELL, small_pile(), vacuum_drops={'case_06': 1})
    speeds = []
    orig = be.move

    def spy(goal, monitor, carrying=None, speed=1.0, **kw):
        if carrying == 'case_06':
            speeds.append((be.t, speed))
        return orig(goal, monitor, carrying=carrying, speed=speed, **kw)
    be.move = spy
    ev = EventLog('test')
    rep = Executive(CELL, be, ev, MissionConfig(SKU.id)).run()
    lost = ev.of('VACUUM_LOST')
    assert lost and lost[0].data['case'] == 'case_06' and sum(lost[0].data['tines_N']) > 0
    assert not ev.of('DROP_DETECTED') and rep.safety_stops == 0
    assert 'case_06' in rep.placed and rep.status == 'verified'
    assert all(sp < 1.0 for t, sp in speeds if t > lost[0].t)


def test_the_tool_turns_the_short_way_to_face_the_bars_to_the_open_side():
    rep, ev, be = _fork_run()
    from shinai_pi.layout import pallet_slots
    for e in ev.of('FORK_OUT'):
        if e.data.get('slot') is None:
            continue
        assert abs(e.data['turned_rad']) <= math.pi + 1e-3          # recorded to three decimals


def test_vacuum_lost_with_the_tines_not_carrying_the_case_is_held_where_it_is():
    # The cups let go, but the tines read almost nothing: the case is not resting on them. Moving on
    # would be guessing; it stays where it is for a person (design 9, review 3).
    be = FakeBackend(CELL, small_pile(), vacuum_drops={'case_06': 1}, unsupported_drops={'case_06'})
    ev = EventLog('test')
    rep = Executive(CELL, be, ev, MissionConfig(SKU.id)).run()
    assert ev.of('VACUUM_LOST') and rep.status == 'interrupted'
    assert any('爪が重さを受けていない' in e.data['reason'] for e in ev.of('HANDOVER'))
    assert 'case_06' not in rep.placed


def test_a_second_vacuum_loss_in_one_carry_is_not_resealed_again():
    be = FakeBackend(CELL, small_pile(), vacuum_drops={'case_06': 2})
    ev = EventLog('test')
    rep = Executive(CELL, be, ev, MissionConfig(SKU.id)).run()
    assert len([e for e in ev.of('VACUUM_LOST') if e.data['case'] == 'case_06']) == 2
    assert any('2 回目' in e.data['reason'] for e in ev.of('HANDOVER'))
    assert rep.status == 'interrupted'


def test_a_contact_force_over_its_limit_leaves_the_run_unverified_for_quality():
    from shinai_pi.backend import MoveResult
    be = FakeBackend(CELL, small_pile())
    orig = be.move

    def heavy(goal, monitor, carrying=None, speed=1.0, stop_on_contact=False, **kw):
        r = orig(goal, monitor, carrying=carrying, speed=speed, stop_on_contact=stop_on_contact, **kw)
        if r.contact and carrying is None:
            r.info = {'kind': 'touch', 'static_N': CELL.contact.touch_static * 2, 'peak_N': 80.0}
        return r
    be.move = heavy
    ev = EventLog('test')
    rep = Executive(CELL, be, ev, MissionConfig(SKU.id)).run()
    risks = ev.of('DAMAGE_RISK')
    assert risks and risks[0].data['kind'] == 'touch'
    assert rep.status == 'moved_quality_unverified'
    assert ev.events[-1].data['damage_risk'] == len(set(e.data['case'] for e in risks))


def test_a_grip_that_failed_after_pressing_too_hard_is_not_tried_again():
    # envelope_B1/B2, a 45 degree case: each failed grip was retried, and each retry pressed harder
    # (70 N, then 87 N, over the 50 N limit). After one over-limit touch the case goes to a person.
    be = FakeBackend(CELL, small_pile(), grip_failures={'case_06': 3})
    orig = be.move

    def heavy(goal, monitor, carrying=None, speed=1.0, stop_on_contact=False, **kw):
        r = orig(goal, monitor, carrying=carrying, speed=speed, stop_on_contact=stop_on_contact, **kw)
        if r.contact and carrying is None:
            r.info = {'kind': 'touch', 'static_N': CELL.contact.touch_static * 1.4, 'peak_N': 80.0}
        return r
    be.move = heavy
    ev = EventLog('test')
    Executive(CELL, be, ev, MissionConfig(SKU.id)).run()
    tries = [e for e in ev.of('GRIP_FAILED') if e.data['case'] == 'case_06']
    assert len(tries) == 1
    assert any(e.data['case'] == 'case_06' and '押し付け' in e.data['reason'] for e in ev.of('HANDOVER'))


# --- the tines stand out 0.22 m past the plate: lowering beside a stack, the tool turns so they miss it
# (S3 102, 2026-10-18: lowered beside pallet B, the tines pushed B's second layer 33 cm)

FLOOR_X = 0.0                   # midway between the pallets: the posts at +-0.33 stay clear of both


def _post(side):
    # a tall object beside a case standing on the floor, where the tines of one of the two grip yaws go down
    return {'post': OBB.make((FLOOR_X + side * 0.33, 0.0, 0.25), (0.12, 0.40, 0.50))}


def _on_floor():
    return {'case_00': OBB.make((FLOOR_X, 0.0, H / 2), SKU.size, yaw_matrix(math.pi / 2))}


@pytest.mark.parametrize('side', [1, -1])
def test_the_tool_turns_half_round_so_the_tines_go_down_away_from_a_stack_beside_the_case(side):
    be = FakeBackend(CELL, _on_floor(), obstacles=_post(side))
    ex = Executive(CELL, be, EventLog('test'), MissionConfig(SKU.id, expected_cases=1))
    toward = side * math.pi / 2                                    # the yaw whose tines point at the post
    out = ex._grip_from_top('case_00', (FLOOR_X, 0.0), H - CELL.gantry.origin_height, toward, CELL.gantry.z_range[1])
    assert out == 'held' and be.fork_hits == []
    yaw = ex.grip_yaw['case_00']
    assert math.sin(yaw) * side < 0                                # the tool's -y, where the tines are, points away


def test_with_stacks_on_both_sides_the_tool_is_not_lowered_and_a_person_is_asked():
    be = FakeBackend(CELL, _on_floor(), obstacles={**_post(1), 'post2': _post(-1)['post']})
    ev = EventLog('test')
    rep = Executive(CELL, be, ev, MissionConfig(SKU.id, expected_cases=1)).run()
    assert be.fork_hits == []
    near = [e.data for e in ev.of('NEAR_MISS') if e.data['case'] == 'case_00']
    assert near and '爪' in near[0]['reason'] and near[0]['what'] in ('post', 'post2')
    assert [c['what'] for c in near[0]['candidates']] in (['post', 'post2'], ['post2', 'post'])   # each yaw, and what closed it
    assert 'case_00' not in rep.placed and not ev.of('GRIP_OK')


def test_the_fake_cell_records_tines_driven_into_something():
    # the check above means nothing unless the fake cell notices tines in a box
    be = FakeBackend(CELL, _on_floor(), obstacles=_post(1))
    from shinai_pi.safety import SafetySupervisor
    be.move([FLOOR_X, 0.0, 0.35 - CELL.gantry.origin_height, math.pi / 2, 0.0, 0.0], SafetySupervisor(CELL).check)
    assert be.fork_hits and be.fork_hits[0][1] == 'post'


# --- a backstop for what the camera cannot see: the empty tool already pressing on something when a touch
# search starts (S3 102: -3,275 before the search began, read afterwards as a 3,000 N press on the case)

def _single_on_floor(**kw):
    be = FakeBackend(CELL, _on_floor(), **kw)
    ev = EventLog('test')
    rep = Executive(CELL, be, ev, MissionConfig(SKU.id, expected_cases=1)).run()
    return rep, ev, be


def test_a_touch_search_that_starts_pressed_against_something_is_a_collision_not_a_press_on_the_case():
    rep, ev, be = _single_on_floor(collisions={'touch': 1})
    hit = ev.of('COLLISION')
    assert len(hit) == 1 and hit[0].data['case'] == 'case_00' and hit[0].data['kind'] == 'preload'
    assert not ev.of('DAMAGE_RISK')                     # not charged to the case: the force was on something else
    assert not ev.of('GRIP_OK') and 'case_00' not in rep.placed
    assert '当たった' in rep.remaining['case_00']


def test_the_empty_tool_meeting_resistance_while_going_down_stops_and_does_not_grip():
    rep, ev, be = _single_on_floor(collisions={'free': 1})
    hit = ev.of('COLLISION')
    assert len(hit) == 1 and hit[0].data['kind'] == 'collision' and hit[0].data['load_N'] > CELL.contact.collision
    assert not ev.of('GRIP_OK') and 'case_00' not in rep.placed
    assert rep.safety_stops == 0                         # a stop of its own, not a light-curtain stop to be reset


def test_without_a_collision_the_same_case_is_taken():
    rep, ev, be = _single_on_floor()
    assert not ev.of('COLLISION') and 'case_00' in rep.placed


def test_after_turning_half_round_the_wrist_still_meets_a_tilted_face_square():
    # S3 103 (2026-10-18): the tool turned half round to keep its tines off a stack, but went down with the
    # wrist tilt worked out for the first yaw: the plate met the face tilted the wrong way and struck the case.
    tilt = math.radians(20)
    rot = quat_to_matrix((math.cos(tilt / 2), math.sin(tilt / 2), 0.0, 0.0)) @ yaw_matrix(math.pi / 2)
    case = OBB.make((FLOOR_X, 0.0, 0.16), SKU.size, rot)
    be = FakeBackend(CELL, {'case_00': case}, obstacles=_post(1))
    ex = Executive(CELL, be, EventLog('test'), MissionConfig(SKU.id, expected_cases=1))
    c, n, _, _ = case.top_face()
    seen = []
    real = be.grip

    def grip(close):
        if close:
            seen.append(float(tool_rotation(be.q)[:, 2] @ n))
        return real(close)
    be.grip = grip
    out = ex._grip_from_top('case_00', c, float(c[2]) - CELL.gantry.origin_height, math.pi / 2, CELL.gantry.z_range[1], normal=n)
    assert out == 'held' and be.fork_hits == []
    assert math.sin(ex.grip_yaw['case_00']) < 0                    # turned: the tines point away from the post
    assert seen and min(seen) > math.cos(math.radians(2))          # the plate square to the face when the cups seal


# --- a case whose tines are blocked by another case waits for that case to be taken, then is taken itself
# (S3 103, 2026-10-18: case_13 was blocked by case_12 on both yaws and handed to a person; case_12 was
# taken next, but case_13 was never tried again, and the cases under it were left as well)

def _standing_beside(side):
    # a case standing on end beside the floor case, taller than the tines when the tool reaches the floor case
    rot = quat_to_matrix((math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0))
    return OBB.make((FLOOR_X + side * 0.28, 0.0, SKU.size[0] / 2), SKU.size, rot)


def _blocked_by_a_case(**kw):
    be = FakeBackend(CELL, {**_on_floor(), 'case_01': _standing_beside(1)}, obstacles=_post(-1), **kw)
    ev = EventLog('test')
    ex = Executive(CELL, be, ev, MissionConfig(SKU.id, expected_cases=2))
    choose = ex._choose

    def floor_case_first():
        # in 103 the order sent the tool to the blocked case twice first; here the taller case would go first
        c = choose()
        if c[0] == 'pick' and ex.blocked.get('case_00', 0) < 2 and c[1].box_id != 'case_00':
            first = [k for k in ex.belief.candidates() if k.box_id == 'case_00']
            return (c[0], first[0], c[2]) if first else c
        return c
    ex._choose = floor_case_first
    return ex.run(), ev, be


def test_a_case_blocked_by_another_case_is_taken_after_that_case_is_gone():
    rep, ev, be = _blocked_by_a_case()
    near = [e.data for e in ev.of('NEAR_MISS') if e.data['case'] == 'case_00']
    assert near and 'case_01' in [c['what'] for c in near[0]['candidates']]      # it was tried first, and blocked
    assert be.fork_hits == []
    assert {'case_00', 'case_01'} <= set(rep.placed)
    assert not [e for e in ev.of('HANDOVER') if e.data['case'] == 'case_00']


def test_a_case_waiting_for_another_case_says_so_rather_than_handing_over():
    rep, ev, be = _blocked_by_a_case()
    waits = [e.data for e in ev.of('DECISION_DEFERRED') if e.data.get('case') == 'case_00']
    assert waits and waits[0]['waiting_for'] == ['case_01']


def test_a_case_still_waiting_when_the_other_cannot_be_taken_goes_to_a_person():
    rep, ev, be = _blocked_by_a_case(grip_failures={'case_01': 3})
    assert 'case_01' not in rep.placed and 'case_00' not in rep.placed
    assert [e for e in ev.of('HANDOVER') if e.data['case'] == 'case_00']
    assert '人が確認' in rep.remaining['case_00']


def test_a_tilted_face_is_met_with_the_wrist_rolled_the_way_that_keeps_the_bars_off_the_ram():
    # review 2026-10-31: rolled towards -y while pitched, the drawn-up bars lean into the ram. A half turn of the tool
    # meets the same face with the opposite roll.
    from shinai_pi import structure
    ex = Executive(CELL, FakeBackend(CELL, small_pile()), EventLog('test'), MissionConfig(SKU.id))
    n = np.array([0.2, 0.2, 0.0])
    n[2] = math.sqrt(1 - n[0] ** 2 - n[1] ** 2)
    yaw, (pitch, roll), crosswise = ex._grip_pose_for(n, 0.0)
    assert structure.wrist_allowed(pitch, roll)
    assert abs(abs(yaw) - math.pi) < 1e-9 and not crosswise


def _model_file(tmp_path, bias):
    from shinai_pi.learning.cascade import FEATURES, CascadeModel
    n = len(FEATURES)
    path = tmp_path / 'model.json'
    path.write_text(CascadeModel(np.zeros(n), np.ones(n), np.zeros(n), bias, {}).to_json(), encoding='utf-8')
    return str(path)


def test_with_a_learned_model_each_pick_records_what_the_rule_alone_would_have_taken(tmp_path):
    be = FakeBackend(CELL, small_pile())
    ev = EventLog('test')
    ex = Executive(CELL, be, ev, MissionConfig(SKU.id, cascade_model=_model_file(tmp_path, -40.0)))
    rep = ex.run()
    picks = ev.of('PICK_PLANNED')
    assert picks and all({'rule_choice', 'rule_risk', 'model_changed_choice'} <= set(e.data) for e in picks)
    assert rep.learned_decisions == len(picks)
    assert rep.learned_changed == sum(e.data['model_changed_choice'] for e in picks)


def test_without_a_learned_model_no_comparison_is_claimed():
    rep, ev, _ = run(small_pile())
    assert all('model_changed_choice' not in e.data for e in ev.of('PICK_PLANNED'))
    assert rep.learned_decisions == 0 and rep.learned_changed == 0


def test_a_look_builds_the_belief_from_the_cameras_alone():
    boxes = small_pile()
    be = FakeBackend(CELL, boxes)
    ex = Executive(CELL, be, EventLog('test'), MissionConfig(SKU.id))
    bel = ex.look(('overhead_a',))
    assert bel is ex.belief and set(bel.boxes) <= set(boxes)
    assert 'case_06' in bel.boxes                                    # the top case is in plain view from above
    assert ex.report.observations == 1


def test_a_look_needs_a_camera():
    ex = Executive(CELL, FakeBackend(CELL, small_pile()), EventLog('test'), MissionConfig(SKU.id))
    with pytest.raises(ValueError):
        ex.look(())
