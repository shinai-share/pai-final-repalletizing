"""The scorer matches each placed case to the slot its PLACED event names, not to its order."""
import numpy as np

from shinai_pi.config import load_cell
from shinai_pi.evaluation.harness import TruthRecorder
from shinai_pi.events import EventLog
from shinai_pi.executive import MissionReport
from shinai_pi.layout import pallet_slots

CELL = load_cell()
SKU = CELL.sku('BEV-500x24')
SLOTS = pallet_slots(CELL, SKU, CELL.pallet.b_center)


def test_placement_error_uses_the_slot_named_in_the_event_when_a_slot_is_skipped():
    final = {'c0': (np.array(SLOTS[0].center), (1, 0, 0, 0)), 'c2': (np.array(SLOTS[2].center), (1, 0, 0, 0))}
    rec = TruthRecorder(lambda: final, CELL, SLOTS)
    ev = EventLog('t')
    ev.subscribe(rec.on_event)
    ev.emit(1.0, 'PLACED', case='c0', slot=0, layer=0)
    ev.emit(2.0, 'PLACED', case='c2', slot=2, layer=0)          # slot 1 was skipped
    out = rec.summary(MissionReport('incomplete', placed=['c0', 'c2']), set())
    assert out['placement_xy_err_max_m'] < 1e-9
    assert out['final_positions']['c2'] == [round(float(v), 3) for v in SLOTS[2].center]


def test_a_placed_case_without_an_event_is_not_scored_against_a_guessed_slot():
    final = {'c0': (np.array(SLOTS[5].center), (1, 0, 0, 0))}
    rec = TruthRecorder(lambda: final, CELL, SLOTS)
    out = rec.summary(MissionReport('incomplete', placed=['c0']), set())
    assert out['placement_xy_err_max_m'] == 0.0 and out['placed_unscored'] == ['c0']


def _yaw_quat(deg):
    h = np.radians(deg) / 2
    return (float(np.cos(h)), 0.0, 0.0, float(np.sin(h)))


def test_true_yaw_error_is_measured_against_the_slot_and_ignores_the_half_turn_symmetry():
    s = SLOTS[3]
    final = {'a': (np.array(s.center), _yaw_quat(s.yaw_deg + 1.5)),
             'b': (np.array(SLOTS[4].center), _yaw_quat(SLOTS[4].yaw_deg + 180.0 - 0.5))}
    rec = TruthRecorder(lambda: final, CELL, SLOTS)
    ev = EventLog('t')
    ev.subscribe(rec.on_event)
    ev.emit(1.0, 'PLACED', case='a', slot=3, layer=0)
    ev.emit(2.0, 'PLACED', case='b', slot=4, layer=0)
    out = rec.summary(MissionReport('incomplete', placed=['a', 'b']), set())
    assert abs(out['placement_errors']['a']['yaw_deg'] - 1.5) < 1e-3
    assert abs(out['placement_errors']['b']['yaw_deg'] - 0.5) < 1e-3
    assert abs(out['placement_yaw_max_deg'] - 1.5) < 1e-2


def test_a_quarter_turned_case_reports_a_large_yaw_error():
    s = SLOTS[0]
    final = {'a': (np.array(s.center), _yaw_quat(s.yaw_deg + 90.0))}
    rec = TruthRecorder(lambda: final, CELL, SLOTS)
    ev = EventLog('t')
    ev.subscribe(rec.on_event)
    ev.emit(1.0, 'PLACED', case='a', slot=0, layer=0)
    out = rec.summary(MissionReport('incomplete', placed=['a']), set())
    assert abs(out['placement_errors']['a']['yaw_deg'] - 90.0) < 1e-3


# --- every case of the load ends in exactly one state (S3 review 2026-10-18: 404 reported 15 placed and
# reasons for 3, the other 6 could not be told from the report)

A = CELL.pallet.a_center
TOP_A = CELL.pallet.size[2]
ON_A = np.array([A[0], A[1], TOP_A + SKU.size[2] / 2])
BESIDE_A = np.array([A[0] - 0.9, A[1], SKU.size[2] / 2])           # spilled on the floor, inside the cell
OUTSIDE = np.array([A[0], CELL.gantry.y_range[1] + 0.8, 0.11])     # beyond the floor the gantry covers
Q0 = (1, 0, 0, 0)


def _ledger(start, final, report, events=(), held=()):
    rec = TruthRecorder(lambda: start, CELL, SLOTS)
    rec.truth_fn = lambda: final
    ev = EventLog('t')
    ev.subscribe(rec.on_event)
    for t, kind, data in events:
        ev.emit(t, kind, **data)
    out = rec.summary(report, set(start), held=held)
    return out, out['case_ledger']


def test_every_case_is_counted_once_and_the_states_add_up_to_the_load():
    start = {'p': (ON_A, Q0), 'h': (ON_A, Q0), 'g': (ON_A, Q0), 'o': (OUTSIDE, Q0), 'r': (BESIDE_A, Q0), 'u': (ON_A, Q0)}
    final = dict(start, p=(np.array(SLOTS[0].center), Q0), h=(np.array([0.95, 0.88, 0.11]), Q0), g=(np.array([0.5, 0.0, 1.0]), Q0))
    report = MissionReport('unresolved', placed=['p'], handed_over=['h'], remaining={'r': '腕が届かない場所にある（人が取り出す）'})
    out, led = _ledger(start, final, report, [(1.0, 'PLACED', {'case': 'p', 'slot': 0, 'layer': 0})], held=['g'])
    assert {k: v['state'] for k, v in led.items()} == {'p': 'placed_on_b', 'h': 'handed_over', 'g': 'held_by_robot',
                                                       'o': 'outside_cell', 'r': 'left_in_cell', 'u': 'left_in_cell'}
    assert sum(out['case_states'].values()) == len(start) == 6
    assert led['r']['detail'] == '腕が届かない場所にある（人が取り出す）'
    assert led['u']['detail'] == '未処理'


def test_a_case_off_the_pallet_before_the_robot_moved_is_told_apart_from_one_pushed_off_by_a_pick():
    start = {'t': (BESIDE_A, Q0), 'k': (ON_A, Q0), 'n': (ON_A + [0.4, 0, 0], Q0)}
    final = dict(start, k=(OUTSIDE, Q0), n=(np.array(SLOTS[0].center), Q0))
    events = [(1.0, 'PICK_PLANNED', {'case': 'n'}), (2.0, 'PLACED', {'case': 'n', 'slot': 0, 'layer': 0})]
    rec = TruthRecorder(lambda: start, CELL, SLOTS)
    ev = EventLog('t')
    ev.subscribe(rec.on_event)
    ev.emit(*events[0][:2], **events[0][2])
    rec.truth_fn = lambda: final                                     # k is pushed off while n is taken
    ev.emit(1.5, 'PILE_CHANGED', removed='n')
    ev.emit(*events[1][:2], **events[1][2])
    led = rec.summary(MissionReport('unresolved', placed=['n']), set(start))['case_ledger']
    assert led['t']['left_pallet'] == 'before_mission'
    assert led['k']['left_pallet'] == 'during_pick:n' and led['k']['state'] == 'outside_cell'
    assert led['n']['left_pallet'] is None                           # moved by the robot on purpose


def test_a_placed_case_carries_its_own_quality_judged_at_the_placement_tolerances():
    s0, s1 = np.array(SLOTS[0].center), np.array(SLOTS[1].center)
    start = {'a': (ON_A, Q0), 'b': (ON_A, Q0), 'c': (ON_A, Q0)}
    final = {'a': (s0 + [0.0099, 0, 0], _yaw_quat(SLOTS[0].yaw_deg)), 'b': (s1 + [0.0101, 0, 0], _yaw_quat(SLOTS[1].yaw_deg)),
             'c': (s1, Q0)}
    events = [(1.0, 'PLACED', {'case': 'a', 'slot': 0, 'layer': 0}), (2.0, 'PLACED', {'case': 'b', 'slot': 1, 'layer': 0})]
    out, led = _ledger(start, final, MissionReport('unresolved', placed=['a', 'b', 'c']), events)
    assert led['a']['quality'] == 'ok' and led['b']['quality'] == 'ng'
    assert led['c']['quality'] == 'unscored'                         # no PLACED event: not judged against a guessed slot
    assert out['case_states']['placed_on_b'] == 3 and out['placed_quality'] == {'ok': 1, 'ng': 1, 'unscored': 1}


def test_a_case_held_while_waiting_for_a_person_is_not_also_counted_as_handed_over():
    start = {'g': (ON_A, Q0)}
    final = {'g': (np.array([0.5, 0.0, 1.0]), Q0)}
    _, led = _ledger(start, final, MissionReport('interrupted', handed_over=['g']), held=['g'])
    assert led['g']['state'] == 'held_by_robot'


def test_the_start_of_the_mission_is_kept_for_every_case():
    start = {'a': (ON_A, Q0), 'b': (BESIDE_A, Q0)}
    out, _ = _ledger(start, start, MissionReport('unresolved'))
    assert out['start_positions'] == {k: [round(float(v), 3) for v in p] for k, (p, _) in start.items()}


def _settled_in_order(steps):
    """steps: (case settling, {case: xy}) - the truth as each PLACEMENT_SETTLED is emitted."""
    world = {}
    rec = TruthRecorder(lambda: {k: (np.array([x, y, 0.3]), (1, 0, 0, 0)) for k, (x, y) in world.items()}, CELL, SLOTS)
    ev = EventLog('t')
    ev.subscribe(rec.on_event)
    for t, (case, xy) in enumerate(steps):
        world.update(xy)
        ev.emit(float(t), 'PLACEMENT_SETTLED', case=case, ok=True, drift_mm=0.0, lean_deg=0.0, waited_s=3.0)
    return rec, world


def test_a_placed_case_pushed_while_its_neighbour_is_placed_is_counted_even_below_the_pick_disturbance():
    rec, _ = _settled_in_order([('a', {'a': (1.0, 1.0)}), ('b', {'b': (1.4, 1.0), 'a': (1.0, 1.015)})])
    out = rec.summary(MissionReport('verified'), set())
    assert out['placed_cases_pushed'] == [{'case': 'a', 'while_placing': 'b', 'moved_mm': 15.0}]


def test_a_pushed_case_put_right_is_measured_again_from_where_it_was_put_back():
    rec, _ = _settled_in_order([('a', {'a': (1.0, 1.0)}), ('b', {'b': (1.4, 1.0), 'a': (1.0, 1.015)}),
                                ('a', {'a': (1.0, 1.0002)}), ('c', {'c': (1.8, 1.0)})])
    out = rec.summary(MissionReport('verified'), set())
    assert [p['case'] for p in out['placed_cases_pushed']] == ['a']        # put back, then left alone


def test_a_millimetre_of_settling_is_not_a_push_but_a_push_after_the_last_placement_is():
    rec, world = _settled_in_order([('a', {'a': (1.0, 1.0)}), ('b', {'b': (1.4, 1.0), 'a': (1.0, 1.001)})])
    assert rec.summary(MissionReport('verified'), set())['placed_cases_pushed'] == []
    world['b'] = (1.4, 1.008)
    assert rec.summary(MissionReport('verified'), set())['placed_cases_pushed'] == [
        {'case': 'b', 'while_placing': None, 'moved_mm': 8.0}]


def test_a_case_that_left_pallet_a_in_transport_carries_the_moment_it_left():
    from shinai_pi.evaluation.harness import PalletExitClock
    clock = PalletExitClock(CELL)
    clock.tick(0.5, {'a': (ON_A, Q0), 'b': (ON_A, Q0)})
    clock.tick(1.2, {'a': (BESIDE_A, Q0), 'b': (ON_A, Q0)})
    clock.tick(1.3, {'a': (OUTSIDE, Q0), 'b': (ON_A, Q0)})        # the first moment is kept
    assert clock.left == {'a': 1.2}


def test_where_an_unplaced_case_is_tells_pallet_work_area_and_whether_the_robot_saw_it_apart():
    start = {'f': (BESIDE_A, Q0), 'o': (OUTSIDE, Q0), 'u': (OUTSIDE, Q0)}
    rec = TruthRecorder(lambda: start, CELL, SLOTS, left_in_transport={'f': 0.8, 'o': 1.1, 'u': 2.4})
    led = rec.summary(MissionReport('unresolved', remaining={'f': '腕が届かない場所にある（人が取り出す）'}), {'f', 'o'})['case_ledger']
    assert led['f']['where'] == {'off_pallet_a': True, 'outside_work_area': False, 'seen_by_robot': True}
    assert led['o']['where'] == {'off_pallet_a': True, 'outside_work_area': True, 'seen_by_robot': True}
    assert led['u']['where'] == {'off_pallet_a': True, 'outside_work_area': True, 'seen_by_robot': False}
    assert led['u']['left_pallet'] == 'before_mission' and led['u']['left_pallet_at_s'] == 2.4


def test_a_case_with_no_transport_clock_says_its_moment_is_not_recorded():
    start = {'u': (OUTSIDE, Q0)}
    led = TruthRecorder(lambda: start, CELL, SLOTS).summary(MissionReport('unresolved'), set())['case_ledger']
    assert led['u']['left_pallet_at_s'] is None
