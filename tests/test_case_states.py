"""Runs recorded before the harness kept a ledger are accounted for from their own record: every case of
the load in exactly one state, and what the record cannot tell (when a case left pallet A) said so."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import case_states as cs  # noqa: E402
from shinai_pi.config import load_cell  # noqa: E402

CELL = load_cell()


def _ev(kind, **data):
    return {'type': kind, 'data': data}


RESULT = {
    'final_positions': {'p': [0.95, 0.0, 0.25], 'q': [0.95, 0.3, 0.25], 'h': [0.95, 0.88, 0.11], 'g': [0.5, 0.0, 1.0],
                        'o': [-0.6, 2.2, 0.11], 'r': [-1.6, 0.0, 0.11], 'n': [-1.6, 1.4, 0.11]},
    'placement_errors': {'p': {'slot': 0, 'xy_m': 0.002, 'z_m': 0.0, 'tilt_deg': 0.1, 'yaw_deg': 0.3},
                         'q': {'slot': 1, 'xy_m': 0.4, 'z_m': 0.0, 'tilt_deg': 90.0, 'yaw_deg': 30.0}},
    'placed_unscored': [], 'handed_over': 1, 'remaining': {'r': '腕が届かない場所にある（人が取り出す）'},
    'cases_never_detected': ['n'],
}
EVENTS = [_ev('HANDOVER', case='h', reason='姿勢を直す台が空いていない', area='handover'),
          _ev('HANDOVER', case='g', reason='置き場がなく元の位置も使えないため、持ったまま待機する（人が受け取る）'),
          _ev('MISSION_INCOMPLETE', status='interrupted')]


def test_every_case_of_a_recorded_run_lands_in_one_state():
    led = cs.ledger_from_record(RESULT, EVENTS, CELL)
    assert {k: v['state'] for k, v in led.items()} == {'p': 'placed_on_b', 'q': 'placed_on_b', 'h': 'handed_over',
                                                       'g': 'held_by_robot', 'o': 'outside_cell', 'r': 'left_in_cell',
                                                       'n': 'left_in_cell'}
    assert led['q']['quality'] == 'ng' and led['p']['quality'] == 'ok'
    assert led['n']['detail'] == '未検出' and led['r']['detail'].startswith('腕が届かない')
    assert all(v['left_pallet'] == 'not_recorded' for v in led.values() if v['state'] in ('outside_cell', 'left_in_cell'))


def test_a_handover_that_kept_the_case_on_the_cups_is_held_not_handed_over():
    led = cs.ledger_from_record(RESULT, EVENTS, CELL)
    assert led['g']['state'] == 'held_by_robot'


def test_the_count_of_handovers_must_match_the_record_or_the_run_is_flagged():
    bad = dict(RESULT, handed_over=2)
    led = cs.ledger_from_record(bad, EVENTS, CELL)
    assert cs.check(led, bad) == ['人に回した数が記録（2）と一致しない（1）']
    assert cs.check(cs.ledger_from_record(RESULT, EVENTS, CELL), RESULT) == []
