import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location('mission_table', Path(__file__).resolve().parents[1] / 'scripts' / 'mission_table.py')
mt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mt)


def ledger(**states):
    """states: name -> (state, quality, left_pallet)"""
    return {k: {'state': s, 'quality': q, 'left_pallet': lp} for k, (s, q, lp) in states.items()}


def result(policy='shinai', seed=1, status='verified', led=None, **kw):
    r = {'seed': seed, 'policy': policy, 'status': status, 'drops_in_transit': 0, 'failed_checks': [],
         'secondary_collapses': 0, 'disturbed_cases': 0, 'learned_decisions': 0, 'learned_changed': 0,
         'sim_seconds': 100.0, 'wall_seconds': 400.0, 'case_ledger': led or {}}
    r.update(kw)
    return r


def test_cases_fallen_before_the_robot_started_are_kept_apart_from_its_own_drops():
    led = ledger(a=('placed_on_b', 'ok', None), b=('outside_cell', None, 'before_mission'),
                 c=('placed_on_b', 'ng', 'before_mission'), d=('outside_cell', None, 'during_pick:a'))
    row = mt.row(result(status='unresolved', led=led, drops_in_transit=1), 'unseen')
    assert row['fell_before_start'] == 2
    assert row['fell_before_start_outside_cell'] == 1
    assert row['robot_drops'] == 1 and row['robot_knocked_off'] == 1
    assert (row['placed_ok'], row['placed_ng']) == (1, 1)
    assert row['s3_passed'] is False and row['s3_reasons']


def test_a_clean_run_passes_s3_and_times_keep_their_meaning():
    row = mt.row(result(led=ledger(a=('placed_on_b', 'ok', None))), 'seen')
    assert row['s3_passed'] is True and row['s3_reasons'] == []
    assert row['sim_seconds_whole_task'] == 100.0 and row['wall_seconds_compute'] == 400.0


def pair(seed, learned_sum, rule_sum, changed=1, learned_status='verified', rule_status='verified'):
    learned = mt.row(result('shinai_learned', seed, learned_status, disturbed_cases=learned_sum, learned_decisions=5,
                            learned_changed=changed), 'seen')
    rule = mt.row(result('shinai', seed, rule_status, disturbed_cases=rule_sum), 'seen')
    return [learned, rule]


def test_better_than_the_rule_only_when_fewer_in_one_pair_and_more_in_none():
    assert mt.compare(pair(1, 0, 1) + pair(2, 0, 0), seeds=[1, 2])['verdict'] == 'better'
    assert mt.compare(pair(1, 0, 1) + pair(2, 1, 0), seeds=[1, 2])['verdict'] == 'not_shown'
    assert mt.compare(pair(1, 0, 0), seeds=[1])['verdict'] == 'not_shown'


def test_losing_s3_only_on_the_learned_side_blocks_better():
    rows = pair(1, 0, 1, learned_status='unresolved', rule_status='verified')
    assert mt.compare(rows, seeds=[1])['verdict'] == 'not_shown'


def test_no_changed_choice_anywhere_is_said_so_and_missing_runs_are_not_run():
    out = mt.compare(pair(1, 0, 0, changed=0), seeds=[1, 2])
    assert out['verdict'] == 'model_never_changed_a_choice'
    assert out['not_run'] == [{'seed': 2, 'policy': 'shinai_learned'}, {'seed': 2, 'policy': 'shinai'}]
