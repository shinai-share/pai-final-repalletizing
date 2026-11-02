"""The S3 gate counts what happened to every case, not only how the run ended (S3 review 2026-10-18:
102 ended 'moved_quality_unverified' with 19 of its 21 placed cases pushed out of tolerance afterwards)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import stage_gate as sg  # noqa: E402


def _run(**ledger_states):
    led = {}
    for i, (state, quality, left) in enumerate(ledger_states.values()):
        led[f'case_{i:02d}'] = {'state': state, 'quality': quality, 'left_pallet': left}
    return {'status': 'verified', 'drops_in_transit': 0, 'secondary_collapses': 0, 'case_ledger': led}


def test_a_clean_run_passes():
    assert sg.judge('S3', _run(a=('placed_on_b', 'ok', None), b=('placed_on_b', 'ok', None))) == []


def test_a_placed_case_out_of_tolerance_at_the_end_fails_the_run():
    why = sg.judge('S3', _run(a=('placed_on_b', 'ok', None), b=('placed_on_b', 'ng', None)))
    assert why == ['置いた後に基準の外にある 1 個']


def test_a_case_the_robot_pushed_out_of_the_cell_fails_but_one_already_off_before_it_started_does_not():
    pushed = sg.judge('S3', _run(a=('outside_cell', None, 'during_pick:case_03')))
    before = sg.judge('S3', _run(a=('outside_cell', None, 'before_mission')))
    assert pushed == ['ロボットの作業中に作業場の外へ出た 1 個'] and before == []


def test_where_a_case_went_unrecorded_it_cannot_pass():
    assert sg.judge('S3', _run(a=('outside_cell', None, 'not_recorded'))) == ['作業場の外へ出た時期が分からない 1 個']
