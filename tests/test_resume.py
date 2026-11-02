"""A run stopped part-way is resumed by running the same command again: what has a result is kept, the rest runs."""
from shinai_pi.evaluation.resume import gate_rows, pending


def _done(d, status='verified'):
    d.mkdir(parents=True)
    (d / 'result.json').write_text('{"status": "%s"}' % status, encoding='utf-8')


def test_only_the_runs_without_a_result_are_run_again(tmp_path):
    _done(tmp_path / 'trial_00')
    (tmp_path / 'trial_01').mkdir()                         # cut off: frames and events, no result
    assert pending([0, 1, 2], lambda n: tmp_path / f'trial_{n:02d}') == [1, 2]


def test_nothing_is_skipped_when_nothing_has_run(tmp_path):
    assert pending([101, 102], lambda s: tmp_path / f'shinai_seed{s:03d}') == [101, 102]


def test_the_gate_counts_every_finished_trial_in_the_folder_not_only_those_of_the_last_process(tmp_path):
    for n in range(4):
        _done(tmp_path / f'trial_{n:02d}')
    (tmp_path / 'trial_04').mkdir()
    _done(tmp_path / 'trial_02_timeout_incomplete')          # set aside by hand: not a trial of this gate
    assert len(gate_rows(tmp_path)) == 4
