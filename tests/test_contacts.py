"""Contacts read from the simulator for scoring only (never shown to the robot): each touch between two
bodies becomes one record with when it began and ended, its peak force and its impulse."""
import numpy as np

from shinai_pi.evaluation.contacts import ContactLog

DT = 1 / 120


def _feed(log, forces_over_time, rows=('case_00', 'case_01'), cols=('case_00', 'case_01', 'plate')):
    for n, f in enumerate(forces_over_time):
        m = np.zeros((len(rows), len(cols), 3))
        for (i, j), v in f.items():
            m[rows.index(i), cols.index(j)] = (0, 0, v)
        log.step(n * DT, DT, rows, cols, m, poses={r: np.array([n * 0.001, 0, 0]) for r in rows})


def test_a_touch_becomes_one_record_with_its_peak_impulse_and_both_bodies():
    log = ContactLog(on_N=1.0)
    _feed(log, [{}, {('case_00', 'plate'): 40.0}, {('case_00', 'plate'): 80.0}, {('case_00', 'plate'): 20.0}, {}])
    [c] = log.records
    assert (c['a'], c['b']) == ('case_00', 'plate')
    assert c['peak_N'] == 80.0 and abs(c['impulse_Ns'] - 140.0 * DT) < 1e-9
    assert c['start_s'] == round(1 * DT, 4) and c['end_s'] == round(4 * DT, 4)
    assert c['moved_mm'] == {'case_00': 3.0}                        # how far the case moved while touched


def test_two_cases_pressing_each_other_are_one_contact_not_two():
    log = ContactLog(on_N=1.0)
    _feed(log, [{('case_00', 'case_01'): 30.0, ('case_01', 'case_00'): 30.0}, {}])
    assert [(c['a'], c['b']) for c in log.records] == [('case_00', 'case_01')]


def test_a_contact_still_on_at_the_end_is_closed_and_marked_so():
    log = ContactLog(on_N=1.0)
    _feed(log, [{('case_00', 'plate'): 5.0}, {('case_00', 'plate'): 5.0}])
    log.close(2 * DT)
    [c] = log.records
    assert c['open_at_end'] is True


def test_noise_below_the_threshold_is_not_a_contact():
    log = ContactLog(on_N=1.0)
    _feed(log, [{('case_00', 'plate'): 0.5}, {}])
    assert log.records == []


def test_a_touch_that_flickers_off_for_a_moment_stays_one_contact():
    log = ContactLog(on_N=1.0, gap_s=0.1)
    on = {('case_00', 'plate'): 30.0}
    _feed(log, [on, on, {}, {}, on, on] + [{}] * 20)                 # off for 2 steps (17 ms), then on again
    [c] = log.records
    assert c['start_s'] == 0.0 and c['peak_N'] == 30.0 and abs(c['impulse_Ns'] - 120.0 * DT) < 1e-9


def test_touches_further_apart_than_the_gap_are_separate():
    log = ContactLog(on_N=1.0, gap_s=0.05)
    on = {('case_00', 'plate'): 30.0}
    _feed(log, [on] + [{}] * 10 + [on] + [{}] * 10)
    assert len(log.records) == 2


def test_a_merged_touch_keeps_how_often_and_how_long_it_was_really_on_and_the_raw_touches_are_kept(tmp_path):
    raw = tmp_path / 'raw.jsonl'
    log = ContactLog(on_N=1.0, gap_s=0.1, raw_path=raw)
    on, hit = {('case_00', 'plate'): 30.0}, {('case_00', 'plate'): 250.0}
    _feed(log, [on, on, {}, {}, hit, {}, on] + [{}] * 20)
    log.close(30 * DT)
    [c] = log.records
    assert c['touches'] == 3 and c['peak_N'] == 250.0                # the short hit is not averaged away
    assert c['contact_s'] == round(4 * DT, 4) and c['gaps_s'] == round(3 * DT, 4)
    assert c['longest_gap_s'] == round(2 * DT, 4)
    assert [r['peak_N'] for r in map(__import__('json').loads, raw.read_text(encoding='utf-8').splitlines())] == [30.0, 250.0, 30.0]
