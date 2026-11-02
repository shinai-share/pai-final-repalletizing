"""scripts/compose_video.py says what a cut leaves out and refuses to cut over a stop, a failure or a hand-over
(review 2026-10-30: the video may only leave out work that went as shown; what the cut hid is told on its card)."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
cv = pytest.importorskip('compose_video')


def ev(t, type_, **data):
    return {'t': t, 'type': type_, 'data': data}


EVENTS = [ev(1.0, 'MISSION_ACCEPTED', slots=24), ev(40.0, 'PLACED'), ev(90.0, 'PLACED'), ev(140.0, 'PLACED'),
          ev(150.0, 'FORK_RETRIED'), ev(190.0, 'PLACED'), ev(240.0, 'PLACED')]


def test_a_cut_tells_the_clock_it_skips_and_the_cases_placed_within_it():
    s = cv.skip_summary(EVENTS, 94.0, 200.0, t0=1.0)
    assert s['placed'] == 2 and s['from'] == '1:33' and s['to'] == '3:19'
    assert s['retries'] == 1


def test_the_cut_card_says_it_in_words():
    c = cv.skip_card(EVENTS, 94.0, 200.0, t0=1.0)
    assert c['title'] == '中略（2 箱）'
    assert any('1:33' in l and '3:19' in l for l in c['lines'])
    assert any('やり直し' in l and '1 回' in l for l in c['lines'])


def test_a_cut_without_retries_does_not_mention_them():
    c = cv.skip_card(EVENTS, 0.0, 100.0, t0=1.0)
    assert not any('やり直し' in l for l in c['lines'])


@pytest.mark.parametrize('kind', ['SAFETY_STOP', 'GRIP_FAILED', 'VACUUM_LOST', 'DROP_DETECTED', 'HANDOVER',
                                  'DAMAGE_RISK', 'MISSION_INCOMPLETE'])
def test_a_cut_over_a_stop_a_failure_or_a_hand_over_is_refused(kind):
    with pytest.raises(ValueError, match=kind):
        cv.skip_summary(EVENTS + [ev(120.0, kind)], 94.0, 200.0, t0=1.0)


def test_a_stop_just_outside_the_cut_is_not_hidden_by_it():
    assert cv.skip_summary(EVENTS + [ev(94.0, 'SAFETY_STOP')], 94.0, 200.0, t0=1.0)['placed'] == 2


def test_the_cut_must_go_forwards():
    with pytest.raises(ValueError):
        cv.skip_summary(EVENTS, 200.0, 94.0, t0=1.0)


RESULT = {'placed_on_b': 22, 'left_on_a': 1, 'handed_over': 1, 'safety_stops': 2, 'grip_failures': 1,
          'secondary_collapses': 0, 'drops_in_transit': 0, 'placement_xy_err_max_m': 0.0037, 'sim_seconds': 1098.0}


def test_the_result_card_shows_what_was_left_and_what_stopped():
    text = '\n'.join(cv.result_lines(RESULT))
    assert '22 / 24' in text and '未処理 2 個' in text
    assert '停止 2 回' in text and 'つかみ損ね 1 回' in text and '人への引き継ぎ 1 個' in text
    assert '18.3 分' in text and '省いた区間を含む' in text


STEP_EVENTS = [{'t': 1.0, 'type': 'MISSION_ACCEPTED', 'data': {}}, {'t': 2.0, 'type': 'GRIP_OK', 'data': {}},
          {'t': 5.0, 'type': 'FORK_IN', 'data': {}}, {'t': 9.0, 'type': 'MISSION_INCOMPLETE', 'data': {}}]


def test_tines_going_in_before_the_press_are_said_from_the_trace():
    assert cv.step_at(STEP_EVENTS, 3.0, tines_out=False) == cv.STEPS['GRIP_OK']
    assert cv.step_at(STEP_EVENTS, 3.0, tines_out=True) == cv.TINES_IN
    assert cv.step_at(STEP_EVENTS, 6.0, tines_out=True) == cv.STEPS['FORK_IN']


def test_an_ending_that_asks_a_person_is_said():
    assert cv.step_at(STEP_EVENTS, 9.5) == cv.STEPS['MISSION_INCOMPLETE']


def test_table_lines_come_from_the_records(tmp_path):
    rows = [{'seed': 3001, 'policy': 'shinai_learned', 'split': 'unseen', 'placed_ok': 30, 'placed_ng': 0,
             'fell_before_start': 6, 'fell_before_start_outside_cell': 6, 'left_in_cell': 0, 'robot_drops': 0,
             'robot_knocked_off': 0, 'secondary_collapses': 0, 'disturbed_cases': 0, 's3_passed': False}]
    p = tmp_path / 'table.json'
    p.write_text(json.dumps({'rows': rows, 'comparison': {'verdict_ja': 'x'}}), encoding='utf-8')
    lines = cv.table_lines(p, 'shinai_learned')
    assert len(lines) == 1 and '3001' in lines[0] and '30' in lines[0] and '6' in lines[0]


def test_a_still_is_fitted_into_the_frame(tmp_path):
    from PIL import Image
    p = tmp_path / 'a.jpg'
    Image.new('RGB', (1280, 480), (200, 0, 0)).save(p)
    img = cv.still((1920, 1080), {'image': str(p), 'title': 't', 'lines': ['l']})
    assert img.shape == (1080, 1920, 3)


def test_a_card_line_too_long_for_the_frame_is_refused(tmp_path):
    from PIL import Image
    pic = tmp_path / 'a.jpg'
    Image.new('RGB', (64, 32)).save(pic)
    with pytest.raises(ValueError):
        cv.card((1920, 1080), {'title': 't', 'lines': ['長い' * 40]})
    with pytest.raises(ValueError):
        cv.still((1920, 1080), {'image': str(pic), 'title': 't', 'lines': ['長い' * 40]})


def test_table_lines_fit_a_small_card(tmp_path):
    rows = [{'seed': 3001, 'policy': 'shinai_learned', 'split': 'unseen', 'placed_ok': 30, 'placed_ng': 0,
             'fell_before_start': 6, 'fell_before_start_outside_cell': 6, 'left_in_cell': 0, 'robot_drops': 0,
             'robot_knocked_off': 0, 'secondary_collapses': 0, 'disturbed_cases': 0, 's3_passed': False}]
    p = tmp_path / 'table.json'
    p.write_text(json.dumps({'rows': rows, 'comparison': {'verdict_ja': 'x'}}), encoding='utf-8')
    cv.card((3840, 2160), {'title': 't', 'small': True, 'lines': cv.table_lines(p, 'shinai_learned')})


def test_a_part_can_end_before_the_drawn_frames_do():
    names = ['00004000.jpg', '00100000.jpg', '00241300.jpg', '00241400.jpg']
    assert cv.frames_until(names, 241.3) == names[:3]
    assert cv.frames_until(names, None) == names
