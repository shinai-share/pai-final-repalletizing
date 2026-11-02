"""scripts/console_view.py: the monitoring screen of the video, read from the run's own physics-step trace."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
cv = pytest.importorskip('console_view')


def sig(n=600, dt=0.05):
    t = np.arange(n) * dt
    s = {'t': t, 'q': np.zeros((n, 6)), 'v': np.zeros((n, 6)), 'z_load': np.zeros(n), 'wrench': np.zeros((n, 6)),
         'fork_raw': np.zeros((n, 4)), 'fork_state': np.tile([-0.08, -0.08, 0.0, 0.0], (n, 1)), 'gripper': np.zeros(n)}
    return s


def test_a_resting_machine_reads_stopped_open_and_tines_home():
    st = cv.states(sig(), 10)
    assert st['X 軸'][0] == '停止' and st['吸着'][0] == '解放' and st['爪'][0] == '退避'


def test_moving_holding_and_pressing_are_read_from_the_signals():
    s = sig()
    s['v'][10, 0] = 0.4
    s['gripper'][10] = 2
    s['fork_state'][10] = [0.24, 0.24, 0.24, 0.24]
    s['fork_raw'][10] = [50.0, 51.0, 0.0, 0.0]
    st = cv.states(s, 10)
    assert st['X 軸'][0] == '移動中' and st['吸着'][0] == '保持' and st['爪'][0] == '押さえ'
    s['fork_raw'][10] = [0.5, 0.4, 0.0, 0.0]
    assert cv.states(s, 10)['爪'][0] == '差し込み'


def test_the_graph_window_ends_at_now_and_never_looks_ahead():
    s = sig()
    tt, _ = cv.window(s, s['z_load'], t=12.0, span=10.0)
    assert tt.max() <= 12.0 + 1e-9 and tt.min() >= 2.0 - 1e-9


def test_the_robots_eye_shows_the_latest_view_taken_before_now(tmp_path):
    (tmp_path / 'perception').mkdir()
    (tmp_path / 'perception' / 'index.json').write_text(json.dumps({'views': [
        {'t': 4.0, 'picture': 'a.jpg'}, {'t': 9.0, 'picture': 'b.jpg'}]}), encoding='utf-8')
    views = cv.eye_views(tmp_path)
    assert cv.eye_at(views, 8.0).name == 'a.jpg' and cv.eye_at(views, 9.5).name == 'b.jpg'
    assert cv.eye_at(views, 1.0) is None


def test_the_screen_is_full_hd(tmp_path):
    from PIL import Image
    main = Image.new('RGB', (1920, 1080), (90, 90, 90))
    img = cv.screen(main, None, None, sig(), 10, t=0.5, t0=0.0, step='吸着でつかむ', placed=0, total=6, speed=2)
    assert img.shape == (1080, 1920, 3)


def test_only_the_cameras_that_look_at_the_load_are_shown(tmp_path):
    (tmp_path / 'perception').mkdir()
    (tmp_path / 'perception' / 'index.json').write_text(json.dumps({'views': [
        {'t': 4.0, 'picture': 'a.jpg', 'camera': 'overhead_a'}, {'t': 5.0, 'picture': 'd.jpg', 'camera': 'down'}]}),
        encoding='utf-8')
    views = cv.eye_views(tmp_path)
    assert cv.eye_at(views, 6.0).name == 'a.jpg' and cv.eye_label(views, 6.0) == '真上から A'


def test_a_small_negative_force_reads_zero():
    assert cv.newtons(-0.4) == '0 N' and cv.newtons(-40.2) == '-40 N'
