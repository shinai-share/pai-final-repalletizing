"""What the robot's own cameras saw, kept with the run, and its estimate drawn over them (owner 2026-10-31: show how
the camera picture is recognised by the robot)."""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from shinai_pi.perception.geometry import CameraFrame, look_at_view
from shinai_pi import sensor_log as sl

W, H = 64, 48


def frame(rgb=True):
    view = look_at_view((0.0, 0.0, 3.0), (0.0, 0.0, 0.0), up=(0, 1, 0))
    depth = np.full((H, W), 3.0)
    inst = np.zeros((H, W), int)
    inst[10:30, 20:40] = 7
    extra = {'rgb': np.full((H, W, 3), 90, np.uint8)} if rgb else {}
    return CameraFrame(depth=depth, instance=inst, instance_map={7: 'case_03', 1: 'floor'}, view=view, fx=40.0, fy=40.0,
                       width=W, height=H, stamp=12.3456, name='overhead_a', extra=extra)


def test_a_world_point_projects_back_onto_the_pixel_it_came_from():
    f = frame()
    pts, _, _ = f.unproject()
    v, u = 17, 41
    p = pts[v * W + u]
    uu, vv, front = sl.project(f, p[None])
    assert front[0] and abs(uu[0] - u) < 1e-6 and abs(vv[0] - v) < 1e-6


def test_a_capture_is_kept_as_picture_segmentation_depth_and_camera(tmp_path):
    log = sl.SensorLog(tmp_path)
    name = log.write(frame())
    meta = json.loads((tmp_path / f'{name}.json').read_text(encoding='utf-8'))
    assert meta['camera'] == 'overhead_a' and abs(meta['t'] - 12.346) < 1e-9
    assert meta['objects'] == {'7': 'case_03', '1': 'floor'}
    assert np.allclose(np.asarray(meta['view']), frame().view)
    seg = np.asarray(Image.open(tmp_path / f'{name}_seg.png'))
    assert (seg[15, 25] != seg[5, 5]).any()                             # the case stands out from the rest
    depth = np.asarray(Image.open(tmp_path / f'{name}_depth.png'))
    assert depth[0, 0] == 3000                                          # millimetres
    assert (tmp_path / f'{name}.jpg').exists()


def test_a_capture_without_a_picture_keeps_the_rest(tmp_path):
    name = sl.SensorLog(tmp_path).write(frame(rgb=False))
    assert not (tmp_path / f'{name}.jpg').exists()
    assert (tmp_path / f'{name}_seg.png').exists() and (tmp_path / f'{name}.json').exists()


def test_the_estimate_is_drawn_where_the_case_is_seen():
    f = frame()
    box = {'id': 'case_03', 'poly': [[-0.2, -0.2], [0.2, -0.2], [0.2, 0.2], [-0.2, 0.2]], 'top': 0.0, 'state': 'risky'}
    img = sl.overlay(f.extra['rgb'], f, [box], chosen='case_03')
    u, v, _ = sl.project(f, np.array([[0.2, 0.0, 0.0]]))
    assert (np.asarray(img)[int(round(v[0])), int(round(u[0]))] != 90).any()
    assert (np.asarray(img)[2, 2] == 90).all()                           # far from the box: untouched


def test_the_screen_legend_shows_the_colours_the_overlay_draws():
    css = (Path(__file__).resolve().parents[1] / 'ui' / 'style.css').read_text(encoding='utf-8')
    for state, (r, g, b) in sl.STATE_COLOURS.items():
        assert f'.k.eye-{state} {{ border-color: rgb({r}, {g}, {b}); }}' in css
