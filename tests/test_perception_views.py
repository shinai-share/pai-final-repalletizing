"""Each capture of the robot's cameras paired with the estimate it led to and the case chosen next, drawn for people."""
import importlib.util
import json
from pathlib import Path

import numpy as np

from shinai_pi.perception.geometry import CameraFrame, look_at_view
from shinai_pi.sensor_log import SensorLog

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('perception_views', ROOT / 'scripts' / 'perception_views.py')
pv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pv)


def _run(tmp_path):
    run = tmp_path / 'run'
    log = SensorLog(run / 'sensors')
    view = look_at_view((0.0, 0.0, 3.0), (0.0, 0.0, 0.0), up=(0, 1, 0))
    for t in (5.0, 40.0):
        log.write(CameraFrame(depth=np.full((48, 64), 3.0), instance=np.zeros((48, 64), int), instance_map={}, view=view,
                              fx=40.0, fy=40.0, width=64, height=48, stamp=t, name='overhead_a',
                              extra={'rgb': np.full((48, 64, 3), 90, np.uint8)}))
    box = lambda k, s: {'id': k, 'poly': [[-0.2, -0.2], [0.2, -0.2], [0.2, 0.2], [-0.2, 0.2]], 'top': 0.0, 'state': s}  # noqa: E731
    rows = [{'t': 4.0, 'boxes': [box('case_01', 'blocked')]}, {'t': 5.2, 'boxes': [box('case_01', 'safe')]},
            {'t': 41.0, 'boxes': [box('case_02', 'risky')]}]
    (run / 'belief.jsonl').write_text('\n'.join(json.dumps(r) for r in rows), encoding='utf-8')
    ev = [{'t': 6.0, 'type': 'PICK_PLANNED', 'data': {'case': 'case_01'}},
          {'t': 45.0, 'type': 'PICK_PLANNED', 'data': {'case': 'case_02'}}]
    (run / 'events.jsonl').write_text('\n'.join(json.dumps(e) for e in ev), encoding='utf-8')
    return run


def test_each_capture_is_paired_with_the_estimate_made_from_it_and_the_next_choice(tmp_path):
    run = _run(tmp_path)
    views = pv.build(run)
    assert [v['t'] for v in views] == [5.0, 40.0]
    first = views[0]
    assert first['belief_t'] == 5.2 and first['chosen'] == 'case_01'
    assert first['counts'] == {'safe': 1, 'risky': 0, 'blocked': 0}
    assert views[1]['chosen'] == 'case_02'
    for v in views:
        assert (run / 'perception' / v['picture']).exists()
    index = json.loads((run / 'perception' / 'index.json').read_text(encoding='utf-8'))
    assert index['views'] == views


def test_a_run_without_camera_records_gives_no_views(tmp_path):
    run = tmp_path / 'empty'
    run.mkdir()
    assert pv.build(run) == []
