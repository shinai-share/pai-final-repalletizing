"""What the robot saw and what it made of it, for people: each capture of its own cameras (run_mission.py --record keeps
them in sensors/), drawn with the estimate made from it (belief.jsonl) and the case it chose next (PICK_PLANNED).

    python scripts/perception_views.py artifacts/stages/S3/shinai_seed3001
Writes <run>/perception/<capture>_view.jpg (picture with the estimate drawn on it, beside the segmentation) and
<run>/perception/index.json."""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shinai_pi.perception.geometry import CameraFrame  # noqa: E402
from shinai_pi.sensor_log import overlay  # noqa: E402

STATES = ('safe', 'risky', 'blocked')
BELIEF_AFTER_S = 2.0               # an estimate this soon after a capture is the one made from it


def _jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def _frame(meta):
    view = np.asarray(meta['view'], float)
    return CameraFrame(depth=np.zeros((1, 1)), instance=np.zeros((1, 1), int), instance_map={}, view=view, fx=meta['fx'],
                       fy=meta['fy'], width=meta['width'], height=meta['height'], stamp=meta['t'], name=meta['camera'])


def build(run):
    run = Path(run)
    sensors = run / 'sensors'
    metas = sorted(sensors.glob('*.json')) if sensors.exists() else []
    if not metas:
        return []
    beliefs = _jsonl(run / 'belief.jsonl')
    picks = [e for e in _jsonl(run / 'events.jsonl') if e['type'] == 'PICK_PLANNED']
    out_dir = run / 'perception'
    out_dir.mkdir(exist_ok=True)
    views = []
    for path in metas:
        meta = json.loads(path.read_text(encoding='utf-8'))
        picture = path.with_suffix('.jpg')
        if not picture.exists():
            continue
        t = meta['t']
        after = [b for b in beliefs if t <= b['t'] <= t + BELIEF_AFTER_S]
        before = [b for b in beliefs if b['t'] <= t]
        belief = after[0] if after else (before[-1] if before else {'t': None, 'boxes': []})
        chosen = next((e['data']['case'] for e in picks if e['t'] >= t), None)
        drawn = overlay(np.asarray(Image.open(picture).convert('RGB')), _frame(meta), belief['boxes'], chosen=chosen)
        seg = Image.open(path.with_name(path.stem + '_seg.png')).convert('RGB')
        both = Image.new('RGB', (drawn.width + seg.width, max(drawn.height, seg.height)))
        both.paste(drawn, (0, 0))
        both.paste(seg, (drawn.width, 0))
        name = f'{path.stem}_view.jpg'
        both.save(out_dir / name, quality=88)
        views.append({'t': t, 'camera': meta['camera'], 'picture': name, 'belief_t': belief['t'], 'chosen': chosen,
                      'counts': {s: sum(1 for b in belief['boxes'] if b.get('state') == s) for s in STATES}})
    (out_dir / 'index.json').write_text(json.dumps({'views': views}, ensure_ascii=False, indent=1), encoding='utf-8')
    return views


if __name__ == '__main__':
    got = build(sys.argv[1])
    print(f'{len(got)} views')
