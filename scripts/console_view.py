"""The monitoring screen of the submission video: the drawn-again run beside instruments read from the same run's
physics-step trace (run_mission.py --trace). Every value on it is a recorded value at the frame's time; nothing is
smoothed into a shape it did not have, and the graphs never show what comes after the frame.

Used by scripts/compose_video.py when a plan says "layout": "console"."""
from pathlib import Path
import json
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT = 'C:/Windows/Fonts/NotoSansJP-VF.ttf'
W, H = 1920, 1080
BG, PANEL, LINE = (12, 14, 17), (22, 25, 30), (52, 57, 64)
INK, DIM, ACCENT = (236, 238, 240), (150, 157, 165), (232, 194, 26)
TONES = {'run': (46, 160, 220), 'ok': (70, 190, 120), 'hold': (232, 194, 26), 'idle': (110, 116, 124)}
MOVING = (0.01, 0.01, 0.01, 0.02)        # m/s for X, Y, Z and rad/s for the turn: slower reads as stopped
PRESSING_N = 20.0                        # tine force (left + right) above which the tines carry the case
KEYS = ('gantry_q', 'gantry_v', 'z_load', 'head_wrench', 'fork_raw', 'fork_state', 'gripper')
STEP_EVERY = 4                            # trace rows kept: 120 Hz physics read at 30 Hz


def font(px, weight='Regular'):
    f = ImageFont.truetype(FONT, int(px))
    try:
        f.set_variation_by_name(weight)
    except (OSError, ValueError):
        pass
    return f


def load_signals(run):
    """The tool's signals from a run's trace folder, every STEP_EVERY-th physics step."""
    folder = Path(run) / 'trace'
    idx = json.loads((folder / 'trace_index.json').read_text(encoding='utf-8'))
    parts = {k: [] for k in ('t',) + KEYS}
    for k in range(idx['chunks']):
        with np.load(folder / f'trace_{k:05d}.npz') as z:
            parts['t'].append(z['t'])
            for key in KEYS:
                parts[key].append(z['tool_' + key])
    s = {k: np.concatenate(v)[::STEP_EVERY] for k, v in parts.items()}
    return {'t': s['t'], 'q': s['gantry_q'], 'v': s['gantry_v'], 'z_load': s['z_load'].reshape(-1),
            'wrench': s['head_wrench'], 'fork_raw': s['fork_raw'], 'fork_state': s['fork_state'],
            'gripper': s['gripper'].reshape(-1)}


def newtons(x):
    return f'{int(round(float(x)))} N'     # int() so that -0.4 reads 0, not -0


def index_at(sig, t):
    return int(min(max(np.searchsorted(sig['t'], t, side='right') - 1, 0), len(sig['t']) - 1))


def states(sig, i):
    """Each device's state at row i, as (state, tone, value)."""
    q, v = sig['q'][i], sig['v'][i]
    out = {}
    for k, name in enumerate(('X 軸', 'Y 軸', 'Z 軸')):
        moving = abs(v[k]) > MOVING[k]
        out[name] = ('移動中' if moving else '停止', 'run' if moving else 'idle', f'{q[k]:+.2f} m')
    turning = abs(v[3]) > MOVING[3]
    out['旋回'] = ('回転中' if turning else '停止', 'run' if turning else 'idle', f'{math.degrees(q[3]):+.0f} 度')
    g = int(sig['gripper'][i])
    out['吸着'] = ({2: '保持', 1: '吸着中'}.get(g, '解放'), {2: 'hold', 1: 'run'}.get(g, 'idle'),
                  newtons(sig['z_load'][i]))
    ext = float(min(sig['fork_state'][i][:2]))
    force = float(sig['fork_raw'][i][0] + sig['fork_raw'][i][1])
    if ext <= 0.0:
        fork = ('退避', 'idle')
    elif force > PRESSING_N:
        fork = ('押さえ', 'hold')
    else:
        fork = ('差し込み', 'run')
    out['爪'] = (*fork, newtons(force))
    return out


def window(sig, values, t, span=30.0):
    """The last `span` seconds up to t (never after it)."""
    m = (sig['t'] <= t) & (sig['t'] >= t - span)
    return sig['t'][m], np.asarray(values)[m]


EYE_CAMERAS = {'overhead_a': '真上から A', 'overhead_b': '真上から B', 'under': '下から（持った箱）'}


def eye_views(run):
    """The robot's own pictures with its estimates drawn on (scripts/perception_views.py), from the cameras that
    look at the load; the tool camera's close views are left out (they show the cups, not the pile)."""
    idx = Path(run) / 'perception' / 'index.json'
    if not idx.exists():
        return []
    views = json.loads(idx.read_text(encoding='utf-8'))['views']
    return sorted(((v['t'], Path(run) / 'perception' / v['picture'], v.get('camera')) for v in views
                   if v.get('camera', 'overhead_a') in EYE_CAMERAS), key=lambda x: x[0])


def eye_at(views, t):
    seen = [(p, c) for vt, p, c in views if vt <= t]
    return seen[-1][0] if seen else None


def eye_label(views, t):
    seen = [c for vt, p, c in views if vt <= t]
    return EYE_CAMERAS.get(seen[-1], '') if seen else ''


def _panel(d, box, title):
    d.rounded_rectangle(box, radius=8, fill=PANEL, outline=LINE)
    d.text((box[0] + 14, box[1] + 8), title, font=font(17, 'Medium'), fill=DIM)


def _graph(d, box, title, unit, tt, yy, t, span, lo, hi, color):
    _panel(d, box, title)
    x0, y0, x1, y1 = box[0] + 14, box[1] + 40, box[2] - 14, box[3] - 30
    d.line((x0, y1, x1, y1), fill=LINE)
    d.text((x0, y1 + 4), f'-{span:.0f} 秒', font=font(13), fill=DIM)
    d.text((x1, y1 + 4), '今', font=font(13), fill=DIM, anchor='ra')
    d.text((x1, box[1] + 10), newtons(yy[-1] if len(yy) else 0) if unit == 'N' else f'{(yy[-1] if len(yy) else 0):.0f} {unit}', font=font(18, 'Bold'), fill=INK, anchor='ra')
    if len(tt) > 1:
        top = max(hi, float(np.max(yy)) * 1.1)
        lo = min(lo, float(np.min(yy)))
        pts = [(x1 - (t - a) / span * (x1 - x0), y1 - (min(max(b, lo), top) - lo) / (top - lo) * (y1 - y0))
               for a, b in zip(tt, yy)]
        d.line(pts, fill=color, width=2)


def _picture(canvas, d, img, box, label):
    _panel(d, box, label)
    inner = (box[0] + 8, box[1] + 34, box[2] - 8, box[3] - 8)
    if img is None:
        d.text(((inner[0] + inner[2]) // 2, (inner[1] + inner[3]) // 2), 'まだ撮っていない', font=font(16), fill=DIM,
               anchor='mm')
        return
    w, h = inner[2] - inner[0], inner[3] - inner[1]
    pic = img.convert('RGB')
    s = min(w / pic.width, h / pic.height)
    pic = pic.resize((int(pic.width * s), int(pic.height * s)), Image.LANCZOS)
    canvas.paste(pic, (inner[0] + (w - pic.width) // 2, inner[1] + (h - pic.height) // 2))


def screen(main, inset, eye, sig, i, *, t, t0, step, placed, total, speed, span=30.0, eye_name=''):
    canvas = Image.new('RGB', (W, H), BG)
    d = ImageDraw.Draw(canvas)
    s = max(0, int(t - t0))
    d.text((16, 14), '監視画面 ・ NVIDIA Isaac Sim（記録した走行の再描画・設備の外観は表示用）', font=font(22, 'Bold'), fill=INK)
    d.text((W - 16, 14), f'{speed:g} 倍速   経過 {s // 60}:{s % 60:02d}   B に置いた箱 {placed} / {total}',
           font=font(22, 'Bold'), fill=INK, anchor='ra')
    canvas.paste(main.convert('RGB').resize((1280, 720), Image.LANCZOS), (16, 60))
    d.rectangle((16, 60, 1296, 780), outline=LINE)
    d.rounded_rectangle((28, 726, 28 + 14 + d.textlength(step, font=font(24, 'Bold')) + 14, 770), radius=6,
                        fill=(14, 16, 18))
    d.rectangle((28, 726, 32, 770), fill=ACCENT)
    d.text((42, 748), step, font=font(24, 'Bold'), fill=INK, anchor='lm')
    graphs = [('吸着の荷重', 'N', sig['z_load'], 0, 200, (46, 160, 220)),
              ('爪の力（左右の和）', 'N', sig['fork_raw'][:, 0] + sig['fork_raw'][:, 1], 0, 150, (232, 194, 26)),
              ('手首の上下の力', 'N', sig['wrench'][:, 2], 0, 200, (70, 190, 120))]
    gw = (1280 - 2 * 12) // 3
    for k, (title, unit, values, lo, hi, color) in enumerate(graphs):
        tt, yy = window(sig, values, t, span)
        x = 16 + k * (gw + 12)
        _graph(d, (x, 796, x + gw, 1064), title, unit, tt, yy, t, span, lo, hi, color)
    _picture(canvas, d, inset, (1312, 60, 1904, 400), '手元のカメラ（表示用）')
    _picture(canvas, d, eye, (1312, 412, 1904, 660), f'ロボットの目　{eye_name}（最新の撮影と見立て）')
    _panel(d, (1312, 672, 1904, 1064), '装置の状態')
    rows = list(states(sig, i).items()) + [('工程', (step, 'run', ''))]
    for k, (name, (state, tone, value)) in enumerate(rows):
        y = 712 + k * 42
        d.text((1328, y), name, font=font(19, 'Medium'), fill=DIM)
        if name == '工程':
            d.text((1420, y), state, font=font(17), fill=INK)
            continue
        pill = font(17, 'Bold')
        w = d.textlength(state, font=pill) + 24
        d.rounded_rectangle((1420, y - 2, 1420 + w, y + 28), radius=14, outline=TONES[tone], width=2)
        d.text((1420 + w / 2, y + 13), state, font=pill, fill=TONES[tone], anchor='mm')
        d.text((1888, y), value, font=font(19, 'Bold'), fill=INK, anchor='ra')
    return np.asarray(canvas)
