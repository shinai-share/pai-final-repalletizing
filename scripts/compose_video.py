"""Put the submission video together from drawn-again frames (scripts/rerender_trace.py) and text cards.

Every picture is a drawn frame of the recorded run, untouched; what is laid over it comes from the run's own records:
the playback speed and the run's clock from the frame's time, the robot's current step from the latest event in
events.jsonl, and the count of cases on pallet B from its PLACED events. Cards say what the video is and is not.

    python scripts/compose_video.py plans/submission_mech2_2x.json
A plan: {"run": DIR, "frames": DIR prefix, "fps": 30, "size": [3840, 2160], "out": "artifacts/video/NAME.mp4",
         "segments": [{"card": {...}, "seconds": 6}, {"frames": "s3_first", "speed": 2}, ...]}
Writes the video and, beside it, a 1920 x 1080 copy and NAME.json listing every part and where it came from."""
import json
import subprocess
import sys
from pathlib import Path

import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
FONT = 'C:/Windows/Fonts/NotoSansJP-VF.ttf'
INK, DIM, ACCENT, PANEL, BG = (245, 246, 247), (178, 184, 190), (232, 194, 26), (14, 16, 18, 190), (16, 18, 21)

# what the robot is doing, said from the event that starts it (held until the next one)
STEPS = {
    'MISSION_ACCEPTED': '指示を受けた：パレット A の荷を、パレット B へ',
    'OBSERVE_REQUESTED': '見えていない所を、見に行く',
    'PICK_PLANNED': '崩れを起こさない箱を選ぶ',
    'GRIP_OK': '吸着でつかむ',
    'LOAD_MEASURED': '持ち上げて、重さと重心を量る',
    'FORK_IN': '爪で下から押さえ、支えて運ぶ',
    'FORK_LANDED': 'B に降ろす',
    'FORK_RETRACTING': '爪を抜く',
    'RELEASED': '吸着を離す',
    'PLACEMENT_CHECKED': '置けたかを、カメラで確かめる',
    'PLACEMENT_SETTLED': '落ち着いたことを確かめる',
    'MISSION_COMPLETE': '積み替えを終えた',
    'MISSION_INCOMPLETE': '取れない箱を人に知らせて、作業を終える',
}
TINES_IN = '持ち上げてすぐ、爪を箱の下へ差し込む'   # tines out (rerender's tines.json) after the grip, before the press
VIEWS = {'overhead_a': '上から荷を見る', 'side_west': '横から、隠れた箱を見る', 'side_east': '横から、隠れた箱を見る',
         'under': '下から、つかんだ位置を測る'}


def font(px, weight='Regular'):
    f = ImageFont.truetype(FONT, int(px))
    try:
        f.set_variation_by_name(weight)
    except (OSError, ValueError):
        pass
    return f


def step_at(events, t, tines_out=False):
    text, last = '', None
    for e in events:
        if e['t'] > t + 1e-6:
            break
        if e['type'] in STEPS:
            text, last = STEPS[e['type']], e['type']
        elif e['type'] == 'OBSERVED' and e['data'].get('view') in VIEWS:
            text, last = VIEWS[e['data']['view']], e['type']
    return TINES_IN if tines_out and last == 'GRIP_OK' else text


def placed_at(events, t):
    return sum(1 for e in events if e['type'] == 'PLACED' and e['t'] <= t + 1e-6)


def clock(t, t0):
    s = max(0, int(t - t0))
    return f'{s // 60}:{s % 60:02d}'


def overlay(img, size, *, speed, t, t0, events, total, inset, tines_out=False):
    W, H = size
    k = W / 1920
    base = img.convert('RGBA').resize((W, H), Image.LANCZOS)
    layer = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    pad = int(36 * k)
    # top left: what this is
    f = font(26 * k, 'Medium')
    label = 'NVIDIA Isaac Sim ・ 記録した走行の再描画'
    w = d.textlength(label, font=f)
    d.rounded_rectangle((pad, pad, pad + w + 2 * 20 * k, pad + 56 * k), radius=int(10 * k), fill=PANEL)
    d.text((pad + 20 * k, pad + 28 * k), label, font=f, fill=INK, anchor='lm')
    # top right: speed and the run's clock
    f2 = font(30 * k, 'Bold')
    right = f'{speed:g} 倍速    経過 {clock(t, t0)}'
    w = d.textlength(right, font=f2)
    d.rounded_rectangle((W - pad - w - 40 * k, pad, W - pad, pad + 56 * k), radius=int(10 * k), fill=PANEL)
    d.text((W - pad - 20 * k, pad + 28 * k), right, font=f2, fill=INK, anchor='rm')
    # bottom left: the step and the count
    step = step_at(events, t, tines_out)
    n = placed_at(events, t)
    f3, f4 = font(44 * k, 'Bold'), font(30 * k, 'Medium')
    w = max(d.textlength(step, font=f3), d.textlength(f'B に置いた箱  {n} / {total}', font=f4))
    y0 = H - pad - 150 * k
    d.rounded_rectangle((pad, y0, pad + w + 56 * k, H - pad), radius=int(12 * k), fill=PANEL)
    d.rectangle((pad, y0, pad + 8 * k, H - pad), fill=ACCENT)
    d.text((pad + 34 * k, y0 + 48 * k), step, font=f3, fill=INK, anchor='lm')
    d.text((pad + 34 * k, y0 + 108 * k), f'B に置いた箱  {n} / {total}', font=f4, fill=DIM, anchor='lm')
    out = Image.alpha_composite(base, layer)
    if inset is not None:
        iw = int(W * 0.30)
        ih = int(iw * inset.height / inset.width)
        x, y = W - pad - iw, pad + int(56 * k) + int(24 * k)   # top right, under the clock: the work below stays clear
        b = int(4 * k)
        frame = Image.new('RGBA', (iw + 2 * b, ih + 2 * b), INK + (255,))
        out.alpha_composite(frame, (x - b, y - b))
        out.alpha_composite(inset.convert('RGBA').resize((iw, ih), Image.LANCZOS), (x, y))
        d2 = ImageDraw.Draw(out)
        f5 = font(24 * k, 'Medium')
        lw = d2.textlength('手元', font=f5)
        d2.rounded_rectangle((x + 14 * k, y + 14 * k, x + 14 * k + lw + 28 * k, y + 56 * k), radius=int(8 * k), fill=PANEL)
        d2.text((x + 28 * k, y + 35 * k), '手元', font=f5, fill=INK, anchor='lm')
    return np.asarray(out.convert('RGB'))


def fit(d, line, f, width):
    """A line that would run off the frame stops the video: it is shortened in the plan, not cut on screen."""
    if d.textlength(line, font=f) > width:
        raise ValueError(f'line wider than the frame: {line}')


def card(size, c):
    W, H = size
    k = W / 1920
    img = Image.new('RGB', (W, H), BG)
    d = ImageDraw.Draw(img)
    x = int(180 * k)
    y = int(300 * k)
    if c.get('kicker'):
        d.text((x, y), c['kicker'], font=font(30 * k, 'Medium'), fill=ACCENT)
        y += int(80 * k)
    d.text((x, y), c['title'], font=font(66 * k, 'Bold'), fill=INK)
    y += int(150 * k)
    f = font((30 if c.get('small') else 36) * k, 'Regular')
    for line in c.get('lines', []):
        fit(d, line, f, W - 2 * x)
        d.text((x, y), line, font=f, fill=INK)
        y += int((64 if c.get('small') else 76) * k)
    if c.get('note'):
        d.text((x, H - int(170 * k)), c['note'], font=font(28 * k, 'Regular'), fill=DIM)
    return np.asarray(img)


# a cut may only leave out work that went as shown: these must be seen, not skipped
NOT_CUT = ('SAFETY_STOP', 'GRIP_FAILED', 'VACUUM_LOST', 'DROP_DETECTED', 'HANDOVER', 'DAMAGE_RISK', 'MISSION_INCOMPLETE')
RETRIES = ('FORK_RETRIED', 'PLACEMENT_REPAIRED', 'MOVE_REJECTED', 'DECISION_DEFERRED', 'NEAR_MISS', 'FORK_SKIPPED')


def skip_summary(events, a, b, t0):
    """What the cut from run time a to b leaves out: refused when a stop, a failure or a hand-over lies within it."""
    if b <= a:
        raise ValueError(f'a cut from {a} to {b} does not go forwards')
    inside = [e for e in events if a < e['t'] <= b]
    hidden = sorted({e['type'] for e in inside if e['type'] in NOT_CUT})
    if hidden:
        raise ValueError(f'the cut from {a} to {b} would hide {", ".join(hidden)}')
    return {'from': clock(a, t0), 'to': clock(b, t0), 'placed': sum(e['type'] == 'PLACED' for e in inside),
            'retries': sum(e['type'] in RETRIES for e in inside)}


def skip_card(events, a, b, t0):
    s = skip_summary(events, a, b, t0)
    lines = [f"作業の時刻 {s['from']} から {s['to']} までを省いています（シミュレーションの中の時刻）。",
             f"この間に B へ置いた箱は {s['placed']} 個で、停止・失敗・人への引き継ぎはありません。"]
    if s['retries']:
        lines.append(f"この間に、ロボットが自分で判断したやり直しが {s['retries']} 回あります。")
    return {'title': f"中略（{s['placed']} 箱）", 'lines': lines}


SPLIT_SHORT = {'seen': '学習と同じ種類', 'unseen': '学習に使っていない種類'}


def table_lines(path, policy):
    """One line per run of `policy` from scripts/mission_table.py's table.json."""
    rows = json.loads(Path(path).read_text(encoding='utf-8'))['rows']
    out = []
    for x in rows:
        if x['policy'] != policy:
            continue
        n = x['placed_ok'] + x['placed_ng']
        out.append(f"{x['seed']}（{SPLIT_SHORT.get(x['split'], x['split'])}）  置いた {n} 箱、基準内 {x['placed_ok']}"
                   f" ／ 運搬中に落ちていた {x['fell_before_start']} 箱 ／ 落下 {x['robot_drops']}"
                   f"・二次崩れ {x['secondary_collapses']}")
    return out


def still(size, c):
    """A recorded picture (the robot's-eye view) fitted between a title and its lines."""
    W, H = size
    k = W / 1920
    img = Image.new('RGB', (W, H), BG)
    d = ImageDraw.Draw(img)
    x = int(180 * k)
    d.text((x, int(90 * k)), c['title'], font=font(52 * k, 'Bold'), fill=INK)
    lines = c.get('lines', [])
    top, bottom = int(200 * k), H - int((80 + 60 * len(lines)) * k)
    pic = Image.open(c['image']).convert('RGB')
    scale = min((W - 2 * x) / pic.width, (bottom - top) / pic.height)
    pic = pic.resize((int(pic.width * scale), int(pic.height * scale)), Image.LANCZOS)
    img.paste(pic, ((W - pic.width) // 2, top))
    y = bottom + int(30 * k)
    f = font(32 * k, 'Regular')
    for line in lines:
        fit(d, line, f, W - 2 * x)
        d.text((x, y), line, font=f, fill=INK)
        y += int(60 * k)
    return np.asarray(img)


def frames_until(names, until):
    """Frames up to the run time `until` (seconds; None keeps them all): a part may end where the plan says."""
    return [n for n in names if until is None or int(n[:-4]) / 1000 <= until + 1e-9]


def result_lines(r):
    total = r['placed_on_b'] + r['left_on_a'] + r['handed_over']
    return [f"パレット B に積んだ箱    {r['placed_on_b']} / {total} 個（未処理 {r['left_on_a'] + r['handed_over']} 個）",
            f"途中の停止 {r['safety_stops']} 回・つかみ損ね {r['grip_failures']} 回・人への引き継ぎ {r['handed_over']} 個",
            f"二次の荷崩れ {r['secondary_collapses']} 回・運んでいる間の落下 {r['drops_in_transit']} 回",
            f"置いた位置の誤差    最大 {r['placement_xy_err_max_m'] * 1000:.1f} mm",
            f"作業の時間    {r['sim_seconds'] / 60:.1f} 分（シミュレーションの中の時間。省いた区間を含む）"]


def main(plan_path):
    plan = json.loads(Path(plan_path).read_text(encoding='utf-8'))
    frames, fps = Path(plan['frames']), int(plan['fps'])
    size = tuple(plan['size'])

    def records(run):
        run = Path(run)
        events = [json.loads(l) for l in (run / 'events.jsonl').read_text(encoding='utf-8').splitlines()]
        return events, json.loads((run / 'result.json').read_text(encoding='utf-8'))
    out = ROOT / plan['out']
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(out, fps=fps, codec='libx264', macro_block_size=8,
                                ffmpeg_params=['-crf', '18', '-preset', 'slow', '-pix_fmt', 'yuv420p',
                                               '-movflags', '+faststart'])
    made_of = {s['frames']: json.loads((frames / s['frames'] / 'manifest.json').read_text(encoding='utf-8'))
               for s in plan['segments'] if 'frames' in s}
    parts, n_total = [], 0
    for i, seg in enumerate(plan['segments']):
        if 'card' in seg:
            c = dict(seg['card'])
            if c.pop('from_result', False):
                c['lines'] = result_lines(records(seg.get('run', plan.get('run')))[1])
            if 'from_table' in c:
                c['lines'] = table_lines(c.pop('from_table'), c.pop('policy', 'shinai_learned'))
            if c.pop('skip', False):
                before = next(s for s in reversed(plan['segments'][:i]) if 'frames' in s)
                after = next(s for s in plan['segments'][i + 1:] if 'frames' in s)
                run_before, run_after = before.get('run', plan.get('run')), after.get('run', plan.get('run'))
                if str(run_before) != str(run_after):
                    raise ValueError('a cut joins two parts of one run; the parts around it come from different runs')
                ev, _ = records(run_after)
                c.update(skip_card(ev, made_of[before['frames']]['to_s'], made_of[after['frames']]['from_s'], ev[0]['t']))
            img = card(size, c)
            n = int(round(seg['seconds'] * fps))
            for _ in range(n):
                writer.append_data(img)
            parts.append({'card': c['title'], 'lines': c.get('lines', []), 'seconds': n / fps})
            n_total += n
            continue
        if 'still' in seg:
            img = still(size, seg['still'])
            n = int(round(seg['seconds'] * fps))
            for _ in range(n):
                writer.append_data(img)
            parts.append({'still': seg['still']['image'], 'seconds': n / fps})
            n_total += n
            continue
        folder = frames / seg['frames']
        made = made_of[seg['frames']]
        run = seg.get('run', plan.get('run'))
        events, _ = records(run)
        total, t0 = events[0]['data']['slots'], events[0]['t']
        tines = json.loads((folder / 'tines.json').read_text(encoding='utf-8')) if (folder / 'tines.json').exists() else {}
        names = frames_until(sorted(p.name for p in (folder / 'frames_presentation').glob('*.jpg')), seg.get('until'))
        if plan.get('layout') == 'console':
            import console_view
            sig, eyes = console_view.load_signals(run), console_view.eye_views(run)
        for name in names:
            t = int(name[:-4]) / 1000
            inset_path = folder / 'frames_closeup' / name
            inset = Image.open(inset_path) if inset_path.exists() else None
            if plan.get('layout') == 'console':
                eye = console_view.eye_at(eyes, t)
                writer.append_data(console_view.screen(
                    Image.open(folder / 'frames_presentation' / name), inset, Image.open(eye) if eye else None,
                    sig, console_view.index_at(sig, t), t=t, t0=t0, step=step_at(events, t, tines.get(name, False)),
                    placed=placed_at(events, t), total=total, speed=made['speed'],
                    eye_name=console_view.eye_label(eyes, t)))
                continue
            writer.append_data(overlay(Image.open(folder / 'frames_presentation' / name), size,
                                       speed=made['speed'], t=t, t0=t0, events=events, total=total, inset=inset,
                                       tines_out=tines.get(name, False)))
        parts.append({'frames': seg['frames'], 'run': str(run), 'from_s': made['from_s'], 'to_s': min(made['to_s'], seg.get('until') or made['to_s']), 'speed': made['speed'],
                      'seconds': len(names) / fps, 'largest_difference_from_trace': made['largest_difference_from_trace'],
                      'rerender_commit': made['commit'], 'run_commit': made['run_commit']})
        n_total += len(names)
    writer.close()
    small = out.with_name(out.stem + '_1080p.mp4')
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-loglevel', 'error', '-i', str(out), '-vf',
                    'scale=1920:1080:flags=lanczos', '-c:v', 'libx264', '-crf', '20', '-preset', 'slow',
                    '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(small)], check=True)
    record = {'video': str(out), 'video_1080p': str(small), 'fps': fps, 'size': list(size),
              'duration_s': n_total / fps, 'parts': parts}
    out.with_suffix('.json').write_text(json.dumps(record, indent=1, ensure_ascii=False), encoding='utf-8',
                                        newline='\n')
    for p in parts:
        print(json.dumps(p, ensure_ascii=False))
    print(f'VIDEO {out} duration_s={n_total / fps:.1f}')


if __name__ == '__main__':
    main(sys.argv[1])
