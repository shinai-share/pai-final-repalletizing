"""Make the submission video by filming the Inspector replay and text cards.

Every replay frame is the real Inspector screen at a given simulated time; nothing is edited in.
A plan lists cards and replay spans in order:

    {"run": "shinai_seed008", "speed": 2, "fps": 15,
     "segments": [{"card": {"title": "...", "lines": ["..."]}, "seconds": 6},
                  {"replay": [0, 150]},
                  {"card": {"title": "中略", "note": "..."}, "seconds": 3},
                  {"replay": [400, 470]},
                  {"replay": [0, 40], "run": "stage_side_seed000"}]}     # another recorded run

    python scripts/make_video.py plans/submission.json
Writes artifacts/video/<name>.mp4 and prints the duration of each part."""
import io
import json
import subprocess
import sys
import time
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PORT = 8791


def result_lines(r):
    """The results card, written from the run's own result file."""
    lines = [f"B に積んだケース {r['placed_on_b']} 個（置いた位置の誤差 最大 {r['placement_xy_err_max_m'] * 100:.1f} cm）",
             f"横倒しで別置きにし人に渡したケース {r['handed_over']} 個・搬送中の落下 {r['drops_in_transit']} 回",
             f"安全停止 {r['safety_stops']} 回・所要時間 {r['sim_seconds'] / 60:.1f} 分（シミュレーション）"]
    known_left = len(r.get('remaining') or {})
    if known_left:
        lines.append(f"A 側に残し、理由を人に報告したケース {known_left} 個")
    if r.get('not_found'):
        lines.append(f"積荷の個数と照合し、作業場の外へ飛んだ可能性のある {r['not_found']} 個を人に報告")
    return lines


def _end_time(run):
    events = (ROOT / 'artifacts' / 'runs' / run / 'events.jsonl').read_text(encoding='utf-8').splitlines()
    return json.loads(events[-1])['t']


def main(plan_path):
    plan = json.loads(Path(plan_path).read_text(encoding='utf-8'))
    run, speed, fps = plan['run'], float(plan.get('speed', 2)), int(plan.get('fps', 15))
    out = ROOT / 'artifacts' / 'video' / f"{Path(plan_path).stem}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    server = subprocess.Popen([sys.executable, '-m', 'shinai_pi.inspector_server', '--port', str(PORT)], cwd=ROOT,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        time.sleep(1.5)
        import imageio.v2 as imageio
        import numpy as np
        from PIL import Image
        from playwright.sync_api import sync_playwright
        writer = imageio.get_writer(out, fps=fps, codec='libx264', quality=8, macro_block_size=8,
                                    ffmpeg_params=['-pix_fmt', 'yuv420p', '-movflags', '+faststart'])
        total, report = 0, []
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            # The capture tool drives the page from outside; the screen's own CSP stays strict for people.
            page = browser.new_page(viewport={'width': 1280, 'height': 720}, device_scale_factor=1.5, bypass_csp=True)   # 1920 x 1080

            def shot():
                return np.asarray(Image.open(io.BytesIO(page.screenshot(type='png'))).convert('RGB'))

            replay_loaded = None
            result = json.loads((ROOT / 'artifacts' / 'runs' / run / 'result.json').read_text(encoding='utf-8'))
            events = (ROOT / 'artifacts' / 'runs' / run / 'events.jsonl').read_text(encoding='utf-8').splitlines()
            end_t = json.loads(events[-1])['t']
            for seg in plan['segments']:
                if 'card' in seg:
                    c = dict(seg['card'])
                    if c.pop('from_result', False):
                        c['lines'] = result_lines(result)
                    qs = [('title', c.get('title', '')), ('kicker', c.get('kicker', '')), ('note', c.get('note', ''))]
                    qs += [('line', l) for l in c.get('lines', [])]
                    page.goto(f'http://127.0.0.1:{PORT}/card.html?' + urllib.parse.urlencode(qs))
                    page.wait_for_function('window.__inspectorReady === true', timeout=30000)
                    img = shot()
                    n = int(round(seg['seconds'] * fps))
                    for _ in range(n):
                        writer.append_data(img)
                    replay_loaded = None
                    report.append(('札', c.get('title', ''), n / fps))
                    total += n
                    continue
                seg_run = seg.get('run', run)
                seg_end = end_t if seg_run == run else _end_time(seg_run)
                t0, t1 = (seg_end - seg['replay']['last'], seg_end) if isinstance(seg['replay'], dict) else seg['replay']
                seg_speed = float(seg.get('speed', speed))     # shown on screen as the playback speed
                if replay_loaded != (seg_run, seg_speed):
                    page.goto(f'http://127.0.0.1:{PORT}/?run={seg_run}&capture=1&speed={seg_speed:g}')
                    page.wait_for_function('window.__inspectorReady === true', timeout=30000)
                    replay_loaded = (seg_run, seg_speed)
                step = seg_speed / fps                  # simulated seconds per video frame
                n = int((t1 - t0) / step)
                for i in range(n):
                    page.evaluate(f'window.__seek({t0 + i * step})')
                    writer.append_data(shot())
                report.append(('再生', f'{seg_run} {t0:.0f}〜{t1:.0f} 秒（{seg_speed:g} 倍速）', n / fps))
                total += n
            browser.close()
        writer.close()
        for kind, what, secs in report:
            print(f'{kind:4s} {secs:6.1f} 秒  {what}')
        print(f'VIDEO {out} duration_s={total / fps:.1f}')
    finally:
        server.terminate()


if __name__ == '__main__':
    main(sys.argv[1])
