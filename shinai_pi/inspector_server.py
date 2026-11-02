"""Inspector server: serves the screen and the recorded run files, nothing else.

Local only (127.0.0.1). Run names and frame names are validated against fixed patterns, so no
request can reach a file outside a run's own directory.

    python -m shinai_pi.inspector_server --port 8770
"""
from __future__ import annotations

import argparse
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UI = ROOT / 'ui'
RUNS = ROOT / 'artifacts' / 'runs'
STATIC = {'/': ('index.html', 'text/html; charset=utf-8'), '/app.js': ('app.js', 'text/javascript; charset=utf-8'),
          '/style.css': ('style.css', 'text/css; charset=utf-8'), '/card.html': ('card.html', 'text/html; charset=utf-8'),
          '/card.js': ('card.js', 'text/javascript; charset=utf-8')}
RUN_RE = re.compile(r'^[a-z_]+_seed\d{3,5}$')        # collapse scenarios use four-digit seeds
FRAME_RE = re.compile(r'^\d{8}$')
VIEW_RE = re.compile(r'^\d{8}_[a-z_]+_view\.jpg$')    # perception_views.py


def read_jsonl(path: Path, since: int = -1):
    if not path.exists():
        return []
    out = []
    with path.open(encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:      # the writer may be mid-line in a live run
                break
            if d.get('seq', 0) > since:
                out.append(d)
    return out


def list_runs():
    runs = []
    if RUNS.exists():
        for d in sorted(RUNS.iterdir()):
            if d.is_dir() and RUN_RE.match(d.name):
                res = d / 'result.json'
                status = json.loads(res.read_text(encoding='utf-8')).get('status') if res.exists() else 'running'
                runs.append({'name': d.name, 'status': status, 'frames': (d / 'frames').exists()})
    return runs


EXTRA_VIEWS = ('side', 'closeup')      # recorded beside the presentation view (frames_<view>/)

class Handler(BaseHTTPRequestHandler):
    def _send(self, status, body: bytes, kind: str, cache: bool = False):
        self.send_response(status)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Cache-Control', 'max-age=3600' if cache else 'no-store')
        self.send_header('Content-Security-Policy', "default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj):
        self._send(200, json.dumps(obj, ensure_ascii=False).encode('utf-8'), 'application/json; charset=utf-8')

    def do_GET(self):
        path, _, query = self.path.partition('?')
        params = dict(p.split('=', 1) for p in query.split('&') if '=' in p)
        if path in STATIC:
            name, kind = STATIC[path]
            return self._send(200, (UI / name).read_bytes(), kind)
        if path == '/api/runs':
            return self._json(list_runs())
        parts = path.strip('/').split('/')
        if len(parts) >= 4 and parts[:2] == ['api', 'run'] and RUN_RE.match(parts[2]):
            run = RUNS / parts[2]
            if not run.is_dir():
                return self._send(404, b'{}', 'application/json')
            since = int(params.get('since', '-1')) if params.get('since', '-1').lstrip('-').isdigit() else -1
            what = parts[3]
            if what == 'events':
                return self._json(read_jsonl(run / 'events.jsonl', since))
            if what == 'beliefs':
                return self._json(read_jsonl(run / 'belief.jsonl', since))
            if what == 'result':
                res = run / 'result.json'
                return self._json(json.loads(res.read_text(encoding='utf-8')) if res.exists() else None)
            if what == 'frames':
                fd = run / 'frames'
                return self._json(sorted(int(p.stem) for p in fd.glob('*.jpg')) if fd.exists() else [])
            if what == 'perception' and len(parts) == 4:
                idx = run / 'perception' / 'index.json'
                return self._json(json.loads(idx.read_text(encoding='utf-8')) if idx.exists() else {'views': []})
            if what == 'perception' and len(parts) == 5 and VIEW_RE.match(parts[4]):
                f = run / 'perception' / parts[4]
                if f.exists():
                    return self._send(200, f.read_bytes(), 'image/jpeg', cache=True)
            if what == 'views':
                return self._json([v for v in EXTRA_VIEWS if (run / f'frames_{v}').is_dir()])
            if what == 'frame' and len(parts) == 5 and parts[4].endswith('.jpg') and FRAME_RE.match(parts[4][:-4]):
                f = run / 'frames' / parts[4]
                if f.exists():
                    return self._send(200, f.read_bytes(), 'image/jpeg', cache=True)
            if (what == 'frame' and len(parts) == 6 and parts[4] in EXTRA_VIEWS and parts[5].endswith('.jpg')
                    and FRAME_RE.match(parts[5][:-4])):
                f = run / f'frames_{parts[4]}' / parts[5]
                if f.exists():
                    return self._send(200, f.read_bytes(), 'image/jpeg', cache=True)
        return self._send(404, b'{}', 'application/json')

    def log_message(self, *args):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8770)
    args = ap.parse_args()
    print(f'Inspector: http://127.0.0.1:{args.port}/', flush=True)
    ThreadingHTTPServer(('127.0.0.1', args.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
