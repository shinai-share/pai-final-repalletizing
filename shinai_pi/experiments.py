"""Episode records for repeated fixed tasks (experiment plan v01, chapters 4-5).

artifacts/experiments/<campaign>/<episode>/ holds:
  episode_manifest.json  ids, scenario and family, repeat, seeds, JST and UTC start, code commit and a hash
                         of uncommitted changes, config hash; 'partial' until complete() hashes every file
  trace.npz              one row per control step of what the robot observed and did (columns obs_*, cmd_*,
                         state, t). Ground truth never goes here.
  truth.json             evaluation only: the true poses, masses, centres of mass
  result.json            the separate verdicts (quality, safety, validity) and the end reason
An episode id is never reused and nothing is overwritten; an aborted episode keeps its record."""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TRACE_PREFIXES = ('obs_', 'cmd_')
TRACE_PLAIN = ('t', 'state')
JST = timezone(timedelta(hours=9))


def _git(*args):
    try:
        out = subprocess.run(['git', *args], cwd=ROOT, capture_output=True, timeout=20).stdout
        return out.decode('utf-8', errors='replace')     # diffs hold Japanese text: never the console's code page
    except (OSError, subprocess.SubprocessError):
        return ''


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class EpisodeStore:
    def __init__(self, root, campaign: str):
        self.dir = Path(root) / campaign
        self.dir.mkdir(parents=True, exist_ok=True)

    def open(self, episode_id: str, scenario_id: str, family: str, repeat: int, seeds: dict, extra: dict) -> 'Episode':
        d = self.dir / episode_id
        if d.exists():
            raise FileExistsError(f'episode {episode_id} already recorded; a rerun takes a new id')
        d.mkdir(parents=True)
        now = datetime.now(timezone.utc)
        config = (ROOT / 'configs' / 'cell.toml').read_bytes()
        manifest = {'episode_id': episode_id, 'campaign': self.dir.name, 'scenario_id': scenario_id, 'family': family,
                    'repeat': repeat, 'seeds': seeds, 'started_utc': now.isoformat(), 'started_jst': now.astimezone(JST).isoformat(),
                    'code_commit': _git('rev-parse', 'HEAD').strip() or 'unknown',
                    'code_dirty_hash': _sha(_git('diff', 'HEAD').encode('utf-8')),
                    'config_hash': _sha(config), 'status': 'partial', **extra}
        return Episode(d, manifest)


class Episode:
    def __init__(self, d: Path, manifest: dict):
        self.dir = d
        self.manifest = manifest
        self.rows = []
        self._write('episode_manifest.json', manifest)

    def _write(self, name, obj):
        (self.dir / name).write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=float), encoding='utf-8')

    def add_trace(self, row: dict):
        bad = [k for k in row if k not in TRACE_PLAIN and not k.startswith(TRACE_PREFIXES)]
        if bad:
            raise ValueError(f'not an observation or a command: {bad} (truth goes to truth.json)')
        self.rows.append(row)

    def write_truth(self, truth: dict):
        self._write('truth.json', truth)

    def write_result(self, result: dict):
        self._write('result.json', result)

    def _save_trace(self):
        cols = sorted({k for r in self.rows for k in r})
        arrays = {}
        for c in cols:
            vals = [r.get(c) for r in self.rows]
            if c == 'state':
                arrays[c] = np.array([str(v) for v in vals])
            else:
                arrays[c] = np.array([np.nan if v is None else v for v in vals], dtype=float)
        np.savez_compressed(self.dir / 'trace.npz', **arrays)

    def _finish(self, status, **extra):
        self._save_trace()
        hashes = {p.relative_to(self.dir).as_posix(): _sha(p.read_bytes()) for p in sorted(self.dir.rglob('*'))
                  if p.is_file() and p.name != 'episode_manifest.json'}
        self.manifest.update(status=status, hashes=hashes, rows=len(self.rows), **extra)
        self._write('episode_manifest.json', self.manifest)

    def complete(self):
        self._finish('complete')

    def abort(self, reason: str):
        self._finish('aborted', reason=reason)
