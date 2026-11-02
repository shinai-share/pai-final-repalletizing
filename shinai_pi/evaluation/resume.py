"""Resuming a stopped evaluation: a run counts as done once its result.json is written (the last thing it
writes), so a run cut off part-way leaves no result and is run again from its start."""
from __future__ import annotations

import json
import re
from pathlib import Path

TRIAL_DIR = re.compile(r'^trial_\d{2}$')


def pending(ids, folder_of) -> list:
    return [i for i in ids if not (folder_of(i) / 'result.json').exists()]


def gate_rows(base: Path) -> list:
    """Every finished trial of a gate folder, whichever process ran it."""
    return [json.loads((d / 'result.json').read_text(encoding='utf-8')) for d in sorted(Path(base).iterdir())
            if d.is_dir() and TRIAL_DIR.match(d.name) and (d / 'result.json').exists()]
