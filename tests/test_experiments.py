"""Episode records for repeated fixed tasks and the learning built on them (experiment plan v01,
chapter 5): one folder per episode, never overwritten, truth apart from what the robot observed,
and complete only when every file is hashed."""
import json

import numpy as np
import pytest

from shinai_pi.experiments import EpisodeStore


def open_one(tmp_path, eid='e001', **kw):
    store = EpisodeStore(tmp_path, 'tilt30')
    args = dict(scenario_id='tilt30-s0', family='tilt30-f0', repeat=0, seeds={'scene': 0}, extra={})
    args.update(kw)
    return store, store.open(eid, **args)


def test_an_episode_starts_partial_and_completes_with_hashes(tmp_path):
    _, ep = open_one(tmp_path)
    m = json.loads((ep.dir / 'episode_manifest.json').read_text(encoding='utf-8'))
    assert m['status'] == 'partial' and m['code_commit'] and m['config_hash']
    ep.add_trace({'t': 0.0, 'obs_tine_l': 1.0, 'obs_tine_r': 2.0, 'state': 'CARRY'})
    ep.add_trace({'t': 0.01, 'obs_tine_l': 1.5, 'obs_tine_r': 2.5, 'state': 'CARRY'})
    ep.write_truth({'mass_kg': 12.0})
    ep.write_result({'quality': 'success'})
    ep.complete()
    m = json.loads((ep.dir / 'episode_manifest.json').read_text(encoding='utf-8'))
    assert m['status'] == 'complete' and set(m['hashes']) >= {'trace.npz', 'truth.json', 'result.json'}
    tr = np.load(ep.dir / 'trace.npz')
    assert list(tr['obs_tine_l']) == [1.0, 1.5] and list(tr['state']) == ['CARRY', 'CARRY']


def test_an_episode_id_is_never_reused_and_nothing_is_overwritten(tmp_path):
    store, ep = open_one(tmp_path)
    ep.complete()
    with pytest.raises(FileExistsError):
        store.open('e001', scenario_id='x', family='x', repeat=0, seeds={}, extra={})


def test_truth_never_enters_the_trace(tmp_path):
    _, ep = open_one(tmp_path)
    with pytest.raises(ValueError):
        ep.add_trace({'t': 0.0, 'truth_mass': 12.0})


def test_an_aborted_episode_keeps_its_record_and_says_why(tmp_path):
    _, ep = open_one(tmp_path)
    ep.add_trace({'t': 0.0, 'obs_tine_l': 1.0})
    ep.abort('GPU のメモリ不足')
    m = json.loads((ep.dir / 'episode_manifest.json').read_text(encoding='utf-8'))
    assert m['status'] == 'aborted' and m['reason'] == 'GPU のメモリ不足' and (ep.dir / 'trace.npz').exists()


def test_files_in_subfolders_are_hashed_too(tmp_path):
    _, ep = open_one(tmp_path)
    (ep.dir / 'run').mkdir()
    (ep.dir / 'run' / 'events.jsonl').write_text('{}', encoding='utf-8')
    ep.complete()
    m = json.loads((ep.dir / 'episode_manifest.json').read_text(encoding='utf-8'))
    assert 'run/events.jsonl' in m['hashes']
