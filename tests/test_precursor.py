"""L1: does what the robot saw just before tell that the cups are about to let go? (experiment plan v01,
chapter 7). Samples come from recorded episodes; features use only the past and only observations;
episodes of one scene family stay on one side of the split."""
import numpy as np
import pytest

from shinai_pi.experiments import EpisodeStore
from shinai_pi.learning import precursor as pc


def record(store, eid, family, drop_at=None, warn=None, steps=600, dt=1 / 120):
    ep = store.open(eid, scenario_id=family, family=family, repeat=0, seeds={}, extra={})
    for k in range(steps):
        t = k * dt
        held = drop_at is None or t < drop_at
        tine = 50.0 + (40.0 if warn is not None and drop_at is not None and drop_at - warn <= t < drop_at else 0.0)
        ep.add_trace({'t': t, 'state': 'FORK_IN', 'obs_gripped': float(held), 'obs_tine_l': tine, 'obs_tine_r': 50.0,
                      'obs_head_fz': 1.0, 'obs_head_tx': 0.0, 'obs_head_ty': 0.0, 'obs_fork_ext': 0.22,
                      **{f'obs_q{i}': 0.0 for i in range(6)}, **{f'cmd_q{i}': 0.0 for i in range(6)}})
    ep.write_result({'quality': 'success'})
    ep.complete()


def test_labels_look_ahead_but_features_only_look_back(tmp_path):
    store = EpisodeStore(tmp_path, 'c')
    record(store, 'e1', 'f1', drop_at=3.0, warn=0.3)
    X, y, fam, t = pc.samples(tmp_path / 'c', horizon=0.5, every=0.1)
    assert y.sum() > 0 and y[t > 3.0].sum() == 0                     # after the drop the case is not held: no samples
    assert all(t[y == 1] >= 2.5 - 1e-9) and all(t[y == 1] < 3.0)
    # The window ends at the sample time: a value that only appears later cannot leak in.
    i = int(np.argmin(np.abs(t - 2.6)))
    assert X[i][pc.FEATURES.index('tine_asym')] < 1.0                 # the warning starts at 2.7


def test_a_family_never_sits_on_both_sides_of_the_split(tmp_path):
    fams = np.array([f'f{i % 7}' for i in range(200)])
    split = pc.split_by_family(fams, seed=0)
    for f in set(fams):
        assert len(set(split[fams == f])) == 1


def test_only_observations_are_features():
    assert all(not f.startswith('truth') for f in pc.FEATURES)


def test_a_learned_precursor_is_compared_with_a_simple_rule_on_held_out_families(tmp_path):
    store = EpisodeStore(tmp_path, 'c')
    rng = np.random.default_rng(0)
    for i in range(24):
        drops = i % 2 == 0
        record(store, f'e{i}', f'f{i}', drop_at=float(rng.uniform(2.0, 4.0)) if drops else None, warn=0.3 if drops else None)
    report = pc.train_and_compare(tmp_path / 'c', horizon=0.5, every=0.1, seed=0)
    assert set(report) >= {'learned', 'rule', 'split', 'samples'}
    assert report['learned']['pr_auc'] >= report['rule']['pr_auc'] - 1e-9
    assert report['split']['families']['test'] > 0
