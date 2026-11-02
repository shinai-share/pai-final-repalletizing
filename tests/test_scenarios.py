"""Collapse scenarios for learning that has to hold across collapses, not one (owner 2026-10-31): each seed gives a
load (layers, product) and a transport, and some conditions are kept out of training to test adapting to them."""
import pytest

from shinai_pi.config import load_cell
from shinai_pi import scenarios as sc

CELL = load_cell()


def test_a_seed_always_gives_the_same_scenario():
    a, b = sc.sample(CELL, 412), sc.sample(CELL, 412)
    assert a.meta == b.meta
    assert a.profile == b.profile


def test_the_scenarios_cover_every_layer_count_and_product():
    metas = [sc.sample(CELL, s).meta for s in range(1000, 1200)]
    assert {m['layers'] for m in metas} == set(sc.LAYERS)
    assert {m['sku'] for m in metas} == {k.id for k in CELL.skus}
    assert max(m['brake_decel'] for m in metas) > sc.TRAIN_BRAKE_MAX


def test_the_cell_of_a_scenario_has_its_layers_and_product():
    s = sc.sample(CELL, 1003)
    assert s.cell.layout.layers == s.meta['layers']
    assert s.sku.id == s.meta['sku']
    assert all(c.sku_id == s.sku.id for c in s.spawns)
    assert len(s.spawns) == len({c.case_id for c in s.spawns})


def test_only_top_layers_lose_their_wrap_and_at_least_one_does():
    for seed in range(1000, 1100):
        s = sc.sample(CELL, seed)
        n = s.meta['layers']
        loose = s.profile.loose_layers
        assert 1 <= len(loose) <= n and loose == tuple(range(n - len(loose), n))


def test_training_never_sees_the_held_out_conditions():
    for seed in range(1000, 1300):
        s = sc.sample(CELL, seed)
        unseen = s.meta['layers'] in sc.HELD_OUT_LAYERS or s.meta['brake_decel'] > sc.TRAIN_BRAKE_MAX
        assert sc.split(s.meta) == ('unseen' if unseen else 'train' if seed % 5 else 'seen')


def test_an_unknown_split_name_is_refused():
    with pytest.raises(ValueError):
        sc.seeds_of(CELL, 'later', 1000, 1010)


def test_seeds_of_a_split_are_listed_in_order():
    got = sc.seeds_of(CELL, 'unseen', 1000, 1100)
    assert got == sorted(got) and got
    assert all(sc.split(sc.sample(CELL, s).meta) == 'unseen' for s in got)
