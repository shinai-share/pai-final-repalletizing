"""Collapse scenarios: what learning and the runs that judge it are drawn from (owner 2026-10-31: adapt to collapses
of every kind, not to one). A seed fixes the load (layers, product, per-case contents) and the transport that
collapses it. Some conditions never reach training, so that adapting to them can be measured."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from .layout import initial_cases
from .transport import sample_transport
from .variation import vary_load

LAYERS = (2, 3, 4)
BRAKE = (3.5, 9.0)              # m/s^2: the configured range (3.5-7.0) widened to harder stops, with more cases down
TRAIN_BRAKE_MAX = 8.0           # harder stops than this are kept out of training
HELD_OUT_LAYERS = (3,)          # three-layer loads are kept out of training
SPLITS = ('train', 'seen', 'unseen')


@dataclass(frozen=True)
class Scenario:
    seed: int
    cell: object
    sku: object
    spawns: list
    profile: object
    meta: dict


def sample(base, seed: int) -> Scenario:
    rng = np.random.default_rng(20_000 + seed)
    layers = int(rng.choice(LAYERS))
    sku = base.skus[int(rng.integers(len(base.skus)))]
    transport = dataclasses.replace(base.transport, brake_decel=BRAKE, wrap_failure_layers=(1, layers))
    cell = dataclasses.replace(base, layout=dataclasses.replace(base.layout, layers=layers), transport=transport)
    profile = sample_transport(cell.transport, layers, seed)
    spawns = vary_load(initial_cases(cell, sku), cell.variation, seed)
    meta = {'seed': seed, 'layers': layers, 'sku': sku.id, 'brake_decel': round(profile.brake_decel, 3),
            'loose_layers': len(profile.loose_layers)}
    return Scenario(seed, cell, sku, spawns, profile, meta)


def split(meta: dict) -> str:
    """'unseen': a condition kept out of training; 'seen': the conditions of training, on seeds kept out of it."""
    if meta['layers'] in HELD_OUT_LAYERS or meta['brake_decel'] > TRAIN_BRAKE_MAX:
        return 'unseen'
    return 'train' if meta['seed'] % 5 else 'seen'


def seeds_of(base, name: str, first: int, last: int) -> list:
    if name not in SPLITS:
        raise ValueError(f'unknown split {name!r}; one of {SPLITS}')
    return [s for s in range(first, last + 1) if split(sample(base, s).meta) == name]
