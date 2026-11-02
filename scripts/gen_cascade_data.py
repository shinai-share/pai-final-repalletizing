"""Self-supervised training data for the cascade predictor, generated in Isaac Sim physics.

For each collapsed pile (transport seed, randomised friction, and per-case mass and centre of mass): take each free case away,
let physics run, record which neighbours actually moved, restore the pile. Features come from a
belief built on poses with perception-like noise. Ground truth is used only as the label.

    python scripts/gen_cascade_data.py --seeds 100 199 --out artifacts/cascade/train.npz
    python scripts/gen_cascade_data.py --seeds 2000 2199 --split train --out artifacts/cascade_v3/train_a.npz
With --input camera the features come from the belief the robot builds from its overhead camera at the start of a
mission (Executive.look), instead of the true poses with noise; the label is the same.
With --split the piles come from shinai_pi/scenarios.py (layers, product and transport vary by seed) and only the seeds
of that split are used: 'train', 'seen' (training's conditions, other seeds) or 'unseen' (conditions kept out).
"""
import argparse
import dataclasses
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shinai_pi.config import load_cell  # noqa: E402
from shinai_pi import scenarios  # noqa: E402
from shinai_pi.layout import initial_cases  # noqa: E402
from shinai_pi.variation import vary_load  # noqa: E402
from shinai_pi.learning.cascade import FEATURES, removal_features  # noqa: E402
from shinai_pi.perception.obb import OBB, quat_to_matrix, yaw_matrix  # noqa: E402
from shinai_pi.pile import Box, SupportGraph  # noqa: E402
from shinai_pi.transport import sample_transport  # noqa: E402

POS_NOISE = 0.01            # m, like the measured perception error of well-seen cases
YAW_NOISE_DEG = 1.5
MOVED_M, DROP_M = 0.05, 0.10
SETTLE_S = 1.5
MAX_REMOVALS = 8
CAMERAS = ('overhead_a',)     # the fixed camera the robot first looks through; side looks move the gantry


def noisy_boxes(truth, size, rng):
    out = []
    for k, (p, q) in truth.items():
        R = yaw_matrix(math.radians(rng.normal(0, YAW_NOISE_DEG))) @ quat_to_matrix(q)
        out.append(Box(k, tuple(p + rng.normal(0, POS_NOISE, 3)), tuple(size), rotation=R))
    return out


INPUTS = ('truth_noise', 'camera')


def belief_graph(kind, truth, size, com_margin, rng, look, static=()):
    """The support graph the features are computed on: true poses with perception-like noise, or what the cameras
    gave the robot (look() returns its belief)."""
    if kind == 'truth_noise':
        return SupportGraph(noisy_boxes(truth, size, rng), floor=0.0, static=list(static), com_margin=com_margin)
    if kind == 'camera':
        return look().graph()
    raise ValueError(f'input must be one of {INPUTS}')


def look_from_cameras(rt, cell, sku):
    """The robot's own first look at pallet A: the overhead camera through the mission's perception."""
    from shinai_pi.events import EventLog
    from shinai_pi.executive import Executive, MissionConfig
    from shinai_pi.sim.isaac_backend import IsaacBackend
    return Executive(cell, IsaacBackend(rt), EventLog('data'), MissionConfig(sku.id)).look(CAMERAS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs=2, required=True, help='first and last seed (inclusive)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--split', default=None, help='train | seen | unseen: draw piles from the collapse scenarios')
    ap.add_argument('--input', default='truth_noise', choices=INPUTS, help='where the features come from')
    args = ap.parse_args()
    base = load_cell()
    from shinai_pi.sim.runtime import IsaacRuntime, run_transport
    rt = IsaacRuntime(base)
    rows_X, rows_y, rows_meta = [], [], []
    t_start = time.perf_counter()
    seeds = (scenarios.seeds_of(base, args.split, args.seeds[0], args.seeds[1]) if args.split
             else range(args.seeds[0], args.seeds[1] + 1))
    for seed in seeds:
        rng = np.random.default_rng(10_000 + seed)
        mu = float(rng.uniform(0.35, 0.65))                     # domain randomisation of friction (mass: per case, below)
        if args.split:
            scen = scenarios.sample(base, seed)
            sku = dataclasses.replace(scen.sku, friction=(mu, 0.8 * mu))
            cell = dataclasses.replace(scen.cell, skus=tuple(sku if k.id == sku.id else k for k in scen.cell.skus))
            profile, spawns, meta = scen.profile, scen.spawns, scen.meta
        else:
            sku = dataclasses.replace(base.skus[0], friction=(mu, 0.8 * mu))
            cell = dataclasses.replace(base, skus=(sku,) + base.skus[1:])
            profile = sample_transport(cell.transport, cell.layout.layers, seed)
            # Per-case contents as in the missions: mass and centre of mass differ case by case.
            spawns, meta = vary_load(initial_cases(cell, sku), cell.variation, seed), {'seed': seed}
        rt.cell = cell
        rt.new_episode()
        rt.build(spawns, wrapped_layers=[l for l in range(cell.layout.layers) if l not in profile.loose_layers])
        rt.start()
        rt.step(30)
        run_transport(rt, profile)
        p = cell.pallet
        pallet = OBB.make((p.a_center[0], p.a_center[1], p.size[2] / 2), p.size)
        truth = rt.truth_case_poses()
        # Same belief as in the missions: the contents may sit off centre (the SKU's prior).
        g = belief_graph(args.input, truth, sku.size, sku.com_uncertainty, rng, lambda: look_from_cameras(rt, cell, sku),
                         static=[pallet])
        # Half free cases, half cases that hold something up: the decisions that matter are about the latter.
        flat = [k for k in g.boxes if g.obbs[k].tilt_deg() <= cell.gripper.max_face_tilt_deg]
        free_set = g.free()
        holders = [k for k in flat if k not in free_set]
        tops = [k for k in flat if k in free_set]
        rng.shuffle(holders)
        rng.shuffle(tops)
        free = holders[:MAX_REMOVALS // 2] + tops[:MAX_REMOVALS - min(len(holders), MAX_REMOVALS // 2)]
        pos0, quat0 = (np.asarray(a.numpy()).copy() for a in rt.cases.get_world_poses())
        ids = rt.case_ids
        for removed in free[:MAX_REMOVALS]:
            keys, X = removal_features(g, removed)
            if not keys:
                continue
            i = ids.index(removed)
            far = pos0.copy()
            far[i] = [6.0 + i * 0.5, 4.0, 0.2]           # take it away (out of the cell)
            rt.cases.set_world_poses(positions=far, orientations=quat0)
            rt.cases.set_velocities(np.zeros((len(ids), 3)), np.zeros((len(ids), 3)))
            rt.step(int(SETTLE_S / cell.physics.dt))
            after = rt.truth_case_poses()
            for k, x in zip(keys, X):
                d = after[k][0] - truth[k][0]
                moved = float(np.linalg.norm(d[:2])) > MOVED_M or -d[2] > DROP_M
                rows_X.append(x)
                rows_y.append(int(moved))
                rows_meta.append({**meta, 'removed': removed, 'other': k, 'friction': sku.friction[0], 'input': args.input})
            rt.cases.set_world_poses(positions=pos0, orientations=quat0)       # restore the pile
            rt.cases.set_velocities(np.zeros((len(ids), 3)), np.zeros((len(ids), 3)))
            rt.step(6)
            back = np.asarray(rt.cases.get_world_poses()[0].numpy())
            if float(np.max(np.linalg.norm(back - pos0, axis=1))) > 0.01:
                print(f'seed {seed}: restore drift {np.max(np.linalg.norm(back - pos0, axis=1)):.3f} m, stopping this pile', flush=True)
                break
        print(f'seed {seed}: samples {len(rows_y)} positives {sum(rows_y)} elapsed {time.perf_counter() - t_start:.0f}s', flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, X=np.array(rows_X), y=np.array(rows_y), features=np.array(FEATURES))
    out.with_suffix('.meta.jsonl').write_text('\n'.join(json.dumps(m) for m in rows_meta), encoding='utf-8')
    print('DATA', out, len(rows_y), sum(rows_y))
    rt.close()


if __name__ == '__main__':
    main()
