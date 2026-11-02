"""Record real sensor frames of collapsed pallets for offline perception development.

frames/<seed>/<view>.npz : depth, instance ids, id->prim map, view & projection matrices
frames/<seed>/truth.json : case poses (evaluation only; never read by perception)
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shinai_pi.config import load_cell  # noqa: E402
from shinai_pi.layout import initial_cases  # noqa: E402
from shinai_pi.motion import plan_move  # noqa: E402
from shinai_pi.transport import sample_transport  # noqa: E402
from shinai_pi.viewpoints import side_viewpoints, park_pose  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--seeds', type=int, nargs='+', default=[3, 8])
args = ap.parse_args()
cell = load_cell()
sku = cell.skus[0]
root = Path(__file__).resolve().parents[1] / 'artifacts' / 'frames'

from shinai_pi.sim.runtime import IsaacRuntime, run_transport  # noqa: E402
from PIL import Image  # noqa: E402


def go(rt, q):
    prof = plan_move(rt.gantry_target, q, cell.gantry.v_max, cell.gantry.a_max)
    n = int(math.ceil(prof.duration / cell.physics.dt))
    for k in range(n + 1):
        rt.set_gantry_target(prof.position(k * cell.physics.dt))
        rt.step(1)
    rt.step(30)


def save(rt, d, name, key):
    f = rt.capture(key)
    np.savez_compressed(d / f'{name}.npz', depth=f['depth'], instance=f['instance'], view=f['view'], projection=f['projection'],
                        resolution=np.array(f['resolution']), stamp=f['stamp'],
                        instance_map=json.dumps(f['instance_map']), gantry=rt.gantry_state()[0])
    Image.fromarray(f['rgb']).save(d / f'{name}.png')


rt = IsaacRuntime(cell)
for seed in args.seeds:
    prof = sample_transport(cell.transport, cell.layout.layers, seed)
    sp = initial_cases(cell, sku)
    rt.new_episode()
    rt.build(sp, wrapped_layers=[l for l in range(cell.layout.layers) if l not in prof.loose_layers])
    rt.start()
    d = root / f'seed{seed:03d}'
    d.mkdir(parents=True, exist_ok=True)
    go(rt, park_pose(cell))
    run_transport(rt, prof)
    save(rt, d, 'overhead_a', 'overhead_a')
    for vp in side_viewpoints(cell, cell.pallet.a_center):
        go(rt, np.array(vp.q))
        save(rt, d, f'tool_{vp.name}', 'tool')
    truth = {cid: {'pos': p.tolist(), 'quat_wxyz': q.tolist()} for cid, (p, q) in rt.truth_case_poses().items()}
    (d / 'truth.json').write_text(json.dumps({'seed': seed, 'sku': sku.id, 'cases': truth}, indent=1), encoding='utf-8')
    print('CAPTURED', seed, sorted(p.name for p in d.glob('*.npz')))
rt.close()
