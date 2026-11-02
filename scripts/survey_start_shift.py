"""The start state a stage S3 mission (two layers) would see, per seed, judged against the start-state rule
(docs/start_state_selection.md). Same steps as scripts/run_mission.py up to the end of the transport.

    python scripts/survey_start_shift.py [--grid COLS ROWS LAYERS] OUT_DIR SEED [SEED ...]"""
import dataclasses
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shinai_pi.config import load_cell  # noqa: E402
from shinai_pi.evaluation.truth import classify  # noqa: E402
from shinai_pi.layout import initial_cases  # noqa: E402
from shinai_pi.transport import sample_transport  # noqa: E402
from shinai_pi.variation import vary_load  # noqa: E402

args = sys.argv[1:]
GRID = None
if args and args[0] == '--grid':                  # --grid COLS ROWS LAYERS: a smaller demonstration load
    GRID, args = tuple(int(x) for x in args[1:4]), args[4:]
OUT = Path(args[0])
seeds = [int(s) for s in args[1:]]
try:
    cell = load_cell()
    cell = dataclasses.replace(cell, layout=dataclasses.replace(cell.layout, layers=2))
    if GRID:
        from shinai_pi.layout import with_grid
        cell = with_grid(cell, *GRID)
    from shinai_pi.sim.runtime import IsaacRuntime, run_transport
    from shinai_pi.viewpoints import park_pose
    from PIL import Image
    rt = IsaacRuntime(cell)
    OUT.mkdir(parents=True, exist_ok=True)
    sku = cell.sku('BEV-500x24')
    for seed in seeds:
        profile = sample_transport(cell.transport, cell.layout.layers, seed)
        spawns = vary_load(initial_cases(cell, sku), cell.variation, seed)
        rt.new_episode()
        rt.build(spawns, wrapped_layers=[l for l in range(cell.layout.layers) if l not in profile.loose_layers])
        rt.start()
        rt.set_gantry_target(park_pose(cell))
        rt.step(int(1.0 / cell.physics.dt))
        info = run_transport(rt, profile)
        poses = rt.truth_case_poses()
        cls = classify(spawns, poses, cell.pallet.a_center, cell.pallet.size)
        top = [s.case_id for s in spawns if s.pos[2] > min(x.pos[2] for x in spawns) + 0.1]
        moved = np.array([cls[c]['moved_m'] for c in top])
        moved_top = sum(cls[c]['state'] in ('shifted', 'tilted') for c in top)
        bad = sum(v['state'] in ('floor', 'fallen') for v in cls.values())
        row = {'seed': seed, 'brake': round(profile.brake_decel, 2), 'loose': list(profile.loose_layers),
               'top_n': len(top), 'top_moved_or_tilted': int(moved_top), 'top_median_m': round(float(np.median(moved)), 3),
               'top_max_m': round(float(moved.max()), 3), 'max_tilt_deg': max(v['tilt_deg'] for v in cls.values()),
               'floor_or_fallen': int(bad),
               'start': {k: [round(float(x), 3) for x in poses[k][0]] for k in sorted(poses)}}
        row['meets'] = (bad == 0 and moved_top * 2 >= len(top) and row['top_max_m'] >= 0.10 and row['top_median_m'] >= 0.04)
        (OUT / f'seed{seed:03d}.json').write_text(json.dumps({**row, 'classes': cls}, indent=1), encoding='utf-8')
        Image.fromarray(rt.render_rgb('presentation')).save(OUT / f'seed{seed:03d}.jpg', quality=88)
        print('SURVEY', json.dumps({k: v for k, v in row.items() if k != 'start'}), flush=True)
    rt.close()
except BaseException as e:
    print('FAILED', type(e).__name__, e, flush=True)
    os._exit(1)
