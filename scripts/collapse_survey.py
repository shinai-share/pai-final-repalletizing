"""Survey how the transport event collapses a full pallet across seeds.

Writes artifacts/collapse/<seed>.json (truth classes, for evaluation only) and images."""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shinai_pi.config import load_cell  # noqa: E402
from shinai_pi.evaluation.truth import classify, summarize  # noqa: E402
from shinai_pi.layout import initial_cases  # noqa: E402
from shinai_pi.transport import sample_transport  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3])
ap.add_argument('--sku', default='BEV-500x24')
args = ap.parse_args()
cell = load_cell()
sku = cell.sku(args.sku)
out_dir = Path(__file__).resolve().parents[1] / 'artifacts' / 'collapse'
out_dir.mkdir(parents=True, exist_ok=True)

from shinai_pi.sim.runtime import IsaacRuntime, run_transport  # noqa: E402
from PIL import Image  # noqa: E402

rows = []
rt = IsaacRuntime(cell)
for seed in args.seeds:
    profile = sample_transport(cell.transport, cell.layout.layers, seed)
    spawns = initial_cases(cell, sku)
    rt.new_episode()
    wrapped = [l for l in range(cell.layout.layers) if l not in profile.loose_layers]
    rt.build(spawns, wrapped_layers=wrapped)
    rt.start()
    rt.set_gantry_target([cell.pallet.b_center[0] + 0.9, 0, 0, 0])    # park clear of camera A
    rt.step(60)
    t0 = time.perf_counter()
    info = run_transport(rt, profile)
    classes = classify(spawns, rt.truth_case_poses(), cell.pallet.a_center, cell.pallet.size)
    s = summarize(classes)
    frame = rt.capture('overhead_a')
    Image.fromarray(frame['rgb']).save(out_dir / f'seed{seed:03d}_overhead_a.png')
    row = {'seed': seed, 'brake_decel': round(profile.brake_decel, 2), 'heading_deg': round(profile.heading * 57.2958, 1),
           'loose_layers': list(profile.loose_layers), **info, 'counts': s, 'wall_s': round(time.perf_counter() - t0, 1)}
    (out_dir / f'seed{seed:03d}.json').write_text(json.dumps({**row, 'classes': classes}, indent=1), encoding='utf-8')
    rows.append(row)
    print('SURVEY', json.dumps(row))
rt.close()
