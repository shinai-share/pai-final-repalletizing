"""Gate G0: the gantry picks one 12 kg case from pallet A by suction and places it on its
slot on pallet B (0.8 m gap). Writes artifacts/g0.json as evidence."""
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shinai_pi.config import load_cell  # noqa: E402
from shinai_pi.layout import pallet_slots  # noqa: E402
from shinai_pi.motion import plan_move  # noqa: E402
from shinai_pi.layout import CaseSpawn  # noqa: E402
from shinai_pi.sim.runtime import IsaacRuntime  # noqa: E402

cell = load_cell()
sku = cell.sku('BEV-500x24')
src = pallet_slots(cell, sku, cell.pallet.a_center)[4]
dst = pallet_slots(cell, sku, cell.pallet.b_center)[0]
rt = IsaacRuntime(cell)
rt.build([CaseSpawn('case_00', sku.id, src.center, src.yaw_deg, 0)])
rt.start()
g = cell.gantry
log = []


def vram():
    out = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], capture_output=True, text=True).stdout
    return int(out.strip())


def move(goal, label, settle=0.3):
    start = rt.gantry_target.copy()
    prof = plan_move(start, goal, g.v_max, g.a_max)
    n = int(math.ceil(prof.duration / cell.physics.dt))
    for k in range(n + 1):
        rt.set_gantry_target(prof.position(k * cell.physics.dt))
        rt.step(1)
    rt.step(int(settle / cell.physics.dt))
    q, _ = rt.gantry_state()
    err = np.abs(q - goal)
    log.append({'move': label, 'duration_s': round(prof.duration, 3), 'final_error': [round(float(e), 4) for e in err], 'gripped': rt.gripped()})
    return q


z_of = lambda face_z: face_z - g.origin_height  # noqa: E731
top_a = src.center[2] + sku.size[2] / 2
safe = z_of(cell.pallet.size[2] + cell.layout.layers * sku.size[2] + g.travel_clearance + sku.size[2])
t0 = time.perf_counter()
rt.step(int(0.5 / cell.physics.dt))
move(np.array([src.center[0], src.center[1], safe, 0.0]), 'above_source')
move(np.array([src.center[0], src.center[1], z_of(top_a + 0.004), 0.0]), 'descend')
rt.grip(True)
rt.step(int(0.4 / cell.physics.dt))
held_after_close = rt.gripped()
move(np.array([src.center[0], src.center[1], safe, 0.0]), 'lift')
move(np.array([dst.center[0], dst.center[1], safe, math.radians(dst.yaw_deg)]), 'traverse')
top_b = dst.center[2] + sku.size[2] / 2
move(np.array([dst.center[0], dst.center[1], z_of(top_b + 0.006), math.radians(dst.yaw_deg)]), 'lower')
rt.grip(False)
rt.step(int(0.3 / cell.physics.dt))
move(np.array([dst.center[0], dst.center[1], safe, math.radians(dst.yaw_deg)]), 'retreat')
rt.step(int(2.0 / cell.physics.dt))
pos, quat = rt.truth_case_poses()['case_00']
yaw = math.degrees(2 * math.atan2(quat[3], quat[0]))
frame = rt.capture('overhead_b')
result = {
    'gate': 'G0', 'sku': sku.id, 'mass_kg': sku.mass,
    'held_after_close': held_after_close,
    'final_position': [round(float(v), 4) for v in pos],
    'slot_position': [round(v, 4) for v in dst.center],
    'position_error_m': round(float(np.linalg.norm(pos - np.array(dst.center))), 4),
    'yaw_deg': round(yaw, 2),
    'sim_seconds': round(rt.sim_time, 2), 'wall_seconds': round(time.perf_counter() - t0, 2),
    'boot_seconds': round(rt.boot_seconds, 1), 'vram_used_mib': vram(),
    'camera_depth_range_m': [round(float(np.nanmin(frame['depth'])), 3), round(float(np.nanmax(frame['depth'][np.isfinite(frame['depth'])])), 3)],
    'camera_instances': sorted(set(frame['instance_map'].values()))[:8],
    'moves': log,
}
result['passed'] = bool(held_after_close == ['case_00'] and result['position_error_m'] < 0.02)
out = Path(__file__).resolve().parents[1] / 'artifacts' / 'g0.json'
out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
from PIL import Image  # noqa: E402
Image.fromarray(frame['rgb']).save(out.with_name('g0_overhead_b.png'))
print(json.dumps(result, indent=2, ensure_ascii=False))
rt.close()
