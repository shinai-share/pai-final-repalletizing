"""Headless Isaac Sim start-up check: physics steps, render frame, peak VRAM. Writes JSON evidence."""
import json
import subprocess
import time
from pathlib import Path

from isaacsim import SimulationApp

t0 = time.perf_counter()
app = SimulationApp({'headless': True, 'width': 640, 'height': 480})
boot = time.perf_counter() - t0

import numpy as np  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.api.objects import DynamicCuboid  # noqa: E402

world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()
cube = world.scene.add(DynamicCuboid(prim_path='/World/box', name='box', position=np.array([0, 0, 1.0]), size=0.3, mass=6.0))
world.reset()
t1 = time.perf_counter()
for _ in range(240):
    world.step(render=False)
physics = time.perf_counter() - t1
t2 = time.perf_counter()
for _ in range(30):
    world.step(render=True)
render = time.perf_counter() - t2
z = float(cube.get_world_pose()[0][2])


def vram():
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10).stdout
        used, total = out.strip().split(',')
        return {'used_mib': int(used), 'total_mib': int(total)}
    except Exception as e:  # evidence must still be written
        return {'error': type(e).__name__}


result = {
    'isaacsim': '6.2.0.0', 'boot_seconds': round(boot, 1),
    'physics_steps_per_second': round(240 / physics, 1), 'render_fps': round(30 / render, 1),
    'box_rest_height_m': round(z, 3), 'box_settled': abs(z - 0.15) < 0.02, 'gpu': vram(),
}
out = Path(__file__).resolve().parents[1] / 'artifacts' / 'isaac-smoke.json'
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2))
app.close()
