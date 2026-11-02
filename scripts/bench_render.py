"""Measure wall time of camera captures and physics in the full cell (evidence for performance)."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shinai_pi.config import load_cell  # noqa: E402
from shinai_pi.layout import initial_cases  # noqa: E402
from shinai_pi.sim.runtime import IsaacRuntime  # noqa: E402

cell = load_cell()
rt = IsaacRuntime(cell)
rt.build(initial_cases(cell, cell.skus[0]), wrapped_layers=[0, 1, 2, 3])
rt.start()
out = {}
for key in ('overhead_a', 'overhead_b', 'tool'):
    rt.capture(key)
rt.render_rgb('presentation')
t = time.perf_counter()
for _ in range(20):
    rt.render_rgb('presentation')
out['presentation_frame_ms'] = round((time.perf_counter() - t) / 20 * 1000, 1)
t = time.perf_counter()
for _ in range(10):
    rt.capture('overhead_a')
out['overhead_capture_ms'] = round((time.perf_counter() - t) / 10 * 1000, 1)
t = time.perf_counter()
rt.step(240)
out['physics_steps_per_s'] = round(240 / (time.perf_counter() - t), 1)
t = time.perf_counter()
for _ in range(240):
    rt.gantry_state(); rt.gripped()
out['state_read_ms'] = round((time.perf_counter() - t) / 240 * 1000, 2)
print('BENCH', json.dumps(out))
(Path(__file__).resolve().parents[1] / 'artifacts' / 'bench_render.json').write_text(json.dumps(out, indent=1))
rt.close()
