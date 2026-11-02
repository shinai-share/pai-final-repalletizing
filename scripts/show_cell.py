"""Open the Isaac Sim window on the work cell, for people to look around (nothing is decided here).

Builds pallet A with a full load, plays the transport collapse of the given seed, parks the gantry
with the tines retracted, and leaves the window open. Orbit with Alt + left drag, pan with the
middle button, zoom with the wheel. Close the window to end.

    .venv-isaac/Scripts/python scripts/show_cell.py [--seed 8] [--intact]
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shinai_pi.config import load_cell  # noqa: E402
from shinai_pi.layout import initial_cases  # noqa: E402
from shinai_pi.transport import sample_transport  # noqa: E402
from shinai_pi.variation import vary_load  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=8)
    ap.add_argument('--intact', action='store_true', help='show the load before the collapse')
    args = ap.parse_args()
    from shinai_pi.sim.runtime import IsaacRuntime, run_transport
    from shinai_pi.viewpoints import park_pose
    cell = load_cell()
    rt = IsaacRuntime(cell, headless=False)
    profile = sample_transport(cell.transport, cell.layout.layers, args.seed)
    rt.build(vary_load(initial_cases(cell, cell.skus[0]), cell.variation, args.seed),
             wrapped_layers=[l for l in range(cell.layout.layers) if l not in profile.loose_layers])
    rt.start()
    rt.set_gantry_target(park_pose(cell))
    rt.step(int(1.0 / cell.physics.dt))
    if not args.intact:
        run_transport(rt, profile)
    try:
        from omni.kit.viewport.utility import get_active_viewport
        from shinai_pi.sim import cell as cellmod
        get_active_viewport().camera_path = cellmod.CAMERAS['presentation']
    except Exception as e:                                   # the free camera still works
        print('viewport camera not set:', e, flush=True)
    rt.timeline.pause()                                      # still picture: look around without the physics running on
    print('READY: the Isaac Sim window shows the cell. Close it to end.', flush=True)
    while rt.app.is_running():
        rt.app.update()
    rt.close()


if __name__ == '__main__':
    main()
