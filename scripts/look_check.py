"""Evidence for how the cell looks to people: stills from every people view after a collapse, the
colour difference between cases, pallet and floor measured in a rendered picture, and render time.

    .venv-isaac/Scripts/python scripts/look_check.py [--seed 8]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shinai_pi.appearance import lab_from_srgb8  # noqa: E402
from shinai_pi.config import load_cell  # noqa: E402
from shinai_pi.layout import initial_cases  # noqa: E402
from shinai_pi.transport import sample_transport  # noqa: E402
from shinai_pi.variation import vary_load  # noqa: E402

PEOPLE = ('presentation', 'side', 'closeup')


def mean_lab(frame, part):
    """Mean CIELAB colour of the pixels whose object path contains `part`."""
    ids = [i for i, label in frame['instance_map'].items() if part in label]
    mask = np.isin(frame['instance'], ids)
    if mask.sum() < 50:
        return None, int(mask.sum())
    return lab_from_srgb8(frame['rgb'][mask].reshape(-1, 3).mean(0)), int(mask.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=8)
    ap.add_argument('--out', default=str(ROOT / 'artifacts' / 'look'))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    from PIL import Image
    from shinai_pi.sim.runtime import IsaacRuntime, run_transport
    from shinai_pi.viewpoints import park_pose
    cell = load_cell()
    rt = IsaacRuntime(cell)
    profile = sample_transport(cell.transport, cell.layout.layers, args.seed)
    rt.build(vary_load(initial_cases(cell, cell.skus[0]), cell.variation, args.seed),
             wrapped_layers=[l for l in range(cell.layout.layers) if l not in profile.loose_layers])
    rt.start()
    rt.set_gantry_target(park_pose(cell))
    rt.step(int(1.0 / cell.physics.dt))
    run_transport(rt, profile)
    report = {'seed': args.seed, 'rendering': {k: list(v) if isinstance(v, tuple) else v
                                               for k, v in vars(cell.rendering).items()},
              'render_ms': {}, 'size': {}}
    for v in PEOPLE:
        rt.render_rgb(v)                                   # the first render builds the camera
        t = time.perf_counter()
        for _ in range(5):
            img = rt.render_rgb(v)
        report['render_ms'][v] = round((time.perf_counter() - t) / 5 * 1000, 1)
        report['size'][v] = [int(img.shape[1]), int(img.shape[0])]
        Image.fromarray(img).save(out / f'{v}.png')
    # The cases cover pallet A's deck from above; the empty pallet B shows its deck under the same light.
    frame_a, frame_b = rt.capture('overhead_a'), rt.capture('overhead_b')
    Image.fromarray(frame_a['rgb']).save(out / 'overhead_a.png')
    case_lab, n_case = mean_lab(frame_a, '/cases/')
    pallet_lab, n_pallet = mean_lab(frame_b, '/pallet_b')
    floor_lab, n_floor = mean_lab(frame_a, '/floor')

    def de(a, b):
        return None if a is None or b is None else round(float(np.linalg.norm(a - b)), 1)
    report['measured_in_picture'] = {'case_vs_pallet_delta_e': de(case_lab, pallet_lab),
                                     'case_vs_floor_delta_e': de(case_lab, floor_lab),
                                     'pallet_vs_floor_delta_e': de(pallet_lab, floor_lab),
                                     'pixels': {'case': n_case, 'pallet': n_pallet, 'floor': n_floor}}
    (out / 'look.json').write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding='utf-8')
    print('LOOK', json.dumps(report, ensure_ascii=False), flush=True)
    rt.close()


if __name__ == '__main__':
    main()
