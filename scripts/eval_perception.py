"""Perception accuracy on recorded Isaac frames: overhead only vs overhead + side views.

Ground truth is read here for scoring only. Writes artifacts/perception_eval.json."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shinai_pi.config import load_cell  # noqa: E402
from shinai_pi.evaluation.metrics import box_axis_error_deg, box_position_error  # noqa: E402
from shinai_pi.perception.boxfit import fit_all  # noqa: E402
from shinai_pi.perception.geometry import frame_from_npz  # noqa: E402
from shinai_pi.perception.obb import quat_to_matrix  # noqa: E402
from shinai_pi.perception.voxels import VoxelMap  # noqa: E402
from shinai_pi.belief_space import pile_region  # noqa: E402

cell = load_cell()
dims = cell.skus[0].size
root = Path(__file__).resolve().parents[1] / 'artifacts'


def score(frames, truth, region):
    vm = VoxelMap(*region, 0.04)
    for f in frames:
        vm.integrate(f, step=3)
    est = fit_all(frames, lambda k: dims if k.startswith('case_') else None, vm)
    rows = []
    for k, e in est.items():
        t = truth[k]
        rows.append({'case': k, 'pos_err': box_position_error(e.obb.center, t['pos']),
                     'axis_err': box_axis_error_deg(e.obb.rotation, e.obb.size, quat_to_matrix(t['quat_wxyz']), dims),
                     'conf': e.confidence})
    lo, hi = np.array(region[0]), np.array(region[1])
    pile_lo = np.array([cell.pallet.a_center[0] - cell.pallet.size[0] / 2, cell.pallet.a_center[1] - cell.pallet.size[1] / 2, cell.pallet.size[2]])
    pile_hi = np.array([cell.pallet.a_center[0] + cell.pallet.size[0] / 2, cell.pallet.a_center[1] + cell.pallet.size[1] / 2, cell.pallet.size[2] + cell.layout.layers * dims[2]])
    unk = vm.unknown_fraction(pile_lo, pile_hi)
    pe = np.array([r['pos_err'] for r in rows]) if rows else np.array([np.nan])
    conf = np.array([r['conf'] for r in rows]) if rows else np.array([np.nan])
    sure = pe[conf > 0.8] if rows else pe
    return {'detected': len(rows), 'cases_total': len(truth),
            'pos_err_median_m': round(float(np.median(pe)), 4), 'pos_err_p90_m': round(float(np.percentile(pe, 90)), 4),
            'axis_err_median_deg': round(float(np.median([r['axis_err'] for r in rows])), 2),
            'confident_count': int(np.sum(conf > 0.8)),
            'confident_pos_err_p90_m': round(float(np.percentile(sure, 90)), 4) if len(sure) else None,
            'pile_unknown_fraction': round(unk, 3)}


report = {}
for d in sorted((root / 'frames').glob('seed*')):
    truth = json.loads((d / 'truth.json').read_text())['cases']
    region = pile_region(cell)
    overhead = [frame_from_npz(d / 'overhead_a.npz', 'overhead_a')]
    allv = overhead + [frame_from_npz(p, p.stem) for p in sorted(d.glob('tool_*.npz'))]
    report[d.name] = {'overhead_only': score(overhead, truth, region), 'overhead_plus_4_sides': score(allv, truth, region)}
    print(d.name, json.dumps(report[d.name]))
(root / 'perception_eval.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
