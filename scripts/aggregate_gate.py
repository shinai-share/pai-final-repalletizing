"""Gate verdict over the result.json files of a stage folder (trials may come from several processes).

    python scripts/aggregate_gate.py <stage folder> <stage>"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage_trials import gate  # noqa: E402
from shinai_pi.evaluation.resume import gate_rows  # noqa: E402

base = Path(sys.argv[1])
rows = gate_rows(base)
g = gate(sys.argv[2], rows)
(base / 'summary.json').write_text(json.dumps({'stage': sys.argv[2], 'gate': g, 'trials': rows}, indent=1, ensure_ascii=False),
                                   encoding='utf-8')
print(sys.argv[2], json.dumps(g))
