"""Held-out scenario evaluation and replanning ablation; writes observed results."""
import json
from pathlib import Path
import numpy as np
from inspector.run import run

rows=[]
for scenario in ['normal','blocked','sensor_dropout','combined','no_route']:
    for seed in range(50,60):
        r=run(scenario,seed)
        rows.append({k:r[k] for k in ('scenario','seed','completed','state','collisions','replans','sim_seconds','control_latency_ms_p95','final_speed')})
ablation=[]
for seed in range(50,60):
    r=run('blocked',seed,allow_replan=False)
    ablation.append({k:r[k] for k in ('seed','completed','state','collisions','sim_seconds')})
summary={}
for scenario in ['normal','blocked','sensor_dropout','combined','no_route']:
    group=[r for r in rows if r['scenario']==scenario]
    summary[scenario]={'runs':len(group),'completed':sum(r['completed'] for r in group),'safe_aborts':sum(r['state']=='ABORTED' and r['collisions']==0 for r in group),'collisions':sum(r['collisions'] for r in group),'max_control_p95_ms':max(r['control_latency_ms_p95'] for r in group),'mean_sim_seconds':float(np.mean([r['sim_seconds'] for r in group]))}
result={'summary':summary,'runs':rows,'ablation_without_replanning':ablation,'variations':'mass/damping +/-20%, gate onset 1.5-3.0s, camera lighting/noise/occlusion; fixed topology','scope':'No real-world generalization claim. Safe abort is expected for no_route.'}
Path('artifacts/evaluation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2)); print('ablation completed',sum(r['completed'] for r in ablation))
