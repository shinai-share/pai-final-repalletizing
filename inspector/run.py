import argparse
import json
import time
import hashlib
import math
import io,base64
from pathlib import Path
import numpy as np
from .brain import Brain
from .simulation import Plant
from .perception import Detector, train

DEFAULT='ポンプA、配電盤B、ファンCを順に点検して'

def run(scenario='combined',seed=42,instruction=DEFAULT,model=None,max_seconds=240,video=None,allow_replan=True,backend='3d',capture_preview=False):
    model=model or ('artifacts/detector_3d.json' if backend=='3d' else 'artifacts/detector.json')
    brain=Brain(instruction,Detector(model),allow_replan=allow_replan,inspection_heading=math.pi/2 if backend=='3d' else None)
    if backend=='3d':
        from .simulation3d import Plant3D
        plant=Plant3D(scenario,seed)
    else:plant=Plant(scenario,seed)
    latencies=[]; trace=[]; maps=[]; preview=[]; recorder=None; start=time.perf_counter()
    if video:
        from .video import Recorder
        recorder=Recorder(video)
    try:
        for tick in range(int(max_seconds/plant.dt)):
            obs=plant.observe()
            before=time.perf_counter()
            v,w=brain.control(obs,plant.camera)
            latencies.append((time.perf_counter()-before)*1000)
            plant.step(v,w)
            if tick%20==0:
                maps.append({'t':round(float(plant.data.time),3),'occupied':[plant_cell for plant_cell in sorted(brain.world.dynamic)],'route':list(brain.route)})
                if capture_preview:
                    from PIL import Image
                    frame={'t':round(float(plant.data.time),3)}
                    for name,array in [('scene',plant.render()),('camera',plant.onboard_view())]:
                        buffer=io.BytesIO();Image.fromarray(array).save(buffer,format='JPEG',quality=70)
                        frame[name]='data:image/jpeg;base64,'+base64.b64encode(buffer.getvalue()).decode('ascii')
                    preview.append(frame)
            if tick%2==0:
                trace.append({'t':round(float(plant.data.time),3),'pose':plant.pose.tolist(),'state':brain.state,'reason':brain.reason,'target':brain.target,'v':v,'w':w,'age':obs['now']-obs['stamp']})
                if recorder: recorder.add(plant,brain,trace[-1])
            if brain.state in ('COMPLETE','ABORTED'):
                for _ in range(40): plant.step(0,0)
                break
        if brain.state not in ('COMPLETE','ABORTED'):
            brain.state='ABORTED';brain.reason='DEADLINE';brain.event(float(plant.data.time),'SAFE_ABORT',reason='DEADLINE')
            for _ in range(40):plant.step(0,0)
        elapsed=time.perf_counter()-start
        from PIL import Image
        evidence_dir=Path(model).parent/'evidence'/f'{scenario}-{seed}'
        evidence_dir.mkdir(parents=True,exist_ok=True)
        evidence=[]
        for i,(target,t,image) in enumerate(brain.evidence):
            file=evidence_dir/f'frame-{i:02}.png'
            Image.fromarray(image).save(file)
            evidence.append({'target':target,'time':round(t,3),'file':file.relative_to(Path(model).parent).as_posix(),'sha256':hashlib.sha256(file.read_bytes()).hexdigest()})
        report={'scenario':scenario,'seed':seed,'instruction':instruction,'state':brain.state,'completed':brain.completed,'sim_seconds':float(plant.data.time),'wall_seconds':elapsed,'collisions':plant.collisions,'final_speed':float(np.linalg.norm(plant.data.qvel[:2])),'replans':brain.replans,'control_latency_ms_p95':float(np.percentile(latencies,95)),'findings':brain.findings,'events':brain.events,'trace':trace,'model':'MuJoCo reduced planar AMR; proxy RGB indicator camera','video_playback_speed':2 if video else None}
        report['evidence']=evidence
        report['maps']=maps
        if preview:report['preview']=preview
        report['backend']=backend
        if backend=='3d':
            report['model']='MuJoCo full 3D free chassis / wheel velocity actuators / contact dynamics / onboard rendered RGB / simulated global localization'
            report['max_tilt_degrees']=math.degrees(plant.max_tilt)
            report['distance_m']=plant.distance
        if recorder: recorder.finish(report)
        return report
    finally:
        plant.close()
        if recorder: recorder.close()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--train',action='store_true');p.add_argument('--scenario',default='combined',choices=['normal','blocked','sensor_dropout','combined','no_route'])
    p.add_argument('--seed',type=int,default=42);p.add_argument('--instruction',default=DEFAULT)
    p.add_argument('--video');p.add_argument('--out',default='artifacts');a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    if a.train: print(json.dumps(train(out),indent=2));return
    r=run(a.scenario,a.seed,a.instruction,str(out/'detector_3d.json'),video=a.video)
    (out/f'{a.scenario}-{a.seed}.json').write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in r.items() if k not in ('trace','events','maps')},ensure_ascii=False,indent=2))

if __name__=='__main__': main()

