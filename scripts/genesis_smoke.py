"""GPU physics smoke test: falling box, measured settling and throughput."""
import json,time
from pathlib import Path
import torch
import genesis as gs
gs.init(backend=gs.cuda,logging_level='warning')
scene=gs.Scene(show_viewer=False,sim_options=gs.options.SimOptions(dt=.01))
scene.add_entity(gs.morphs.Plane())
box=scene.add_entity(gs.morphs.Box(pos=(0,0,1),size=(.2,.2,.2)))
t=time.perf_counter();scene.build();build=time.perf_counter()-t
for _ in range(100):scene.step()
torch.cuda.synchronize();t=time.perf_counter()
for _ in range(200):scene.step()
torch.cuda.synchronize();elapsed=time.perf_counter()-t
pos=box.get_pos().cpu().tolist()
result={'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'cuda_runtime':torch.version.cuda,'build_seconds':build,'steps_per_second':200/elapsed,'final_box_position':pos,'settled':abs(pos[2]-.1)<.03}
Path('artifacts/genesis-smoke.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
