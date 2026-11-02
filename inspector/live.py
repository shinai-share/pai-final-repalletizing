"""Interactive 3D physics viewer: mouse to orbit; E latches emergency stop."""
import time,math,argparse
import mujoco.viewer
from .simulation3d import Plant3D
from .brain import Brain
from .perception import Detector
from .run import DEFAULT

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--scenario',default='combined');args=parser.parse_args()
    p=Plant3D(args.scenario);b=Brain(DEFAULT,Detector('artifacts/detector_3d.json'),inspection_heading=math.pi/2)
    def key(k):
        if k in (69,101):b.emergency=True
    try:
        with mujoco.viewer.launch_passive(p.model,p.data,key_callback=key) as viewer:
            viewer.cam.lookat[:]=[0,0,.35];viewer.cam.distance=13;viewer.cam.azimuth=110;viewer.cam.elevation=-48
            while viewer.is_running():
                start=time.perf_counter()
                with viewer.lock():
                    v,w=b.control(p.observe(),p.camera);p.step(v,w)
                viewer.sync()
                time.sleep(max(0,p.dt-(time.perf_counter()-start)))
    finally:p.close()

if __name__=='__main__':main()
