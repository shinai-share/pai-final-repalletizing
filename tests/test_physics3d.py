import numpy as np
from inspector.simulation3d import Plant3D

def test_wheels_drive_and_brake_real_body():
    p=Plant3D('normal',42)
    try:
        start=p.pose.copy()
        for _ in range(40):p.step(.5,0)
        assert .8<p.pose[0]-start[0]<1.1
        assert abs(p.pose[1]-start[1])<.03
        assert .4<np.linalg.norm(p.data.qvel[:2])<.6
        before=p.pose.copy()
        for _ in range(20):p.step(0,0)
        assert np.linalg.norm(p.pose[:2]-before[:2])<.15
        assert np.linalg.norm(p.data.qvel[:2])<.01
        assert p.max_tilt<.15
    finally:p.close()

def test_differential_wheels_rotate_without_translation():
    p=Plant3D('normal',42)
    try:
        start=p.pose.copy()
        for _ in range(40):p.step(0,.6)
        assert .9<p.pose[2]-start[2]<1.4
        assert np.linalg.norm(p.pose[:2]-start[:2])<.08
    finally:p.close()

def test_physical_barrier_prevents_pass_through_without_brain():
    p=Plant3D('blocked',42)
    try:
        for _ in range(200):p.step(.6,0)
        assert p.pose[0]<-.3
        assert p.collisions>0
    finally:p.close()
