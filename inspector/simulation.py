"""MuJoCo planar rigid-body AMR plant with collisions and ray-range observations.

The chassis is a reduced 3-DOF base with velocity servos, not a wheel-slip model.
The mission brain sees only timestamped observations, never the gate schedule.
"""
import math
import numpy as np
import mujoco
from .planning import WALLS, TARGETS
from .perception import panel

class Plant:
    dt=.05

    def __init__(self, scenario='blocked', seed=42):
        self.scenario=scenario
        self.rng=np.random.default_rng(seed)
        walls=''.join(f'<geom name="wall{i}" type="box" pos="{x} {y} .6" size="{sx} {sy} .6" rgba=".25 .32 .42 1"/>' for i,(x,y,sx,sy) in enumerate(WALLS))
        stations=''.join(f'<geom name="station{i}" type="cylinder" pos="{x} {y+1.15} .4" size=".28 .4" rgba="{c}"/>' for i,((x,y),c) in enumerate(zip(TARGETS.values(),['.92 .38 .16 1','.18 .65 .65 1','.25 .45 .85 1'])))
        self.xml=f'''<mujoco model="inspection lab">
        <option timestep=".005" gravity="0 0 -9.81" integrator="implicitfast"/>
        <visual><global offwidth="960" offheight="720"/><quality shadowsize="2048"/><headlight ambient=".4 .4 .4"/></visual>
        <asset><texture name="grid" type="2d" builtin="checker" rgb1=".12 .17 .23" rgb2=".15 .20 .27" width="256" height="256"/><material name="floor" texture="grid" texrepeat="12 8" reflectance=".08"/></asset>
        <worldbody><light pos="0 -1 9" diffuse=".8 .8 .8"/><geom name="floor" type="plane" size="6 4 .1" material="floor"/>
        <geom type="box" pos="-6 0 .4" size=".1 4 .4" rgba=".2 .3 .4 1"/><geom type="box" pos="6 0 .4" size=".1 4 .4" rgba=".2 .3 .4 1"/>
        <geom type="box" pos="0 -4 .4" size="6 .1 .4" rgba=".2 .3 .4 1"/><geom type="box" pos="0 4 .4" size="6 .1 .4" rgba=".2 .3 .4 1"/>
        {walls}{stations}
        <body name="gate" mocap="true" pos="0 -2 -3"><geom name="gate_geom" type="box" size=".18 .75 .5" rgba="1 .48 .08 1"/></body>
        <body name="robot" pos="0 0 .3"><joint name="x" type="slide" axis="1 0 0" damping="3"/><joint name="y" type="slide" axis="0 1 0" damping="3"/><joint name="yaw" type="hinge" axis="0 0 1" damping="2"/>
        <geom name="chassis" type="cylinder" size=".26 .16" mass="12" rgba=".14 .8 .8 1"/>
        <geom type="box" pos=".18 0 .2" size=".11 .09 .08" mass=".2" rgba=".85 .96 1 1" contype="0" conaffinity="0"/>
        <geom type="cylinder" pos="0 0 .25" size=".08 .035" mass=".1" rgba=".05 .08 .12 1" contype="0" conaffinity="0"/>
        </body></worldbody>
        <actuator><velocity joint="x" kv="180" ctrllimited="true" ctrlrange="-.7 .7" forcelimited="true" forcerange="-45 45"/><velocity joint="y" kv="180" ctrllimited="true" ctrlrange="-.7 .7" forcelimited="true" forcerange="-45 45"/><velocity joint="yaw" kv="25" ctrllimited="true" ctrlrange="-1.5 1.5" forcelimited="true" forcerange="-15 15"/></actuator>
        </mujoco>'''
        self.model=mujoco.MjModel.from_xml_string(self.xml)
        self.model.body_mass[self.model.body('robot').id]*=self.rng.uniform(.8,1.2)
        self.model.dof_damping[:]*=self.rng.uniform(.8,1.2)
        self.gate_time=float(self.rng.uniform(1.5,3.0))
        self.data=mujoco.MjData(self.model)
        self.data.qpos[:]=[-4,-2,0]
        self.robot_id=self.model.body('robot').id
        self.collisions=0
        self.gate_closed=False
        self.last_scan=None
        self.renderer=None
        mujoco.mj_forward(self.model,self.data)

    @property
    def pose(self):
        return self.data.qpos[:3].copy()

    def observe(self):
        now=float(self.data.time)
        # Fault injection is confined to sensor transport, not brain state.
        if self.scenario in ('sensor_dropout','combined') and 12<now<13.5 and self.last_scan is not None:
            ranges,points,stamp=self.last_scan
        else:
            x,y,yaw=self.pose
            ranges=[]; points=[]
            for a in np.linspace(-math.pi,math.pi,72,endpoint=False)+yaw:
                direction=np.array([math.cos(a),math.sin(a),0.])
                gid=np.array([-1],dtype=np.int32)
                distance=mujoco.mj_ray(self.model,self.data,np.array([x,y,.35]),direction,None,1,self.robot_id,gid)
                if distance<0 or distance>3.0:
                    ranges.append(3.0)
                else:
                    ranges.append(distance)
                    points.append((x+distance*direction[0],y+distance*direction[1]))
            ranges=np.array(ranges)
            stamp=now
            self.last_scan=(ranges,points,stamp)
        return {'pose':self.pose,'ranges':ranges.copy(),'points':list(points),'stamp':stamp,'now':now,'speed':float(np.linalg.norm(self.data.qvel[:2]))}

    def camera(self, target):
        # Proxy RGB indicator: rendered only at inspection range; truth withheld.
        if math.dist(self.pose[:2],TARGETS[target])>.35:
            return None
        return panel(target=='ポンプA',self.rng,shift=True)

    def step(self,v,w):
        t=float(self.data.time)
        if self.scenario in ('blocked','combined','no_route') and t>=self.gate_time and not self.gate_closed:
            self.data.mocap_pos[0]=[0,-2,.5]
            self.gate_closed=True
            if self.scenario=='no_route':
                self.model.geom_size[self.model.geom('gate_geom').id]=[.18,3.7,.5]
                self.data.mocap_pos[0]=[0,0,.5]
            mujoco.mj_forward(self.model,self.data)
        yaw=self.pose[2]
        self.data.ctrl[:]=[v*math.cos(yaw),v*math.sin(yaw),w]
        for _ in range(10):
            mujoco.mj_step(self.model,self.data)
            for c in self.data.contact:
                bodies=(self.model.geom_bodyid[c.geom1],self.model.geom_bodyid[c.geom2])
                if self.robot_id in bodies and c.dist<-.001:
                    self.collisions+=1

    def render(self):
        if self.renderer is None:
            self.renderer=mujoco.Renderer(self.model,height=720,width=960)
        cam=mujoco.MjvCamera(); cam.lookat[:]=[0,0,0]; cam.distance=13; cam.azimuth=90; cam.elevation=-65
        self.renderer.update_scene(self.data,camera=cam)
        return self.renderer.render().copy()

    def close(self):
        if self.renderer:
            self.renderer.close()

