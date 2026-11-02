"""Full 3D differential-drive AMR with wheel torques, contacts and onboard RGB.

Global pose is currently a simulated localization observation (not implemented SLAM).
No body pose/velocity is set during mission execution: only wheel actuators move it.
"""
import math
import numpy as np
import mujoco
from .planning import TARGETS,WALLS

class Plant3D:
    dt=.05
    wheel_radius=.12
    track=.52
    def __init__(self,scenario='blocked',seed=42):
        self.scenario=scenario;self.rng=np.random.default_rng(seed)
        self.gate_time=float(self.rng.uniform(1.5,3))
        walls=''.join(f'<geom name="wall{i}" type="box" pos="{x} {y} .65" size="{sx} {sy} .65" material="wall"/>' for i,(x,y,sx,sy) in enumerate(WALLS))
        stations=''
        for i,(name,(x,y)) in enumerate(TARGETS.items()):
            sy=y+1.15
            stations+=f'''<body name="equipment{i}" pos="{x} {sy} 0">
            <geom name="equipment_geom{i}" type="box" pos="0 0 .55" size=".37 .28 .55" material="metal"/>
            <geom type="box" pos="0 -.29 .82" size=".28 .025 .2" rgba=".06 .08 .10 1"/>
            <geom name="indicator{i}" type="sphere" pos="0 -.335 .87" size=".09" material="{'red' if i==0 else 'green'}" contype="0" conaffinity="0"/>
            <geom type="box" pos="0 -.32 .65" size=".21 .01 .025" rgba=".7 .8 .85 1" contype="0" conaffinity="0"/>
            <geom type="cylinder" pos=".46 0 .5" size=".09 .5" material="metal"/>
            <geom type="box" pos="0 0 1.15" size=".44 .35 .05" material="metal"/>
            </body>'''
        self.xml=f'''<mujoco model="3D Inspection Facility">
        <compiler angle="radian"/>
        <option timestep=".0025" integrator="implicitfast" gravity="0 0 -9.81" cone="elliptic" iterations="60"/>
        <default><geom friction=".9 .005 .0001" condim="4" solref=".008 1"/><joint damping=".02" armature=".002"/></default>
        <visual><global offwidth="960" offheight="720"/><quality shadowsize="2048"/><headlight ambient=".25 .25 .25" diffuse=".4 .4 .4"/></visual>
        <asset><texture name="grid" type="2d" builtin="checker" rgb1=".18 .21 .24" rgb2=".22 .25 .28" width="256" height="256"/>
        <material name="floor" texture="grid" texrepeat="12 8" reflectance=".12"/>
        <material name="wall" rgba=".33 .4 .47 1"/><material name="metal" rgba=".32 .38 .42 1" specular=".5" shininess=".6"/>
        <material name="red" rgba="1 .015 .005 1" emission=".5"/><material name="green" rgba=".005 .9 .08 1" emission=".5"/></asset>
        <worldbody><light name="main" pos="-3 -1 6" diffuse=".8 .8 .8"/><light pos="4 2 5" diffuse=".6 .6 .65"/>
        <geom name="floor" type="plane" size="6 4 .1" material="floor"/>
        <geom name="bound0" type="box" pos="-6 0 .35" size=".1 4 .35" material="wall"/>
        <geom name="bound1" type="box" pos="6 0 .35" size=".1 4 .35" material="wall"/>
        <geom name="bound2" type="box" pos="0 -4 .35" size="6 .1 .35" material="wall"/>
        <geom name="bound3" type="box" pos="0 4 .35" size="6 .1 .35" material="wall"/>
        {walls}{stations}
        <body name="gate" mocap="true" pos="0 -2 -3"><geom name="gate_geom" type="box" size=".18 .75 .5" rgba="1 .48 .08 1"/>
        <geom type="box" pos="0 -.02 .25" size=".19 .77 .07" rgba=".08 .08 .08 1"/></body>
        <body name="robot" pos="-4 -2 .225"><freejoint name="base"/>
        <geom name="chassis" type="box" size=".25 .20 .10" mass="11" rgba=".06 .5 .56 1"/>
        <geom name="cover" type="box" pos="0 0 .14" size=".21 .18 .04" mass=".4" rgba=".15 .77 .78 1"/>
        <geom name="mast" type="cylinder" pos="0 0 .4" size=".025 .3" mass=".15" material="metal"/>
        <geom name="camera_housing" type="box" pos=".03 0 .64" size=".07 .08 .04" mass=".1" rgba=".08 .11 .13 1"/>
        <camera name="onboard" pos=".105 0 .64" xyaxes="0 -1 0 0 0 1" fovy="52"/>
        <geom name="lidar" type="cylinder" pos="0 0 .26" size=".07 .03" mass=".1" rgba=".05 .08 .1 1"/>
        <body name="left_wheel" pos="0 .26 -.105"><joint name="left" type="hinge" axis="0 1 0"/>
        <geom name="wheel_left" type="cylinder" quat=".70710678 .70710678 0 0" size=".12 .045" mass=".6" rgba=".035 .045 .05 1" friction="1.1 .005 .0001"/></body>
        <body name="right_wheel" pos="0 -.26 -.105"><joint name="right" type="hinge" axis="0 1 0"/>
        <geom name="wheel_right" type="cylinder" quat=".70710678 .70710678 0 0" size=".12 .045" mass=".6" rgba=".035 .045 .05 1" friction="1.1 .005 .0001"/></body>
        <geom name="caster_front" type="sphere" pos=".19 0 -.165" size=".06" mass=".1" priority="1" friction=".002 .0001 .00001" rgba=".04 .04 .04 1"/>
        <geom name="caster_back" type="sphere" pos="-.19 0 -.165" size=".06" mass=".1" priority="1" friction=".002 .0001 .00001" rgba=".04 .04 .04 1"/>
        </body></worldbody>
        <actuator><velocity name="left_motor" joint="left" kv="2" ctrllimited="true" ctrlrange="-12 12" forcelimited="true" forcerange="-4 4"/>
        <velocity name="right_motor" joint="right" kv="2" ctrllimited="true" ctrlrange="-12 12" forcelimited="true" forcerange="-4 4"/></actuator>
        </mujoco>'''
        self.model=mujoco.MjModel.from_xml_string(self.xml)
        self.data=mujoco.MjData(self.model)
        self.robot_id=self.model.body('robot').id
        self.robot_bodies={self.robot_id,self.model.body('left_wheel').id,self.model.body('right_wheel').id}
        # group 1 excludes robot geoms from range rays, but not from physics.
        for i in range(self.model.ngeom):
            if self.model.geom_bodyid[i] in self.robot_bodies:self.model.geom_group[i]=1
        self.model.body_mass[self.robot_id]*=self.rng.uniform(.9,1.1)
        for name in ['wheel_left','wheel_right']:
            self.model.geom_friction[self.model.geom(name).id,0]*=self.rng.uniform(.9,1.1)
        self.collisions=0;self.gate_closed=False;self.last_scan=None;self.renderer=None;self.cam_renderer=None
        self.last_image=None;self.last_rgb_time=-1.;self.max_tilt=0.;self.distance=0.;self.last_position=np.array([-4.,-2.])
        self.yaw_integral=0.
        mujoco.mj_forward(self.model,self.data)
        for _ in range(200):mujoco.mj_step(self.model,self.data)
        self.data.time=0

    @property
    def pose(self):
        rot=self.data.xmat[self.robot_id].reshape(3,3)
        return np.array([*self.data.xpos[self.robot_id,:2],math.atan2(rot[1,0],rot[0,0])])

    def observe(self):
        now=float(self.data.time);x,y,yaw=self.pose
        if self.scenario in ('sensor_dropout','combined') and 12<now<13.5 and self.last_scan is not None:
            ranges,points,stamp=self.last_scan
        else:
            ranges=[];points=[]
            origin=self.data.xpos[self.robot_id].copy();origin[2]+=.26
            groups=np.array([1,0,1,1,1,1],dtype=np.uint8)
            for a in np.linspace(-math.pi,math.pi,72,endpoint=False)+yaw:
                direction=np.array([math.cos(a),math.sin(a),0.]);gid=np.array([-1],dtype=np.int32)
                distance=mujoco.mj_ray(self.model,self.data,origin,direction,groups,1,-1,gid)
                if distance<0 or distance>3:ranges.append(3.)
                else:
                    ranges.append(distance);points.append((x+distance*direction[0],y+distance*direction[1]))
            ranges=np.array(ranges);stamp=now;self.last_scan=(ranges,points,stamp)
        return {'pose':self.pose,'ranges':ranges.copy(),'points':list(points),'stamp':stamp,'now':now,'speed':float(np.linalg.norm(self.data.qvel[:2]))}

    def camera(self,target):
        if math.dist(self.pose[:2],TARGETS[target])>.35:return None
        return self.onboard_view()

    def onboard_view(self):
        if self.cam_renderer is None:self.cam_renderer=mujoco.Renderer(self.model,height=240,width=320)
        self.cam_renderer.update_scene(self.data,camera='onboard')
        self.last_image=self.cam_renderer.render().copy()
        self.last_rgb_time=float(self.data.time)
        return self.last_image

    def step(self,v,w):
        if self.scenario in ('blocked','combined','no_route') and self.data.time>=self.gate_time and not self.gate_closed:
            self.data.mocap_pos[0]=[0,-2,.5];self.gate_closed=True
            if self.scenario=='no_route':
                self.model.geom_size[self.model.geom('gate_geom').id]=[.18,3.7,.5];self.data.mocap_pos[0]=[0,0,.5]
            mujoco.mj_forward(self.model,self.data)
        # Low-level yaw-rate servo uses simulated IMU angular velocity, with
        # anti-windup and torque-limited wheel actuators. Chassis is never moved directly.
        measured_yaw_rate=float(self.data.qvel[5])
        if v==0 and w==0:
            self.yaw_integral=0.;drive_w=0.
        else:
            error=w-measured_yaw_rate
            self.yaw_integral=float(np.clip(self.yaw_integral+error*self.dt*1.5,-1.8,1.8))
            drive_w=float(np.clip(w+1.5*error+self.yaw_integral,-2.5,2.5))
        self.data.ctrl[:]=np.clip([(v-drive_w*self.track/2)/self.wheel_radius,(v+drive_w*self.track/2)/self.wheel_radius],-12,12)
        for _ in range(20):
            mujoco.mj_step(self.model,self.data)
            for contact in self.data.contact:
                g1,g2=contact.geom1,contact.geom2
                b1,b2=self.model.geom_bodyid[g1],self.model.geom_bodyid[g2]
                if (b1 in self.robot_bodies) != (b2 in self.robot_bodies):
                    other=g2 if b1 in self.robot_bodies else g1
                    if other!=self.model.geom('floor').id and contact.dist<0:self.collisions+=1
        rot=self.data.xmat[self.robot_id].reshape(3,3)
        self.max_tilt=max(self.max_tilt,math.acos(float(np.clip(rot[2,2],-1,1))))
        pos=self.pose[:2];self.distance+=float(np.linalg.norm(pos-self.last_position));self.last_position=pos

    def render(self):
        if self.renderer is None:self.renderer=mujoco.Renderer(self.model,height=720,width=960)
        cam=mujoco.MjvCamera();cam.lookat[:]=[0,0,.35];cam.distance=12.8;cam.azimuth=110;cam.elevation=-48
        self.renderer.update_scene(self.data,camera=cam)
        return self.renderer.render().copy()

    def close(self):
        for r in (self.renderer,self.cam_renderer):
            if r:r.close()

