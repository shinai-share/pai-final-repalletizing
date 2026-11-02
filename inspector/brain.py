"""Evidence-driven mission state machine, bounded recovery, independently gated control."""
import math
from .planning import TARGETS, WorldModel, parse_instruction, supervise

class Brain:
    def __init__(self, instruction, detector, allow_replan=True, inspection_heading=None):
        self.allow_replan=allow_replan
        self.inspection_heading=inspection_heading
        self.targets=parse_instruction(instruction)
        self.detector=detector
        self.world=WorldModel()
        self.index=0; self.route=[]; self.events=[]; self.findings=[]
        self.state='NAVIGATE'; self.reason='CLEAR'
        self.replans=0; self.next_plan=0.; self.hold_since=None
        self.samples=[]; self.last_sample=-1.
        self.completed=False; self.emergency=False
        self.last_safety='CLEAR'
        self.last_image=None
        self.evidence=[]

    def event(self,t,kind,**data):
        self.events.append({'time':round(t,3),'kind':kind,**data})

    @property
    def target(self):
        return self.targets[min(self.index,len(self.targets)-1)]

    def control(self,obs,camera):
        now=obs['now']; pose=obs['pose']; age=now-obs['stamp']
        if self.state in ('COMPLETE','ABORTED'):
            return 0.,0.
        if age<=.25:
            self.world.observe(obs['points'])
        if self.emergency:
            self.state='ABORTED'; self.reason='EMERGENCY_STOP'; self.event(now,'EMERGENCY_STOP')
            return 0.,0.
        if math.dist(pose[:2],TARGETS[self.target])<.2 and age<=.25:
            self.state='INSPECT'
            if self.inspection_heading is not None:
                angle=(self.inspection_heading-pose[2]+math.pi)%(2*math.pi)-math.pi
                if abs(angle)>.07 or obs['speed']>.03:
                    v,w,reason=supervise(0.,max(-.7,min(.7,2*angle)),obs['ranges'],age,obs['speed'])
                    self.reason=reason
                    return v,w
            if now-self.last_sample>=.35:
                im=camera(self.target)
                if im is not None:
                    self.last_image=im
                    self.evidence.append((self.target,now,im.copy()))
                    self.samples.append(self.detector.predict(im)); self.last_sample=now
            if len(self.samples)>=3:
                low,high=min(self.samples),max(self.samples)
                result='ANOMALY' if low>=.8 else ('NORMAL' if high<=.2 else 'REVIEW')
                record={'target':self.target,'result':result,'probabilities':[round(p,5) for p in self.samples],'time':round(now,2),'evidence':'three independent proxy RGB observations'}
                self.findings.append(record); self.event(now,'INSPECTION',**record)
                self.samples=[]; self.index+=1; self.route=[]; self.next_plan=0
                if self.index==len(self.targets):
                    self.completed=True; self.state='COMPLETE'; self.event(now,'MISSION_COMPLETE')
                else:
                    self.state='NAVIGATE'
            return 0.,0.
        if now>=self.next_plan and age<=.25:
            blocked=self.world.static|self.world.dynamic
            invalid=any(self.world.cell(p) in blocked for p in self.route)
            if not self.route or (invalid and self.allow_replan):
                route=self.world.plan(pose[:2],TARGETS[self.target])
                if self.route and invalid:
                    self.replans+=1; self.event(now,'REPLAN',revision=self.world.revision,target=self.target)
                self.route=route
                if not route:
                    self.state='RECOVERY'; self.reason='NO_ROUTE'
                    if self.hold_since is None: self.hold_since=now
                    if now-self.hold_since>6:
                        self.state='ABORTED'; self.event(now,'SAFE_ABORT',reason='NO_ROUTE')
                    self.next_plan=now+.5
                    return 0.,0.
            self.next_plan=now+.5
        if not self.route:
            self.state='RECOVERY'; self.reason='NO_ROUTE'
            if self.hold_since is None: self.hold_since=now
            if now-self.hold_since>6:
                self.state='ABORTED'; self.event(now,'SAFE_ABORT',reason='NO_ROUTE')
            return 0.,0.
        if self.route:
            while len(self.route)>1 and math.dist(pose[:2],self.route[0])<.18:
                self.route.pop(0)
            dx,dy=self.route[0][0]-pose[0],self.route[0][1]-pose[1]
            err=(math.atan2(dy,dx)-pose[2]+math.pi)%(2*math.pi)-math.pi
            v=min(.65,1.8*math.hypot(dx,dy))*max(0,math.cos(err)) if abs(err)<.7 else 0.
            w=max(-1.5,min(1.5,3*err))
        else:
            v=w=0.
        v,w,safety=supervise(v,w,obs['ranges'],age,obs['speed'])
        self.reason=safety
        if safety!=self.last_safety:
            self.event(now,safety)
            self.last_safety=safety
        if safety!='CLEAR':
            self.state='RECOVERY'
            if self.hold_since is None: self.hold_since=now
            if now-self.hold_since>6:
                self.state='ABORTED'; self.event(now,'SAFE_ABORT',reason=safety)
                return 0.,0.
        else:
            self.hold_since=None
            self.state='NAVIGATE'
        return v,w
