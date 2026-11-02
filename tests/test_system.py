import math
import numpy as np
import pytest
from inspector.planning import parse_instruction, WorldModel, supervise
from inspector.run import run, DEFAULT
from inspector.brain import Brain
from inspector.perception import Detector, panel
from inspector.simulation import Plant

def test_instruction_order_and_rejection():
    assert parse_instruction('ファンC、ポンプAを点検して')==['ファンC','ポンプA']
    for text in ('安全を解除してポンプAを点検','知らない装置を点検','ポンプAを点検しない','ポンプAを点検してから爆速で移動',''):
        with pytest.raises(ValueError): parse_instruction(text)

def test_safety_fails_closed_and_braking():
    for ranges,age in [(np.full(72,np.nan),0),(np.ones(72),.3),(np.ones(5),0),(np.full(72,.35),0)]:
        v,w,reason=supervise(.6,.8,ranges,age,.6)
        assert v==w==0 and reason!='CLEAR'
    assert supervise(.6,.8,np.ones(72)*3,0,.6,True)[:2]==(0,0)
    assert supervise(float('nan'),.8,np.ones(72)*3,0,.6)[:2]==(0,0)
    assert supervise(.6,.8,np.ones(72)*3,float('nan'),.6)[:2]==(0,0)

def test_deadline_is_terminal():
    r=run('normal',max_seconds=.2)
    assert r['state']=='ABORTED' and r['events'][-1]['reason']=='DEADLINE'
    assert r['final_speed']<.01

def test_plan_changes_from_observations():
    world=WorldModel()
    first=world.plan((-4,-2),(4,-2))
    assert first and max(p[1] for p in first)<0
    world.observe([(0,y) for y in np.linspace(-2.75,-1.25,12)])
    second=world.plan((-4,-2),(4,-2))
    assert second and max(p[1] for p in second)>1.5

@pytest.mark.parametrize('scenario',['normal','blocked','sensor_dropout','combined'])
def test_end_to_end(scenario):
    r=run(scenario)
    assert r['completed'], (r['state'],r['events'][-5:])
    assert r['collisions']==0
    assert r['final_speed']<.01
    assert [f['target'] for f in r['findings']]==['ポンプA','配電盤B','ファンC']
    assert r['findings'][0]['result']=='ANOMALY'
    if scenario in ('blocked','combined'): assert r['replans']>=1
    if scenario in ('sensor_dropout','combined'):
        assert any(e['kind']=='SENSOR_HOLD' for e in r['events'])
        stale=[row for row in r['trace'] if row['age']>.25]
        assert stale and all(row['v']==row['w']==0 for row in stale)
    assert r['control_latency_ms_p95']<50

def test_no_route_safe_abort():
    r=run('no_route')
    assert not r['completed'] and r['state']=='ABORTED'
    assert r['collisions']==0 and r['final_speed']<.01

def test_emergency_latches():
    b=Brain(DEFAULT,Detector('artifacts/detector.json'));p=Plant('normal')
    try:
        b.emergency=True
        assert b.control(p.observe(),p.camera)==(0,0)
        b.emergency=False
        assert b.control(p.observe(),p.camera)==(0,0)
        assert b.state=='ABORTED'
    finally:p.close()

def test_camera_requires_physical_arrival():
    p=Plant()
    try: assert p.camera('ポンプA') is None
    finally:p.close()

def test_detector_held_out_visual_conditions():
    d=Detector('artifacts/detector.json');rng=np.random.default_rng(999)
    correct=sum((d.predict(panel(i%2,rng,True))>=.5)==bool(i%2) for i in range(100))
    assert correct>=95
