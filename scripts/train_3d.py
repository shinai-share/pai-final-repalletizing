"""Generate labeled onboard-camera images from the actual 3D scene, then train."""
import json,math,time
from pathlib import Path
import numpy as np
import mujoco
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score,confusion_matrix
from inspector.simulation3d import Plant3D
from inspector.planning import TARGETS
from inspector.perception import features

out=Path('artifacts');out.mkdir(exist_ok=True)
plant=Plant3D('normal',100)
renderer=mujoco.Renderer(plant.model,height=240,width=320)
examples=[]
def dataset(seed,n,shift=False):
    rng=np.random.default_rng(seed);xs=[];ys=[]
    for i in range(n):
        idx=int(rng.integers(0,3));target=list(TARGETS)[idx];x,y=TARGETS[target]
        label=int(rng.integers(0,2));ys.append(label)
        variation=.16 if shift else .09
        yaw=math.pi/2+rng.uniform(-.12 if shift else -.07,.12 if shift else .07)
        plant.data.qpos[:7]=[x+rng.uniform(-variation,variation),y+rng.uniform(-variation,variation),.225,math.cos(yaw/2),0,0,math.sin(yaw/2)]
        plant.model.light_diffuse[:]=rng.uniform(.3 if shift else .5,1.2)
        for j in range(3):
            gid=plant.model.geom(f'indicator{j}').id;plant.model.geom_matid[gid]=-1
            color=np.array([1,.02,.01,1]) if (label if j==idx else int(rng.integers(2))) else np.array([.01,.9,.05,1])
            plant.model.geom_rgba[gid]=color
        mujoco.mj_forward(plant.model,plant.data)
        renderer.update_scene(plant.data,camera='onboard')
        im=renderer.render().copy()
        im=np.clip(im.astype(float)+rng.normal(0,8 if shift else 3,im.shape),0,255).astype(np.uint8)
        xs.append(features(im))
        if len(examples)<12:examples.append(im)
    return np.array(xs),np.array(ys)

start=time.perf_counter()
try:
    x,y=dataset(100,600)
    mean,scale=x.mean(0),x.std(0)+1e-8
    clf=LogisticRegression(C=1,max_iter=1000,random_state=100).fit((x-mean)/scale,y)
    model={'type':'3D rendered onboard camera / logistic image feature classifier','mean':mean.tolist(),'scale':scale.tolist(),'weights':clf.coef_[0].tolist(),'bias':float(clf.intercept_[0]),'train_seed':100,'train_count':len(y)}
    (out/'detector_3d.json').write_text(json.dumps(model,indent=2),encoding='utf-8')
    metrics={}
    for name,seed,shift in [('validation',200,False),('shifted_test',300,True)]:
        tx,ty=dataset(seed,200,shift);prob=clf.predict_proba((tx-mean)/scale)[:,1]
        metrics[name]={'n':len(ty),'seed':seed,'balanced_accuracy':float(balanced_accuracy_score(ty,prob>=.5)),'confusion_matrix':confusion_matrix(ty,prob>=.5).tolist(),'review_rate':float(((prob>.2)&(prob<.8)).mean())}
    metrics['seconds']=time.perf_counter()-start
    metrics['source']='MuJoCo actual onboard RGB render, 3 equipment bodies, randomized light/view/noise. No real-world validation.'
    (out/'training_3d_metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    collage=Image.new('RGB',(960,960))
    for i,im in enumerate(examples):collage.paste(Image.fromarray(im),(i%3*320,i//3*240))
    collage.save(out/'training-3d-samples.jpg')
    np.savez_compressed(out/'training_3d_features.npz',x=x,y=y)
    print(json.dumps(metrics,indent=2))
finally:renderer.close();plant.close()
