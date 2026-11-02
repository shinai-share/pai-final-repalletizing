"""Synthetic camera-panel generator and trainable image evidence classifier.

This is a proxy inspection camera, not a photorealistic thermal sensor.
Labels are available only to the generator/evaluator, never the classifier.
"""
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

def panel(label, rng, shift=False):
    light = rng.uniform(.35 if shift else .55, 1.15)
    bg = rng.integers(25,80)
    im = Image.new('RGB',(64,64),(int(bg),)*3)
    d = ImageDraw.Draw(im)
    x,y = rng.integers(20,40,2)
    r = int(rng.integers(8,15))
    color = (230,55,35) if label else (35,210,100)
    color = tuple(int(np.clip(c*light+rng.normal(0,12),0,255)) for c in color)
    d.ellipse((x-r,y-r,x+r,y+r),fill=color,outline=(150,150,150),width=2)
    d.rectangle((5,53,59,57),fill=(85,90,96))
    if rng.random() < (.5 if shift else .25):
        d.rectangle((int(x),0,int(x)+rng.integers(3,9),64),fill=(int(bg),)*3)
    a = np.asarray(im).astype(float)+rng.normal(0,10 if shift else 4,(64,64,3))
    return np.clip(a,0,255).astype(np.uint8)

def features(a):
    x = np.asarray(a,dtype=float)/255
    rg = x[:,:,0]-x[:,:,1]
    return np.array([np.maximum(rg,0).mean(), np.maximum(-rg,0).mean(),
                     (rg>.15).mean(), (rg<-.15).mean(), x.mean(), x.std()])

class Detector:
    def __init__(self, path):
        self.model = json.loads(Path(path).read_text(encoding='utf-8'))

    def predict(self, image):
        m=self.model
        x=(features(image)-np.array(m['mean']))/np.array(m['scale'])
        z=float(x@np.array(m['weights'])+m['bias'])
        return float(1/(1+np.exp(-np.clip(z,-50,50))))

def dataset(seed, n, shift=False):
    rng=np.random.default_rng(seed)
    y=rng.integers(0,2,n)
    return np.array([features(panel(label,rng,shift)) for label in y]),y

def train(output):
    """Reproducible logistic training; validation and shifted test seeds disjoint."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix, balanced_accuracy_score
    out=Path(output); out.mkdir(parents=True,exist_ok=True)
    x,y=dataset(100,2400)
    mean,scale=x.mean(0),x.std(0)+1e-8
    model=LogisticRegression(C=1.,max_iter=1000,random_state=100).fit((x-mean)/scale,y)
    payload={'mean':mean.tolist(),'scale':scale.tolist(),'weights':model.coef_[0].tolist(),'bias':float(model.intercept_[0]),'train_seed':100,'train_count':len(y),'type':'logistic_regression_on_image_features'}
    (out/'detector.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    metrics={}
    for name,seed,shift in [('validation',200,False),('shifted_test',300,True)]:
        tx,ty=dataset(seed,800,shift)
        prob=model.predict_proba((tx-mean)/scale)[:,1]
        pred=prob>=.5
        metrics[name]={'n':len(ty),'seed':seed,'balanced_accuracy':float(balanced_accuracy_score(ty,pred)), 'confusion_matrix':confusion_matrix(ty,pred).tolist(),'review_rate':float(((prob>.2)&(prob<.8)).mean())}
    metrics['limitations']='Synthetic colored indicator recognition only; not real defect or sim-to-real validation.'
    (out/'training_metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    np.savez_compressed(out/'training_features.npz',x=x,y=y)
    return metrics
