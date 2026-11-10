"""Fixed owned visual-composition splits; no learned renderer/answer inference."""
import itertools
import numpy as np
from .vision_fixtures import expected,eligible

SHAPES=('circle','square','triangle');COLORS=('red','green')


def signature(scene):return tuple((o['shape'],o['color']) for o in scene['objects'])


def scene_split(split):
    if split not in ('train','development'):raise ValueError('Unknown fixedsplit')
    rng=np.random.default_rng(33101 if split=='train' else 33103);scenes=[]
    def add(shapes,colors):
        n=len(shapes);objects=[]
        for i,(shape,color) in enumerate(zip(shapes,colors)):
            if n==1:x=64+int(rng.integers(-3,4));y=64+int(rng.integers(-3,4));h=int(rng.integers(20,29))
            elif n==2:x=(36,92)[i]+int(rng.integers(-3,4));y=64+int(rng.integers(-4,5));h=int(rng.integers(14,19))
            else:x=(24,64,104)[i]+int(rng.integers(-2,3));y=64+int(rng.integers(-3,4));h=int(rng.integers(12,17))
            objects.append({'shape':shape,'color':color,'x':x,'y':y,'halfsize':h})
        scenes.append({'id':f'{split}-{len(scenes)+1:03d}','split':split,'objects':objects})
    singles=(('circle','red'),('circle','green'),('square','red'),('triangle','green')) if split=='train' else (('square','green'),('triangle','red'),('circle','red'),('circle','green'))
    for shape,color in singles:
        for _ in range(6):add((shape,),(color,))
    pairs=(('circle','square'),('square','triangle'),('triangle','circle')) if split=='train' else (('square','circle'),('triangle','square'),('circle','triangle'))
    for shapes,colors in itertools.product(pairs,itertools.product(COLORS,repeat=2)):
        for _ in range(3):add(shapes,colors)
    triples=(('circle','square','triangle'),('square','triangle','circle'),('triangle','circle','square')) if split=='train' else (('circle','triangle','square'),('triangle','square','circle'),('square','circle','triangle'))
    for shapes,colors in itertools.product(triples,(('red','red','green'),('green','green','red'),('red','green','red'),('green','red','green'))):
        for _ in range(3):add(shapes,colors)
    assert len(scenes)==96;return scenes


def qa_cases(scenes,questions,train_signatures):
    out=[]
    for scene in scenes:
        allowed={1,2,5,6} if scene['split']=='train' else {3,4,7,8}
        for q in questions:
            if int(q['id'].rsplit('_',1)[1]) not in allowed or not eligible(scene,q['kind']):continue
            out.append({'id':scene['id']+'/'+q['id'],'scene_id':scene['id'],'split':scene['split'],'question_id':q['id'],
                'prompt':q['question'],'kind':q['kind'],'language':q['language'],
                'target':expected(scene,q['kind'],q['language']),'novel_composition':signature(scene) not in train_signatures,
                'target_source':'Owned renderer manifest, not teacher/model answer'})
    return out
