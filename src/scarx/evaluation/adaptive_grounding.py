"""Offline developmentdata only. No model instances, inference, or training."""
import hashlib,io,itertools,json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
from .vision_fixtures import PALETTE,COLORS_VI,SHAPES_VI

SHAPES=('circle','square','triangle')
FACT_SOURCES={
 'identity':{'path':'README.md','anchors':['SCAR-X','ALIF']},
 'limited_abilities':{'path':'README.md','anchors':['Chưa có hiểu screenshot','Bản thử nghiệm còn yếu']},
 'learning_tab':{'path':'src/scarx/app/learning_channel.py','anchors':['explicit lesson from learning tab','learning_tab_feedback']},
 'guarded_updates':{'path':'src/scarx/app/generation.py','anchors':['LearningGuardRejected','guard_checked']},
 'not_self_labels':{'path':'src/scarx/learning/dialogue.py','anchors':['NEVER evidence of correctness']},
 'persistence':{'path':'src/scarx/app/generation.py','anchors':['def _persist','def _pack']}}


def source_receipts(root):
    result={}
    for name,source in FACT_SOURCES.items():
        path=Path(root)/source['path'];raw=path.read_bytes();text=raw.decode('utf-8')
        if not all(anchor in text for anchor in source['anchors']):raise ValueError('Source anchors changed: '+name)
        result[name]={**source,'sha256':hashlib.sha256(raw).hexdigest(),'basis':'Manual projectscope review with code/docs anchors; checksum alone does not prove semantictruth'}
    return result


def scene_bank():
    rng=np.random.default_rng(43301);scenes=[]
    def geometry(shapes):
        n=len(shapes);centers={1:(64,),2:(36,92),3:(24,64,104)}[n]
        lo,hi=(22,27) if n==1 else (13,18) if n==2 else (11,15)
        return [{'shape':s,'x':x+int(rng.integers(-2,3)),'y':64+int(rng.integers(-2,3)),'halfsize':int(rng.integers(lo,hi))} for s,x in zip(shapes,centers)]
    def add(group,objects,colors):
        scenes.append({'id':f'image-{len(scenes)+1:03d}','counterfactual_group':group,'objects':[{**o,'color':c} for o,c in zip(objects,colors)]})
    for shape in SHAPES:
        for replica in range(2):
            objects=geometry((shape,));group=f'single-{shape}-{replica}'
            objects[0]['halfsize']=22+2*replica+int(rng.integers(0,2))
            add(group,objects,('red',));add(group,objects,('green',))
    for index,shapes in enumerate(itertools.permutations(SHAPES,2)):
        objects=geometry(shapes);group=f'pair-{index}'
        add(group,objects,('red','green'));add(group,objects,('green','red'))
    for index,shapes in enumerate(itertools.permutations(SHAPES)):
        objects=geometry(shapes);group=f'triple-{index}'
        add(group,objects,('red','red','green'));add(group,objects,('green','green','red'))
    if len(scenes)!=36 or Counter(len(s['objects']) for s in scenes)!={1:12,2:12,3:12}:raise ValueError('Fixed balanced scenes')
    return scenes


def pixel_objects(payload):
    """Independent component/color/row-width oracle from the actual PNG raster."""
    with Image.open(io.BytesIO(payload)) as image:
        if image.size!=(128,128) or image.mode!='RGB':raise ValueError('Owned RGB fixture required')
        rgb=np.asarray(image).copy()
    palette={tuple(value):key for key,value in PALETTE.items()};pixels=set(map(tuple,rgb.reshape(-1,3)))
    if not pixels.issubset(set(palette)|{(255,255,255)}):raise ValueError('Unexpected raster palette')
    remaining=set(map(tuple,np.argwhere(np.any(rgb!=255,axis=2))));out=[]
    while remaining:
        first=remaining.pop();pending=[first];component=[first]
        while pending:
            y,x=pending.pop()
            for other in ((y+1,x),(y-1,x),(y,x+1),(y,x-1)):
                if other in remaining:remaining.remove(other);pending.append(other);component.append(other)
        y=np.array([p[0] for p in component]);x=np.array([p[1] for p in component]);colors=set(map(tuple,rgb[y,x]))
        if len(colors)!=1:raise ValueError('Mixedcolor/touching components')
        top,bottom,left,right=int(y.min()),int(y.max()),int(x.min()),int(x.max());width=right-left+1;height=bottom-top+1
        if width!=height or width<9:raise ValueError('Unexpected fixture extent')
        row_width=np.bincount(y-top,minlength=height);density=len(component)/(width*height)
        if np.all(row_width==width):shape='square'
        elif row_width[0]==1 and row_width[-1]==width and np.all(np.diff(row_width)>=0) and .45<=density<=.6:shape='triangle'
        elif np.array_equal(row_width,row_width[::-1]) and .7<=density<=.9 and row_width[height//2]==width:shape='circle'
        else:raise ValueError('Raster not recognized by independent geometric oracle')
        out.append({'shape':shape,'color':palette[next(iter(colors))],'x':(left+right)//2,'y':(top+bottom)//2,'halfsize':(width-1)//2})
    return sorted(out,key=lambda o:o['x'])


def eligible(objects,kind):
    return kind=='object_count' or (len(objects)==1 if kind in ('single_color','single_shape') else len(objects)==2 if kind=='left_color' else False)


def semantic_value(objects,kind,language):
    if language not in ('vi','en') or not eligible(objects,kind):raise ValueError('Question/scene mismatch')
    if kind=='object_count':return str(len(objects))
    obj=objects[0] if kind.startswith('single_') else min(objects,key=lambda o:o['x'])
    if kind=='single_shape':return SHAPES_VI[obj['shape']] if language=='vi' else obj['shape']
    return COLORS_VI[obj['color']] if language=='vi' else obj['color']


def reference_sentence(value,kind,language):
    if language=='vi':
        return f'Có {value} vật thể.' if kind=='object_count' else f'Đó là {value}.' if kind=='single_shape' else f'Vật bên trái có màu {value}.' if kind=='left_color' else f'Vật thể có màu {value}.'
    return (f'There is {value} object.' if value=='1' else f'There are {value} objects.') if kind=='object_count' else f'It is a {value}.' if kind=='single_shape' else f'The object on the left is {value}.' if kind=='left_color' else f'The object is {value}.'


def distribution_ceiling(cases,key):
    buckets=defaultdict(Counter)
    for row in cases:buckets[key(row)][row['semantic_target']]+=1
    return sum(max(c.values()) for c in buckets.values())/len(cases)


def chat_examples(wordings,references):
    rows=[r for r in wordings if r['modality']=='chat'];answers=references['answers']
    if len(rows)!=95 or set(answers)!={r['id'] for r in rows}:raise ValueError('Every reviewed chatprompt needs exactly one authored reference')
    out=[]
    for r in rows:
        text=answers[r['id']]
        if not isinstance(text,str) or not 1<=len(text)<=512:raise ValueError('Bounded reference answer')
        facts=['identity','limited_abilities'] if r['family']=='identity' else ['limited_abilities','guarded_updates'] if r['family']=='capability' else ['learning_tab','guarded_updates','not_self_labels','persistence'] if r['family']=='learning' else []
        out.append({'id':r['id'],'input':{'prompt':r['prompt']},'language':r['language'],'family':r['family'],'reference_answer':text,
                    'reference_kind':'project_behavior' if facts else 'social_style','fact_sources':facts,
                    'rubric':{'intent':r['family'],'reply_language':r['language'],'exact_reference_required':False,'unsupported_capability_claims_allowed':False},
                    'label_basis':'Projectauthored source-reviewed reference, never teacher/model output','automatic_learning_allowed':False})
    return out


def model_input(row,dataset_root):
    """Only prompt/imagebytes; no annotation, family, ID or filename as input."""
    if set(row['input'])=={'prompt'}:return {'prompt':row['input']['prompt']}
    if set(row['input'])!={'prompt','image_file'}:raise ValueError('Unknown input schema')
    root=Path(dataset_root).resolve();path=(root/row['input']['image_file']).resolve()
    if not path.is_relative_to(root/'images') or path.suffix!='.png' or path.stat().st_size>20_000:raise ValueError('Owned image path/budget required')
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=row['image_sha256']:raise ValueError('Image checksum')
    return {'prompt':row['input']['prompt'],'image_bytes':raw}
