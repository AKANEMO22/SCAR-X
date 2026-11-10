"""Public pinned visual control data/sampler; no Torch or serving imports."""
from pathlib import Path
import hashlib,itertools,json,os,tempfile
from scarx.evaluation.visual_words import VisualWords
from scarx.neural.language import tokenize

ROOT=Path(__file__).resolve().parents[1]
SEEDS=(33113,33119,33133)
KINDS=('single_color','single_shape','object_count','left_color');LANGUAGES=('vi','en')
SOURCE='runs/20261109T222354Z-visual-word-readout-v1-a11424'
DATA_SHA='0fb04abefe67451952a2d73634fded9c86243d5874356d68299ff047deecf6a2'


def atomic(path,value):
    from scarx.runtime.atomic_file import replace_with_retry
    fd,tmp=tempfile.mkstemp(dir=path.parent,suffix='.tmp')
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,allow_nan=False);f.flush();os.fsync(f.fileno())
        replace_with_retry(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def source_data():
    receipt=json.loads((ROOT/'artifacts/visual-word-dataset-latest.json').read_text(encoding='utf-8'))
    path=(ROOT/receipt['data']).resolve()
    if not path.is_relative_to(ROOT/'data/candidates') or receipt['sha256']!=DATA_SHA or hashlib.sha256(path.read_bytes()).hexdigest()!=DATA_SHA:raise ValueError('Pinned visualdataset changed')
    data=json.loads(path.read_text(encoding='utf-8'));train=[r for r in data['cases'] if r['split']=='train'];dev=[r for r in data['cases'] if r['split']=='development']
    if len(train)!=864 or len(dev)!=864 or len(data['scenes'])!=192 or data['automatic_learning_allowed'] is not False:raise ValueError('Visualdataset contract')
    images={};files=[receipt['data'],SOURCE+'/trials.json',SOURCE+'/manifest.json']
    for scene in data['scenes']:
        p=(path.parent/scene['file']).resolve()
        if not p.is_relative_to(path.parent) or p.stat().st_size>10000 or hashlib.sha256(p.read_bytes()).hexdigest()!=scene['sha256']:raise ValueError('Owned PNG source changed')
        images[scene['id']]=p.read_bytes();files.append(p.relative_to(ROOT).as_posix())
    vocab=['<eos>','<bos>']+sorted({t for r in train for t in tokenize(r['target'])})
    assert all(t in vocab for r in dev for t in tokenize(r['target']))
    rows=json.loads((ROOT/SOURCE/'trials.json').read_text(encoding='utf-8'));assert len(rows)==9
    return receipt,data,train,dev,images,vocab,rows,files


def calibrated(vocab,seed,train,images):
    m=VisualWords(vocab,seed);m.calibrate([{'prompt':r['prompt'],'image':images[r['scene_id']]} for r in train]);return m


def sample_stream(model,train,count):
    pools={}
    for kind,language in itertools.product(KINDS,LANGUAGES):
        values=[r for r in train if r['kind']==kind and r['language']==language]
        pools[(kind,language)]={target:[r for r in values if r['target']==target] for target in sorted({r['target'] for r in values})}
    for _ in range(count):
        kind=KINDS[int(model.sampler_rng.integers(4))];language=LANGUAGES[int(model.sampler_rng.integers(2))];group=pools[(kind,language)]
        target=sorted(group)[int(model.sampler_rng.integers(len(group)))];pool=group[target];yield pool[int(model.sampler_rng.integers(len(pool)))]


def verify_baseline(model,rows,seed):
    row=next(r for r in rows if r['seed']==seed and r['condition']=='full');path=ROOT/row['checkpoint']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
    old=VisualWords.loads(path.read_bytes())
    import numpy as np
    for n in ('w_in','w_rec','indices'):np.testing.assert_array_equal(getattr(old.core,n),getattr(model.core,n))
    for n in ('mean','m2'):np.testing.assert_array_equal(getattr(old.head,n),getattr(model.head,n))
    return row
