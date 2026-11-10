"""Read-only existing visual path and auxiliary closed-form probe diagnostics.

No neural weight updates, Torch, new encoder or serving policy. Probe weights
are disposable analysis arrays and must never format or replace neural answers.
"""
import hashlib,io,json,zipfile
import numpy as np
from .visual_words import VisualWords


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def protected(model):
    h=hashlib.sha256()
    for name in model.core.weight_names+('indices',):h.update(getattr(model.core,name).tobytes())
    h.update(model.head.mean.tobytes());h.update(model.head.m2.tobytes())
    h.update(json.dumps({'updates':model.core.updates,'experiences':model.experiences,'head_updates':model.head.updates,
        'head_count':model.head.count,'head_frozen':model.head.frozen,
        'sampler_rng':model.sampler_rng.bit_generator.state,'calibration_rng':model.calibration_rng.bit_generator.state},sort_keys=True).encode())
    return h.hexdigest()


def source_model(payload,control=False):
    if not control:return VisualWords.loads(payload)
    if len(payload)>4_000_000:raise ValueError('Bounded research control only')
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        if set(z.namelist())!={'reference.visualword','optimizer.npz','control.json'} or len(z.namelist())!=3 or sum(i.file_size for i in z.infolist())>4_000_000 or z.getinfo('control.json').file_size>65536:raise ValueError('Control research members')
        raw=z.read('reference.visualword');state=z.read('optimizer.npz');meta=json.loads(z.read('control.json'))
    checksum=meta.pop('metadata_sha256')
    if meta.get('format')!='scarx-isolated-visual-word-bptt-control-v1' or meta.get('control_only') is not True or hashlib.sha256(json.dumps(meta,sort_keys=True).encode()).hexdigest()!=checksum or hashlib.sha256(raw).hexdigest()!=meta['reference_sha256'] or hashlib.sha256(state).hexdigest()!=meta['optimizer_sha256']:raise ValueError('Control research checksum')
    return VisualWords.loads(raw)


def observe(model,question,image):
    """No target or task/scene annotations accepted in this forward interface."""
    before=protected(model);core=model.core;original=core.step;had_override='step' in core.__dict__;inputs=[]
    def record(x,track=True):inputs.append(np.asarray(x,np.float32).copy());return original(x,track)
    core.step=record
    try:
        model.prepare(question,image)
        prepared=np.concatenate([getattr(core,n) for n in ('voltage','adaptation','spikes','features')]).copy()
        visual=model.visual_feature.copy();q=model.question_feature.copy();bos=model.word_feature(1,0).copy()
    finally:
        if had_override:core.step=original
        else:del core.step
    tape=np.stack(inputs)
    if len(tape)!=36 or len(visual)!=48:raise ValueError('Existing fixed forward path changed')
    transformed=model.head.transform(bos);logits=model.head.weights@transformed+model.head.bias
    actual=int(model.head.probabilities(bos,model.output_mask).argmax())
    assert protected(model)==before
    rgb=tape[16:32,model.encoder.image_start:model.encoder.image_start+48].reshape(-1)
    return {'vectors':{'question':q,'rgb_question':np.concatenate((rgb,q)),'prepared_state':prepared,'visual_trace':visual,
        'bos_feature':bos,'head_feature':transformed,'logits':logits},'input_tape_sha256':digest(tape),'first_token':actual,
        'clipped_fraction':float(np.mean(np.abs(transformed)>=5.)),'ticks':36}


def ridge(train,dev,labels,classes,exposures,*,mask=None,shuffle_seed=None):
    """One alpha=1 weighted ridge; exact exposure multiset, no dev labels/stats."""
    x=np.asarray(train,np.float64);d=np.asarray(dev,np.float64);y=np.asarray(labels,np.int64);e=np.asarray(exposures,np.int64)
    if x.ndim!=2 or d.ndim!=2 or d.shape[1]!=x.shape[1] or y.shape!=(len(x),) or e.ndim!=1 or not len(e) or e.min()<0 or e.max()>=len(x) or y.min()<0 or y.max()>=classes or not np.isfinite(x).all() or not np.isfinite(d).all():raise ValueError('Finite diagnostic probe arrays')
    counts=np.bincount(e,minlength=len(x)).astype(np.float64);n=float(len(e))
    mean=(x*counts[:,None]).sum(0)/n;scale=np.maximum(np.sqrt(((x-mean)**2*counts[:,None]).sum(0)/n),.05)
    z=(x-mean)/scale;zd=(d-mean)/scale;exposure_targets=y[e].copy()
    if shuffle_seed is not None:exposure_targets=np.random.default_rng(shuffle_seed).permutation(exposure_targets)
    totals=np.zeros((len(x),classes),np.float64);np.add.at(totals,(e,exposure_targets),1.)
    mean_y=totals.sum(0)/n;gram=z.T@(z*counts[:,None]);gram.flat[::len(gram)+1]+=1.
    weights=np.linalg.solve(gram,z.T@(totals-counts[:,None]*mean_y))
    train_scores=z@weights+mean_y;dev_scores=zd@weights+mean_y
    if mask is not None:
        mask=np.asarray(mask,bool)
        if mask.shape!=(classes,) or not mask.any():raise ValueError('Diagnostic outputmask')
        train_scores=np.where(mask,train_scores,-np.inf);dev_scores=np.where(mask,dev_scores,-np.inf)
    if not np.isfinite(weights).all():raise ValueError('Nonfinite diagnostic ridge')
    return {'train':train_scores.argmax(1),'dev':dev_scores.argmax(1),'exposure_counts':counts.astype(np.int64),
        'fit_array_bytes':int(weights.nbytes+gram.nbytes+totals.nbytes),'train_sample_count':len(e),
        'training_label_histogram':np.bincount(exposure_targets,minlength=classes).tolist()}


def accuracy(prediction,labels,cases,weights=None):
    p=np.asarray(prediction);y=np.asarray(labels);ok=p==y
    if len(p)!=len(cases) or p.shape!=y.shape:raise ValueError('Probe score alignment')
    w=np.ones(len(cases)) if weights is None else np.asarray(weights,dtype=float)
    if w.shape!=ok.shape or not np.isfinite(w).all() or np.any(w<0) or not w.sum():raise ValueError('Probe scoreweights')
    def score(indices):
        indices=np.asarray(indices,bool);mass=w[indices].sum()
        return float(np.sum(ok[indices]*w[indices])/mass) if mass else None
    def macro(indices):
        values=[]
        for kind in ('single_color','single_shape','object_count','left_color'):
            for language in ('vi','en'):
                r=score([take and c['kind']==kind and c['language']==language for take,c in zip(indices,cases)])
                if r is not None:values.append(r)
        return float(np.mean(values)) if values else None
    all_cases=np.ones(len(cases),bool)
    return {'macro':macro(all_cases),'micro':score(all_cases),
        'by_kind':{k:macro([c['kind']==k for c in cases]) for k in ('single_color','single_shape','object_count','left_color')},
        'by_language':{l:macro([c['language']==l for c in cases]) for l in ('vi','en')},
        'novel_macro':macro([c['novel_composition'] for c in cases])}


def collision(hashes,labels,cases):
    if not len(hashes) or len(hashes)!=len(labels) or len(hashes)!=len(cases):raise ValueError('Collision alignment')
    group_mass={};groups={};bucket={}
    for c in cases:group_mass[(c['kind'],c['language'])]=group_mass.get((c['kind'],c['language']),0)+1
    for h,label,c in zip(hashes,labels,cases):
        g=groups.setdefault(h,{});g[int(label)]=g.get(int(label),0)+1
        b=bucket.setdefault(h,{});b[int(label)]=b.get(int(label),0.)+1/(len(group_mass)*group_mass[(c['kind'],c['language'])])
    return {'cases':len(cases),'distinct_vectors':len(groups),'collision_groups':sum(sum(g.values())>1 for g in groups.values()),
        'conflicting_groups':sum(len(g)>1 for g in groups.values()),'oracle_micro_ceiling':sum(max(g.values()) for g in groups.values())/len(cases),
        'oracle_macro_ceiling':float(sum(max(g.values()) for g in bucket.values()))}


def counterfactual_pairs(cases,semantic_labels):
    by_prompt={};by_scene={}
    for i,c in enumerate(cases):by_prompt.setdefault(c['prompt'],[]).append(i);by_scene.setdefault(c['scene_id'],[]).append(i)
    result={}
    for kind,groups in (('image',by_prompt),('question',by_scene)):
        pairs=set()
        for group in groups.values():
            for i in group:
                j=next((j for j in group if semantic_labels[i]!=semantic_labels[j]),None)
                if j is not None:pairs.add(tuple(sorted((i,j))))
        result[kind]=sorted(pairs)
    return result


def pair_summary(hashes,pairs,vectors=None):
    if not pairs:return {'pairs':0,'exact_collapse_fraction':None,'median_l2':None}
    same=[hashes[i]==hashes[j] for i,j in pairs]
    distance=None if vectors is None else float(np.median([np.linalg.norm(vectors[i].astype(np.float64)-vectors[j]) for i,j in pairs]))
    return {'pairs':len(pairs),'exact_collapse_fraction':float(np.mean(same)),'median_l2':distance}


def rank_summary(matrix):
    x=np.asarray(matrix,np.float64);x=x-x.mean(0);s=np.linalg.svd(x,compute_uv=False)
    return {'dimensions':x.shape[1],'centered_rank_rtol1e6':int(np.count_nonzero(s>s.max()*1e-6)) if len(s) and s.max()>0 else 0,
        'constant_channels':int(np.count_nonzero(np.ptp(matrix,axis=0)==0))}
