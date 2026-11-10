"""Readonly prefix/task/credit diagnostics; never neurallearning or a decoder."""
import hashlib,io,json,zipfile
import numpy as np
from .direct_words import DirectWords
from ..neural.language import render


def load_reference(payload,control=False):
    if not control:return DirectWords.loads(payload)
    if len(payload)>6_000_000:raise ValueError('Boundedreadonlycontrol wrapper')
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        if set(z.namelist())!={'reference.words','control.json'} or len(z.namelist())!=2 or sum(i.file_size for i in z.infolist())>6_000_000 or z.getinfo('control.json').file_size>65536:raise ValueError('Readonlycontrol members')
        raw=z.read('reference.words');meta=json.loads(z.read('control.json'))
    checksum=meta.pop('metadata_sha256')
    if meta.get('format')!='scarx-direct-word-bptt-control-v1' or meta.get('control_only') is not True or hashlib.sha256(json.dumps(meta,sort_keys=True).encode()).hexdigest()!=checksum or hashlib.sha256(raw).hexdigest()!=meta['reference_sha256']:raise ValueError('Readonlycontrol checksum')
    return DirectWords.loads(raw)


def protected(model):
    h=hashlib.sha256()
    for name in model.core.weight_names+('indices',):h.update(getattr(model.core,name).tobytes())
    for name in ('mean','m2'):h.update(getattr(model.head,name).tobytes())
    for name in ('moment','variance'):h.update(getattr(model.learner,name).tobytes())
    value={'core_updates':model.core.updates,'head_updates':model.head.updates,'head_count':model.head.count,'head_frozen':model.head.frozen,
        'responses_seen':model.responses_seen,'hidden_updates':model.learner.hidden_updates,'segments_learned':model.learner.segments_learned,
        'signal_rng':model.learner.signal_rng.bit_generator.state,'sampler_rng':model.sampler_rng.bit_generator.state,'calibration_rng':model.calibration_rng.bit_generator.state,
        'condition':model.learner.condition,'vocabulary':model.vocabulary}
    h.update(json.dumps(value,sort_keys=True,ensure_ascii=False).encode());return h.hexdigest()


def free_forward(model,prompt):
    """No targetargument. Previous tokens exclusively neural, original36policy."""
    ticks=model.ticks;model.prepare(prompt);previous=1;ids=[];rows=[];stopped=False
    for position in range(8):
        f=model.word(previous,position);p=model.head.probabilities(f,model.learner.output_mask);token=int(p.argmax())
        rows.append({'predicted':token,'probabilities':p.copy(),'tick':model.ticks,'spikes':model.core.spikes.copy(),'voltage':model.core.voltage.copy()})
        if token==0:stopped=True;break
        ids.append(token);previous=token
    model.learner.segment_active=False
    return {'text':render([model.vocabulary[t] for t in ids]),'tokens':ids,'stopped':stopped,'rows':rows,'actual_ticks':model.ticks-ticks}


def prefix_forward(model,prompt,previous_tokens,*,zero_question=False,measure_credit=False):
    """Teacherprovidedprefix explicitlydiagnostic; currenttarget NOT an input."""
    if not isinstance(previous_tokens,list) or len(previous_tokens)>7 or any(type(t) is not int or not 0<t<len(model.vocabulary) or t==1 for t in previous_tokens):raise ValueError('Boundedexplicitprefix')
    if measure_credit and model.learner.condition!='correct':raise ValueError('Potentialunshuffledcredit measured ONLYcorrectsources')
    had_hook='_context' in model.__dict__;original=model._context
    if zero_question:
        def blank(text):return np.zeros_like(original(text)) # retain validation before zeroing
        model._context=blank
    try:
        model.prepare(prompt,measure_credit);rows=[];previous=1
        for position in range(len(previous_tokens)+1):
            f=model.word(previous,position);p=model.head.probabilities(f,model.learner.output_mask)
            row={'predicted':int(p.argmax()),'probabilities':p.copy()}
            if measure_credit:
                row.update(feature=f.copy(),eligibility=model.learner.e_f.copy())
            rows.append(row)
            if position<len(previous_tokens):previous=previous_tokens[position]
        return rows
    finally:
        model.learner.segment_active=False
        if zero_question:
            if had_hook:model._context=original
            else:del model._context


def score_prefix(rows,gold):
    if len(rows)!=len(gold):raise ValueError('Prefixscorealignment')
    return [{'correct':r['predicted']==t,'nll':-float(np.log(max(float(r['probabilities'][t]),1e-12))),
        'predicted':r['predicted'],'target':t} for r,t in zip(rows,gold)]


def potential_direction(model,row,target):
    """Finalstate hypothetical35signal, not an applied/historical weightdelta."""
    p=row['probabilities'];error=-p.copy();error[target]+=1;scale=np.maximum(np.sqrt(model.head.m2/(model.head.count-1)),.05).astype(np.float32)
    raw=(row['feature']-model.head.mean.astype(np.float32))/scale
    signal=(model.head.weights.T@error)*(np.abs(raw)<5)/scale
    value=signal[:,None]*row['eligibility']
    if not np.isfinite(value).all():raise ValueError('Nonfinite readonlypotentialdirection')
    return value


def cosine(a,b):
    a=np.asarray(a,np.float64).ravel();b=np.asarray(b,np.float64).ravel();denom=np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.clip(a@b/denom,-1,1)) if denom>1e-12 else None


def credit_summary(model,directions):
    first=directions[0];later=np.sum(directions[1:],axis=0);all_sum=first+later;v=len(model.vocabulary)
    groups={'previous_word':slice(0,v),'question':slice(v,v+128),'reserved_image':slice(v+128,v+192),'control_flags':slice(v+192,v+198),'recurrent':slice(v+198,None)}
    norm=lambda a:float(np.linalg.norm(np.asarray(a,np.float64)))
    return {'first_norm':norm(first),'later_sum_norm':norm(later),'first_later_cosine':cosine(first,later),'first_all_cosine':cosine(first,all_sum),
        'groups':{name:{'first_norm':norm(first[:,s]),'later_sum_norm':norm(later[:,s])} for name,s in groups.items()},
        'scope':'Potentialunpermuted35direction at FINALfixedcheckpoint, never applied or reconstructedpasttrainingdelta.'}


def corpus_structure(target_sequences):
    branches={};total=nonempty=deterministic=punct_or_eos=0
    for tokens,vocabulary in target_sequences:
        for position,token in enumerate(tokens):
            prefix=tuple(tokens[:position]);branches.setdefault(prefix,set()).add(token);total+=1
            if position:nonempty+=1
            if token==0 or not any(ch.isalnum() for ch in vocabulary[token]):punct_or_eos+=1
    for tokens,_ in target_sequences:
        for position in range(1,len(tokens)):deterministic+=int(len(branches[tuple(tokens[:position])])==1)
    return {'targets':len(target_sequences),'unique_first_words':len({t[0] for t,_ in target_sequences}),'total_positions_uniform_templates':total,
        'later_positions':nonempty,'later_fraction':nonempty/total,'punctuation_or_EOS_positions':punct_or_eos,
        'nonempty_prefix_determines_next_fraction':deterministic/nonempty,'PREFIX_STRUCTURE_SHORTCUT':deterministic==nonempty,
        'scope':'Finiteownedtemplateproperty only; not neuralanswers, skill or deployedlookup.'}
