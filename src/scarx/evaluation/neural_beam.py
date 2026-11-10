"""Bounded search on one shared-weight ALIF clone; virtual state is diagnostic only."""
import math,time
import numpy as np
from ..neural.language import render


def capture(model):
    return {'arrays':{n:getattr(model.core,n).copy() for n in model.core.state_names},
        'ticks':model.ticks,'last_active':model.last_active.copy(),'active_count':model.active_count}


def restore(model,state):
    for name,value in state['arrays'].items():setattr(model.core,name,value.copy())
    model.ticks=state['ticks'];model.last_active=state['last_active'].copy();model.active_count=state['active_count']


def penalty(length,alpha):return ((5+length)/6)**alpha


def decode(model,prompt,width=4,alpha=.6,max_words=24,node_budget=256,seconds=5.):
    if type(width) is not int or not 1<=width<=8 or alpha not in (0.,.6) or type(max_words) is not int or not 1<=max_words<=32 or type(node_budget) is not int or not 1<=node_budget<=256 or not 0<seconds<=5:raise ValueError('Invalid bounded search settings')
    began=time.perf_counter();context=model.context(prompt);fingerprint=model.core.fingerprint();weight_ids=tuple(id(getattr(model.core,n)) for n in model.core.weight_names)
    initial=model.ticks;model.boundary();work_ticks=model.ticks-initial;root=capture(model)
    active=[{'ids':[],'previous':1,'logp':0.,'probs':[],'state':root}];finished=[];expanded=0;peak_state_bytes=0;reason='length'
    def score(node):return node['logp']/penalty(len(node['probs']),alpha)
    def memory(nodes):
        unique={id(n['state']):n['state'] for n in nodes}
        return sum(sum(a.nbytes for a in s['arrays'].values())+s['last_active'].nbytes for s in unique.values())
    for position in range(max_words):
        children=[];budget=False
        for node in active:
            if expanded>=node_budget or time.perf_counter()-began>seconds:budget=True;reason='work_budget';break
            restore(model,node['state']);before=model.ticks
            p=model.word_step(node['previous'],context,position,False,prompt);work_ticks+=model.ticks-before;expanded+=1
            state=capture(model);valid=np.flatnonzero(p>0)
            # Stable descending probabilities preserve lower-index ties, as argmax.
            best=valid[np.argsort(-p[valid],kind='stable')[:width]]
            for token in best:
                probability=float(p[token]);child={'ids':node['ids']+([] if token==0 else [int(token)]),'previous':int(token),
                    'logp':node['logp']+math.log(max(probability,1e-12)),'probs':node['probs']+[probability],'state':state}
                if token==0:child['finished']=True;finished.append(child)
                else:child['finished']=False;children.append(child)
        finished=sorted(finished,key=score,reverse=True)[:width]
        if children:active=sorted(children,key=lambda n:n['logp'],reverse=True)[:width]
        elif not budget:active=[]
        peak_state_bytes=max(peak_state_bytes,memory(active+finished+children))
        if budget:break
        if not active:reason='eos';break
        if finished and max(score(n) for n in finished)>=max(n['logp']/penalty(max_words,alpha) for n in active):reason='eos';break
    if finished:chosen=max(finished,key=score);stopped=True;stop_reason='eos'
    elif active:chosen=max(active,key=score);stopped=False;stop_reason=reason
    else:raise RuntimeError('No neural search hypothesis')
    restore(model,chosen['state'])
    if model.core.fingerprint()!=fingerprint or tuple(id(getattr(model.core,n)) for n in model.core.weight_names)!=weight_ids:raise RuntimeError('Search changed/shared weights identity')
    tokens=[model.vocabulary[i] for i in chosen['ids']]
    return {'text':render(tokens),'tokens':tokens,'stopped':stopped,'stop_reason':stop_reason,
        'mean_probability':float(np.mean(chosen['probs'])),'source':'same ALIF neural probabilities; bounded generic beam, no target/template',
        'search':{'width':width,'alpha':alpha,'expanded_word_steps':expanded,'node_budget':node_budget,'computation_ticks':work_ticks,
            'selected_virtual_ticks':model.ticks-initial,'peak_unique_state_array_bytes':peak_state_bytes,
            'shared_weights_once':True,'seconds':time.perf_counter()-began,'budget_exhausted':reason=='work_budget',
            'runtime_clock_compatibility_proven':False}}
