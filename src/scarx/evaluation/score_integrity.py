"""Exact finite-path and sampling audit of local score eligibility, no autograd."""
import itertools
import numpy as np


def fixture(name):
    x=np.array([[1.,.1],[0.,.8],[.2,-.1],[0.,.1],[.3,.9],[.1,.7]],np.float64)
    wi=np.array([[.18,-.05],[-.07,.22]],np.float64);wr=np.array([[.14],[-.1]],np.float64)
    if name=='strong':x*=10;wi/=10
    elif name=='delay':x[:]=0;x[0]=[1.,.1];x[-1]=[.1,.8]
    elif name!='moderate':raise ValueError('Unknown fixture')
    return {'name':name,'x':x,'wi':wi,'wr':wr,'indices':np.array([[1],[0]]),
        'wo':np.array([[.8,-.7],[-.6,.9]],np.float64),'bias':np.array([.1,-.1]),'target':1,
        'leak':.85,'threshold':.65,'adaptation_strength':.08,'adaptation_decay':.95,
        'feature_decay':.92,'temperature':.4}


def paths(count=6,hidden=2):
    return np.array(list(itertools.product((0.,1.),repeat=count*hidden)),np.float64).reshape(-1,count,hidden)


def run(fix,spikes=None,uniforms=None,trace_clip=None,advantage_clip=None,baseline=-.7,omit_logit_mask=False):
    if (spikes is None)==(uniforms is None):raise ValueError('Exactly one history source')
    source=spikes if spikes is not None else uniforms
    b,t,h=source.shape;d=fix['x'].shape[1];k=fix['indices'].shape[1]
    v=np.zeros((b,h));a=v.copy();z=v.copy();f=v.copy();logprob=np.zeros(b)
    qin=np.zeros((b,h,d));qrec=np.zeros((b,h,k));ein=qin.copy();erec=qrec.copy()
    saturated=0
    for step in range(t):
        old=z.copy();x=fix['x'][step]
        v=fix['leak']*v+x@fix['wi'].T+np.sum(fix['wr'][None,:,:]*old[:,fix['indices']],axis=2)
        limit=fix['threshold']+fix['adaptation_strength']*a;logit=(v-limit)/fix['temperature']
        p=1/(1+np.exp(-np.clip(logit,-20,20)))
        z=spikes[:,step] if spikes is not None else (uniforms[:,step]<p).astype(np.float64)
        logprob+=np.sum(z*np.log(p)+(1-z)*np.log1p(-p),axis=1)
        qin=fix['leak']*qin+x[None,None,:]
        qrec=fix['leak']*qrec+old[:,fix['indices']]
        mask=(np.abs(logit)<20);saturated+=int(np.count_nonzero(~mask))
        residual=(z-p)/fix['temperature']
        if not omit_logit_mask:residual*=mask
        ein+=residual[:,:,None]*qin;erec+=residual[:,:,None]*qrec
        if trace_clip is not None:
            np.clip(ein,-trace_clip,trace_clip,out=ein);np.clip(erec,-trace_clip,trace_clip,out=erec)
        v-=z*limit;a=fix['adaptation_decay']*a+z;f=fix['feature_decay']*f+(1-fix['feature_decay'])*z
    logits=f@fix['wo'].T+fix['bias'];logits-=logits.max(1,keepdims=True)
    probabilities=np.exp(logits);probabilities/=probabilities.sum(1,keepdims=True)
    reward=np.log(probabilities[:,fix['target']]);advantage=reward-baseline
    if advantage_clip is not None:advantage=np.clip(advantage,-advantage_clip,advantage_clip)
    score=np.concatenate((ein.reshape(b,-1),erec.reshape(b,-1)),axis=1)
    return {'path_probability':np.exp(logprob),'reward':reward,'score':score,
        'estimator':advantage[:,None]*score,'saturated_elements':saturated}


def exact(fix):
    history=paths(len(fix['x']),len(fix['wi']));actual=run(fix,spikes=history)
    total=actual['path_probability'].sum();objective=float(actual['path_probability']@actual['reward'])
    gradient=actual['path_probability']@actual['estimator'];finite=[];eps=1e-5
    for name in ('wi','wr'):
        for index in np.ndindex(fix[name].shape):
            sides=[]
            for sign in (1,-1):
                changed={**fix,name:fix[name].copy()};changed[name][index]+=sign*eps
                value=run(changed,spikes=history);sides.append(value['path_probability']@value['reward'])
            finite.append((sides[0]-sides[1])/(2*eps))
    finite=np.array(finite)
    variants={}
    for name,kwargs in (('trace_clip10',{'trace_clip':10}),('advantage_clip2',{'advantage_clip':2}),
        ('both_clipped',{'trace_clip':10,'advantage_clip':2}),('omit_logit_mask',{'omit_logit_mask':True})):
        result=run(fix,spikes=history,**kwargs);mean=result['path_probability']@result['estimator']
        variants[name]={'gradient':mean.tolist(),'bias_l2':float(np.linalg.norm(mean-finite)),'bias_max_abs':float(np.max(np.abs(mean-finite)))}
    return {'probability_sum':float(total),'objective':objective,'score_gradient':gradient.tolist(),'finite_gradient':finite.tolist(),
        'max_absolute_error':float(np.max(np.abs(gradient-finite))),'relative_l2_error':float(np.linalg.norm(gradient-finite)/max(np.linalg.norm(finite),1e-12)),
        'oracle_pass':bool(abs(total-1)<1e-10 and np.max(np.abs(gradient-finite))<=1e-7 and np.linalg.norm(gradient-finite)/max(np.linalg.norm(finite),1e-12)<1e-5),
        'path_count':len(history),'variants':variants}


def variance(fix,seed,pairs=8192):
    rng=np.random.default_rng(seed);u=rng.random((pairs,len(fix['x']),len(fix['wi'])))
    first=run(fix,uniforms=u)['estimator'];independent=run(fix,uniforms=rng.random(u.shape))['estimator'];opposite=run(fix,uniforms=1-u)['estimator']
    out={}
    for name,value in (('single',first),('independent_pair',(first+independent)/2),('antithetic_pair',(first+opposite)/2)):
        out[name]={'mean':value.mean(0).tolist(),'variance':value.var(0,ddof=1).tolist(),'variance_trace':float(value.var(0,ddof=1).sum()),'sample_count':pairs}
    out['antithetic_to_independent_variance_ratio']=out['antithetic_pair']['variance_trace']/out['independent_pair']['variance_trace']
    return out
