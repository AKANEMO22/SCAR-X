"""ISOLATED BACKPROP CONTROL. Never imported by production neural modules.

Same ALIF forward equations, sparse topology and initial weights. A triangular
surrogate derivative supplies BPTT. All synapses train, unlike rotating blocks
in the local rule; this compute difference is disclosed in the report.
"""
import json
import hashlib,io,zipfile
from pathlib import Path
import time
import numpy as np
import torch
from ..neural.core import Config, SpikingCore
from ..runtime.resources import ResourceMonitor
from .tasks import episodes, accuracy
from .runs import new_run, write_json


class Spike(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x):
        ctx.save_for_backward(x)
        return (x>=0).float()

    @staticmethod
    def backward(ctx,grad):
        (x,) = ctx.saved_tensors
        return grad*(1-x.abs()).clamp(min=0)


class BPTTControl:
    def __init__(self,config):
        self.config = config
        base = SpikingCore(config)
        self.parameters = [torch.tensor(getattr(base,n),requires_grad=True) for n in ("w_in","w_rec","w_out","bias")]
        self.indices = torch.tensor(base.indices,dtype=torch.long)
        self.optimizer = torch.optim.SGD(self.parameters,lr=config.readout_lr)
        self.logits = None

    def forward(self,sequence,track=False):
        c = self.config
        wi,wr,wo,b = self.parameters
        v,a,z,f = [torch.zeros(c.hidden) for _ in range(4)]
        with torch.set_grad_enabled(track):
            for item in sequence:
                x = torch.from_numpy(item)
                v = c.leak*v+wi@x+(wr*z[self.indices]).sum(dim=1)
                limit = c.threshold+c.adaptation_strength*a
                z = Spike.apply(v-limit)
                v = v-z*limit
                a = c.adaptation_decay*a+z
                f = c.feature_decay*f+(1-c.feature_decay)*z
            self.logits = wo@f+b
            return self.logits.softmax(dim=0).detach().numpy().copy()

    def learn(self,target):
        self.optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(self.logits[None,:],torch.tensor([target]))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters,1.)
        self.optimizer.step()
        with torch.no_grad():
            for p in self.parameters: p.clamp_(-self.config.weight_clip,self.config.weight_clip)
        return float(loss.detach())


def run_control(root):
    torch.set_num_threads(2)
    settings = json.loads((Path(root)/"configs/phase2.json").read_text())
    run = new_run(root,"bptt-control-v1",settings)
    rows = []
    with ResourceMonitor() as monitor:
        for seed in settings["seeds"]:
            c = Config(inputs=5,hidden=settings["hidden"],fan_in=settings["fan_in"],plastic_rows=settings["plastic_rows"],seed=seed)
            m = BPTTControl(c)
            start = time.perf_counter()
            for x,y in episodes(1000+seed,settings["train_examples"]):
                m.forward(x,track=True); m.learn(y)
            row = {"seed":seed,"accuracy":float(accuracy(m,list(episodes(2000+seed,settings["test_examples"]))).mean()),
                   "delayed_accuracy":float(accuracy(m,list(episodes(2500+seed,settings["test_examples"],delay=8))).mean()),
                   "seconds":time.perf_counter()-start}
            rows.append(row)
            print(row,flush=True)
            monitor.check()
    result = {"control_only":True,"trials":rows,"resources":monitor.report(),
              "torch_version":torch.__version__,"device":"cpu","continual_test":"not_run",
              "note":"Matched forward ALIF/topology/initialization and 512 training examples; surrogate BPTT updates all weights, uses temporal graph. Not local learning."}
    write_json(run/"result.json",result)
    (run/"report.md").write_text("# Backprop control only\n\n```json\n"+json.dumps(result,indent=2)+"\n```\n",encoding="utf-8")
    return run


def run_phase21_control(root):
    from .tasks21 import representation
    torch.set_num_threads(1)
    settings=json.loads((Path(root)/'configs/phase21.json').read_text())
    output=new_run(root,'phase21-bptt-control-v1',settings); rows=[]
    with ResourceMonitor() as monitor:
        for seed in settings['confirmation_seeds']:
            for task in settings['representation_tasks']:
                model=BPTTControl(Config(inputs=8,outputs=2,hidden=48,fan_in=8,plastic_rows=12,seed=seed))
                x,y=representation(task,10000+seed,512)
                start=time.perf_counter()
                for seq,label in zip(x,y): model.forward(seq,track=True); model.learn(int(label))
                elapsed=time.perf_counter()-start
                tx,ty=representation(task,30000+seed,256)
                correct=[int(np.argmax(model.forward(seq)))==label for seq,label in zip(tx,ty)]
                row={'seed':seed,'task':task,'accuracy':float(np.mean(correct)),'train_seconds':elapsed,'updates':512}
                rows.append(row)
                with (output/'trials.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
                print('BPTT21 '+json.dumps(row),flush=True); monitor.check()
    result={'trials':rows,'resources':monitor.report(),'control_only':True,
            'caveat':'Same sparse shape and training examples; deterministic surrogate-ALIF vs stochastic local network. Updates matched, actual CPU time reported; not evidence of fully matched hardware efficiency.'}
    write_json(output/'result.json',result)
    (output/'report.md').write_text('# Phase2.1 isolated BPTT control\n\n```json\n'+json.dumps(result,indent=2)+'\n```\n')
    return output


class ClassifierLearnabilityControl:
    """Isolated matched-forward P1 diagnostic. Never imported by neural/runtime.

    Main task head retains its existing local delta update; only this control
    sends its task loss backward through all hidden neurons and time steps.
    """
    def __init__(self,config,head,hidden_frozen=False,optimizer_name='adam',row_policy='all'):
        if optimizer_name not in ('adam','sgd') or row_policy not in ('all','rotating'):raise ValueError('Invalid isolated optimizer/coverage')
        self.config=config;self.head=head;self.hidden_frozen=hidden_frozen
        self.optimizer_name=optimizer_name;self.row_policy=row_policy
        base=SpikingCore(config)
        self.parameters=[torch.tensor(getattr(base,n),requires_grad=not hidden_frozen) for n in ('w_in','w_rec')]
        self.indices=torch.tensor(base.indices,dtype=torch.long)
        optimizer=torch.optim.Adam if optimizer_name=='adam' else torch.optim.SGD
        self.optimizer=None if hidden_frozen else optimizer(self.parameters,lr=.002)
        self.updates=0;self.hidden_updates=0;self.last_features=None
        self.last_rows=np.arange(config.hidden);self.last_gradient_norm=None;self.last_clipped_elements=0

    def features(self,sequence,track=False):
        c=self.config;wi,wr=self.parameters
        values=np.asarray(sequence,np.float32)
        single=values.ndim==2
        if single:values=values[None,:,:]
        if values.ndim!=3 or values.shape[2]!=c.inputs:raise ValueError('Invalid control input')
        with torch.set_grad_enabled(track):
            v,a,z,f=[torch.zeros((len(values),c.hidden)) for _ in range(4)]
            for tick in range(values.shape[1]):
                x=torch.as_tensor(values[:,tick]);v=c.leak*v+x@wi.T+(wr[None,:,:]*z[:,self.indices]).sum(2)
                limit=c.threshold+c.adaptation_strength*a;z=ClassifierSurrogate.apply(v,limit)
                v=v-z*limit;a=c.adaptation_decay*a+z;f=c.feature_decay*f+(1-c.feature_decay)*z
            self.last_state=[value[0] if single else value for value in (v,a,z,f)]
            self.last_features=f[0] if single else f
            return self.last_features

    def learn(self,sequence,target):
        f=self.features(sequence,track=not self.hidden_frozen)
        old=[p.detach().clone() for p in self.parameters] if not self.hidden_frozen else []
        if not self.hidden_frozen:
            self.optimizer.zero_grad(set_to_none=True)
            mean=torch.as_tensor(self.head.mean,dtype=torch.float32)
            scale=torch.as_tensor(np.maximum(np.sqrt(self.head.m2/(self.head.count-1)),.05),dtype=torch.float32)
            transformed=((f-mean)/scale).clamp(-5,5)
            logits=torch.as_tensor(self.head.weights)@transformed+torch.as_tensor(self.head.bias)
            loss=torch.nn.functional.cross_entropy(logits[None,:],torch.tensor([target]))
            loss.backward();self.last_gradient_norm=float(torch.nn.utils.clip_grad_norm_(self.parameters,1.))
            self.last_rows=((np.arange(self.config.plastic_rows)+self.updates*self.config.plastic_rows)%self.config.hidden
                if self.row_policy=='rotating' else np.arange(self.config.hidden))
            active=torch.zeros(self.config.hidden,dtype=torch.bool);active[self.last_rows]=True
            old_moments=[]
            if self.row_policy=='rotating':
                for p in self.parameters:
                    p.grad[~active]=0
                    old_moments.append({k:v.detach().clone() for k,v in self.optimizer.state[p].items() if torch.is_tensor(v) and v.shape==p.shape})
            self.optimizer.step();self.last_clipped_elements=0
            with torch.no_grad():
                for index,(p,before) in enumerate(zip(self.parameters,old)):
                    delta=p-before
                    if self.row_policy=='rotating':
                        delta[~active]=0
                        for key,value in self.optimizer.state[p].items():
                            if torch.is_tensor(value) and value.shape==p.shape:
                                previous=old_moments[index].get(key)
                                value[~active]=previous[~active] if previous is not None else 0
                    self.last_clipped_elements+=int((delta.abs()>.025).sum())
                    updated=(before+delta.clamp(-.025,.025)).clamp(-3,3)
                    if self.row_policy=='rotating':updated[~active]=before[~active]
                    p.copy_(updated)
            self.hidden_updates+=1
        # Same pre-update features and local main-head rule as candidate.
        result=self.head.learn(f.detach().numpy(),target);self.updates+=1;return result

    def copied_core(self):
        base=SpikingCore(self.config)
        for name,p in zip(('w_in','w_rec'),self.parameters):getattr(base,name)[:]=p.detach().numpy()
        base.w_out[:]=self.head.weights;base.bias[:]=self.head.bias;base.updates=self.updates
        if hasattr(self,'last_state') and self.last_state[0].ndim==1:
            for name,value in zip(('voltage','adaptation','spikes','features'),self.last_state):getattr(base,name)[:]=value.detach().numpy()
        return base

    def optimizer_bytes(self):
        if self.optimizer is None:return 0
        return sum(v.numel()*v.element_size() for state in self.optimizer.state.values() for v in state.values() if torch.is_tensor(v))


class ClassifierSurrogate(torch.autograd.Function):
    @staticmethod
    def forward(ctx,voltage,limit):
        ctx.save_for_backward(voltage,limit);return (voltage>=limit).float()

    @staticmethod
    def backward(ctx,grad):
        voltage,limit=ctx.saved_tensors
        psi=.3/limit.clamp(min=.001)*(1-((voltage-limit)/limit.clamp(min=.001)).abs()).clamp(min=0)
        return grad*psi,-grad*psi


class ByteBPTTControl:
    """Isolated same-forward byte learner. Never a production/local-learning path."""
    def __init__(self, calibrated_byte_model, hidden_frozen=False):
        from ..neural.input_events import InputEvents
        self.config=calibrated_byte_model.core.config
        self.encoder=InputEvents(256)
        self.parameters=[torch.tensor(getattr(calibrated_byte_model.core,n),requires_grad=(not hidden_frozen or n in ('w_out','bias')))
          for n in ('w_in','w_rec','w_out','bias')]
        self.indices=torch.tensor(calibrated_byte_model.core.indices,dtype=torch.long)
        self.state=[torch.tensor(getattr(calibrated_byte_model.core,n)) for n in ('voltage','adaptation','spikes','features')]
        self.mean=torch.tensor(calibrated_byte_model.head.mean.astype(np.float32))
        self.scale=torch.tensor(np.maximum(np.sqrt(calibrated_byte_model.head.m2/(calibrated_byte_model.head.count-1)),.05).astype(np.float32))
        self.optimizer=torch.optim.Adam([p for p in self.parameters if p.requires_grad],lr=.001)
        self.updates=0;self.byte_steps=0;self.hidden_frozen=hidden_frozen

    def step(self,x):
        c=self.config;wi,wr,wo,b=self.parameters;v,a,z,f=self.state
        current=wi@torch.as_tensor(x,dtype=torch.float32)+(wr*z[self.indices]).sum(dim=1)
        v=c.leak*v+current
        limit=c.threshold+c.adaptation_strength*a;z=Spike.apply(v-limit);v=v-z*limit
        a=c.adaptation_decay*a+z;f=c.feature_decay*f+(1-c.feature_decay)*z
        self.state=[v,a,z,f]

    def prepare(self,messages):
        events=self.encoder.dialogue(messages)
        self.state=[s.detach() for s in self.state]
        for _ in range(12):self.step(np.zeros(454,np.float32))
        context=events[:,256:384].sum(0);context/=max(1.,float(np.linalg.norm(context)))
        for x in events:self.step(x)
        return context

    def byte_step(self,previous,context,position):
        x=np.zeros(454,np.float32)
        if previous>=2:x[previous-2]=.8
        x[256:384]=context*1.6;x[-6:]=[1.,0.,-1. if previous==1 else .2,position/128.,.2*np.sin(position),.2*np.cos(position)]
        for _ in range(3):self.step(x)
        self.byte_steps+=1

    def logits(self,mask):
        f=((self.state[3]-self.mean)/self.scale).clamp(-5,5)
        return (self.parameters[2]@f+self.parameters[3]).masked_fill(~torch.as_tensor(mask),float('-inf'))

    def learn(self,messages,target):
        from ..neural.byte_codec import UTF8Codec,UTF8Cursor
        tokens=UTF8Codec.encode(target);cursor=UTF8Cursor();context=self.prepare(messages);previous=1;losses=[]
        self.optimizer.zero_grad(set_to_none=True)
        for position,token in enumerate(tokens+[0]):
            self.byte_step(previous,context,position)
            losses.append(torch.nn.functional.cross_entropy(self.logits(cursor.mask(257-position))[None,:],torch.tensor([token])))
            if token:cursor.accept(token)
            previous=token
        loss=torch.stack(losses).mean();loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in self.parameters if p.requires_grad],1.)
        self.optimizer.step()
        with torch.no_grad():
            for index,p in enumerate(self.parameters):p.clamp_(-20 if index<2 else -3,20 if index<2 else 3)
        self.updates+=1
        return float(loss.detach())

    def generate(self,messages,max_bytes=128):
        from ..neural.byte_codec import UTF8Codec,UTF8Cursor
        with torch.no_grad():
            cursor=UTF8Cursor();context=self.prepare(messages);previous=1;tokens=[];stopped=False
            for position in range(max_bytes):
                self.byte_step(previous,context,position);token=int(self.logits(cursor.mask(max_bytes-position)).argmax())
                if token==0:stopped=True;break
                cursor.accept(token);tokens.append(token);previous=token
            return {'text':UTF8Codec.decode(tokens),'stopped':stopped,'bytes':len(tokens),'utf8_valid':True}

    def score(self,examples):
        from ..neural.byte_codec import UTF8Codec,UTF8Cursor
        from ..neural.language import tokenize
        saved=[s.detach().clone() for s in self.state];steps=self.byte_steps
        outputs=[];count=0;correct=0;nll=0.
        try:
            with torch.no_grad():
                for row in examples:
                    generated=self.generate(row['messages']);text=generated['text'];tokens=UTF8Codec.encode(row['target'])
                    outputs.append({'id':row['id'],'target':row['target'],'generated':text,'exact':tokenize(text)==tokenize(row['target']),'stopped':generated['stopped']})
                    cursor=UTF8Cursor();context=self.prepare(row['messages']);previous=1
                    for position,token in enumerate(tokens+[0]):
                        self.byte_step(previous,context,position);logits=self.logits(cursor.mask(257-position))
                        nll+=float(torch.nn.functional.cross_entropy(logits[None,:],torch.tensor([token])));correct+=int(logits.argmax()==token);count+=1
                        if token:cursor.accept(token)
                        previous=token
        finally:self.state=saved;self.byte_steps=steps
        return {'exact':sum(r['exact'] for r in outputs)/len(outputs),'byte_nll':nll/count,'teacher_forced_byte_accuracy':correct/count,'target_bytes_including_eos':count,'outputs':outputs}

    def export_arrays(self,path):
        arrays={n:p.detach().numpy() for n,p in zip(('w_in','w_rec','w_out','bias'),self.parameters)}
        arrays.update({n:s.detach().numpy() for n,s in zip(('voltage','adaptation','spikes','features'),self.state)})
        arrays.update(mean=self.mean.numpy(),scale=self.scale.numpy(),indices=self.indices.numpy())
        np.savez(path,metadata=np.array(json.dumps({'format':'isolated-byte-bptt-control-arrays','not_a_scarx_checkpoint':True,'optimizer_resume_included':False,'updates':self.updates})),**arrays)


class WordTapeSurrogate(torch.autograd.Function):
    """CONTROL ONLY: observed NumPy spikes, biased logistic backward surrogate."""
    @staticmethod
    def forward(ctx,voltage,limit,observed):
        raw=(voltage-limit)/.4;p=torch.sigmoid(raw.clamp(-20,20))
        ctx.save_for_backward(p,(raw.abs()<20));return observed.clone()

    @staticmethod
    def backward(ctx,gradient):
        p,active=ctx.saved_tensors;g=gradient*p*(1-p)/.4*active
        return g,-g,None


class WordLearnabilityControl:
    """Isolated BPTT credit signal; unchanged authoritative ALIF/head/block timing.

    Never import into serving. Torch graphs recompute observed trajectories;
    production NumPy state/moments/RNG live in an experimental reference copy.
    """
    def __init__(self,reference):
        from .score_word import ScoreWord
        if not isinstance(reference,ScoreWord) or reference.condition not in ('full','frozen'):
            raise ValueError('Expected isolated full/frozen ScoreWord reference')
        self.reference=reference;self.backward_passes=0;self.surrogate_forward_ticks=0
        self.max_graph_state_error=0.;self.max_graph_feature_error=0.;self.last_gradient_norm=None

    def _record_rollout(self,sequence,uniforms,rows):
        core=self.reference.core;core.reset_state();core.rows[:]=rows;spikes=[]
        for x,u in zip(sequence,uniforms):core.step(x,u,True);spikes.append(core.spikes.copy())
        return core.features.copy(),np.stack(spikes),[getattr(core,n).copy() for n in ('voltage','adaptation','spikes','features')]

    def _gradient(self,sequence,first,second,target):
        core=self.reference.core;c=core.config
        wi,wr=[torch.tensor(getattr(core,n),requires_grad=True) for n in ('w_in','w_rec')]
        indices=torch.as_tensor(core.indices,dtype=torch.long);inputs=torch.as_tensor(sequence)
        observed=torch.as_tensor(np.stack((first[1],second[1])))
        v,a,z,f=[torch.zeros((2,c.hidden),dtype=torch.float32) for _ in range(4)]
        for tick in range(len(sequence)):
            v=c.leak*v+(wi@inputs[tick])[None,:]+(wr[None,:,:]*z[:,indices]).sum(2)
            limit=c.threshold+c.adaptation_strength*a;z=WordTapeSurrogate.apply(v,limit,observed[:,tick])
            v=v-z*limit;a=c.adaptation_decay*a+z;f=c.feature_decay*f+(1-c.feature_decay)*z
        errors=[]
        for n,state in enumerate((v,a,z,f)):
            expected=np.stack((first[2][n],second[2][n]));actual=state.detach().numpy()
            error=float(np.max(np.abs(actual-expected)));errors.append(error)
            np.testing.assert_allclose(actual,expected,atol=1e-6 if n==3 else 3e-5,rtol=0)
        self.max_graph_state_error=max(self.max_graph_state_error,*errors[:3]);self.max_graph_feature_error=max(self.max_graph_feature_error,errors[3])
        transformed=((f-torch.as_tensor(core.head_mean))/torch.as_tensor(core.head_scale)).clamp(-5,5)
        logits=transformed@torch.as_tensor(core.w_out).T+torch.as_tensor(core.bias)
        logits=torch.where(torch.as_tensor(core.output_mask)[None,:],logits,torch.tensor(-float('inf')))
        loss=torch.nn.functional.cross_entropy(logits,torch.tensor([target,target],dtype=torch.long));loss.backward()
        gradients=[-p.grad.detach().numpy().copy() for p in (wi,wr)]
        if not all(np.isfinite(g).all() for g in gradients):raise ValueError('Nonfinite isolated word gradient')
        self.last_gradient_norm=float(np.sqrt(sum(float(np.sum(g.astype(np.float64)**2)) for g in gradients)))
        self.backward_passes+=1;self.surrogate_forward_ticks+=2*len(sequence)
        return [g[core.rows] for g in gradients]

    def learn_example(self,sequence,target):
        core=self.reference.core;c=core.config;x=np.asarray(sequence,np.float32)
        if not core.calibrated or type(target) is not int or not 0<=target<c.outputs or not core.output_mask[target]:raise ValueError('Control target/calibration')
        if x.ndim!=2 or not 1<=len(x)<=128 or x.shape[1]!=c.inputs or not np.isfinite(x).all():raise ValueError('Control input')
        if core.frozen:return core.learn_example(x,target)
        features=[];losses=[];blocks=(c.hidden+c.plastic_rows-1)//c.plastic_rows
        for block in range(blocks):
            rows=(np.arange(c.plastic_rows)+(core.updates+block)*c.plastic_rows)%c.hidden
            u=core.rng.random((len(x),c.hidden));v=core.rng.random(u.shape)
            first=self._record_rollout(x,u,rows);second=self._record_rollout(x,v,rows)
            rewards=np.array([np.log(max(float(core.probabilities(a[0])[target]),1e-12)) for a in (first,second)])
            core.update_hidden(self._gradient(x,first,second,target))
            core.reward_mean=.95*core.reward_mean+.05*float(rewards.mean())
            features.extend((first[0],second[0]));losses.extend((-rewards).tolist())
        f=np.mean(features,axis=0).astype(np.float32);p=core.probabilities(f);error=-p.copy();error[target]+=1
        core.w_out+=np.clip(c.readout_lr*error[:,None]*core.transform(f)[None,:],-c.update_clip,c.update_clip)
        core.bias+=np.clip(c.readout_lr*error,-c.update_clip,c.update_clip)
        np.clip(core.w_out,-c.weight_clip,c.weight_clip,out=core.w_out);np.clip(core.bias,-c.weight_clip,c.weight_clip,out=core.bias)
        core.updates+=1
        return {'loss':-float(np.log(max(float(p[target]),1e-12))),'trajectory_loss':float(np.mean(losses)),
            'rollouts':blocks*2,'ticks':len(x)*blocks*2,'hidden_updates':blocks}

    def learn(self,prompt,target):
        from ..neural.language import tokenize
        reference=self.reference;tokens=tokenize(target)
        if not tokens or len(tokens)>23 or any(t not in reference.ids for t in tokens):raise ValueError('Control target vocabulary')
        prefix=[];losses=[];trajectory=[]
        for position,token in enumerate([reference.ids[t] for t in tokens]+[0]):
            reference.mask(position);r=self.learn_example(reference.sequence(prompt,prefix),int(token))
            losses.append(r['loss']);trajectory.append(r['trajectory_loss'])
            if token:prefix.append(int(token))
        reference.responses_seen+=1
        return {'loss':float(np.mean(losses)),'trajectory_loss':float(np.mean(trajectory)),'tokens':len(losses)}

    def generate(self,prompt,seed=None):
        result=self.reference.generate(prompt,seed);result['scope']='ISOLATED BPTT/readout control; never SCAR-X main'
        return result

    def dumps(self):
        raw=self.reference.dumps();meta={'format':'scarx-isolated-word-bptt-control-v1','control_only':True,
            'reference_sha256':hashlib.sha256(raw).hexdigest(),'backward_passes':self.backward_passes,
            'surrogate_forward_ticks':self.surrogate_forward_ticks,'max_graph_state_error':self.max_graph_state_error,
            'max_graph_feature_error':self.max_graph_feature_error,'last_gradient_norm':self.last_gradient_norm}
        out=io.BytesIO()
        with zipfile.ZipFile(out,'w') as z:z.writestr(zipfile.ZipInfo('reference.word'),raw);z.writestr(zipfile.ZipInfo('control.json'),json.dumps(meta))
        return out.getvalue()

    @classmethod
    def loads(cls,payload):
        from .score_word import ScoreWord
        if len(payload)>22000000:raise ValueError('Oversized isolated control')
        with zipfile.ZipFile(io.BytesIO(payload)) as z:
            if set(z.namelist())!={'reference.word','control.json'} or len(z.namelist())!=2 or sum(i.file_size for i in z.infolist())>22000000 or z.getinfo('control.json').file_size>65536:raise ValueError('Isolated control members')
            raw=z.read('reference.word');meta=json.loads(z.read('control.json'))
        if meta['format']!='scarx-isolated-word-bptt-control-v1' or meta.get('control_only') is not True or hashlib.sha256(raw).hexdigest()!=meta['reference_sha256']:raise ValueError('Control provenance/checksum')
        m=cls(ScoreWord.loads(raw))
        for key in ('backward_passes','surrogate_forward_ticks'):
            if type(meta[key]) is not int or meta[key]<0:raise ValueError('Control counter')
            setattr(m,key,meta[key])
        for key in ('max_graph_state_error','max_graph_feature_error'):
            value=meta[key]
            if type(value) not in (int,float) or not np.isfinite(value) or value<0:raise ValueError('Control error statistic')
            setattr(m,key,value)
        value=meta['last_gradient_norm']
        if value is not None and (type(value) not in (int,float) or not np.isfinite(value) or value<0):raise ValueError('Control gradient norm')
        m.last_gradient_norm=value;return m


class VisualWordLearnabilityControl:
    """CONTROL ONLY: NumPy ALIF forward, isolated response-level surrogate BPTT.

    Output head remains the unchanged local rule. No production imports or
    export compatibility: this wrapper is intentionally not a main checkpoint.
    """
    state_names=('voltage','adaptation','spikes','features')

    def __init__(self,reference):
        from .visual_words import VisualWords
        if not isinstance(reference,VisualWords) or reference.condition!='full':raise ValueError('Expected FULL visualword research reference')
        self.reference=reference
        for name in ('in','rec'):
            for prefix in ('m_','v_'):setattr(self,prefix+name,np.zeros_like(getattr(reference.core,'w_'+name)))
        self.optimizer_steps=0;self.backward_passes=0;self.surrogate_forward_ticks=0
        self.max_graph_state_error=0.;self.max_graph_feature_error=0.;self.max_graph_loss_error=0.
        self.last_gradient_norm=None;self.zero_gradient_responses=0

    def _observe(self,question,image,target):
        m=self.reference;tokens=m.target_tokens(target)
        if not m.head.frozen:raise ValueError('Calibrate on unlabeled TRAINinputs first')
        core=m.core;inputs=[];states=[];words=[];losses=[];original=core.step;had_override='step' in core.__dict__
        def record(x,track=True):
            inputs.append(np.asarray(x,np.float32).copy());out=original(x,track)
            states.append([getattr(core,n).copy() for n in self.state_names]);return out
        core.step=record
        try:
            m.prepare(question,image);previous=1
            for position,token in enumerate(tokens):
                feature=m.word_feature(previous,position)
                words.append({'end_tick':len(inputs),'feature':feature.copy(),'mask':m.output_mask.copy(),
                    'weights':m.head.weights.copy(),'bias':m.head.bias.copy(),'target':token})
                losses.append(m.head.learn(feature,token,m.output_mask));core.updates+=1;previous=token
            m.experiences+=1
        finally:
            if had_override:core.step=original
            else:del core.step
        if len(inputs)!=33+3*len(tokens):raise ValueError('Fixed visualword tape budget changed')
        return np.stack(inputs),np.asarray(states),words,losses

    def _gradient(self,inputs,states,words,losses):
        m=self.reference;c=m.core.config
        wi=torch.tensor(m.core.w_in,requires_grad=True);wr=torch.tensor(m.core.w_rec,requires_grad=True)
        indices=torch.as_tensor(m.core.indices,dtype=torch.long);v,a,z,f=[torch.zeros(c.hidden) for _ in range(4)]
        observed=torch.as_tensor(states[:,:, :]);features=[];loss_terms=[];word_by_tick={w['end_tick']:i for i,w in enumerate(words)}
        mean=torch.as_tensor(m.head.mean.astype(np.float32));scale=torch.as_tensor(np.maximum(np.sqrt(m.head.m2/(m.head.count-1)),.05).astype(np.float32))
        state_error=0.;feature_error=0.;loss_error=0.;cache=None
        for tick,x in enumerate(inputs):
            current=wi@torch.as_tensor(x)+(wr*z[indices]).sum(1);v=c.leak*v+current
            limit=c.threshold+c.adaptation_strength*a;z=WordTapeSurrogate.apply(v,limit,observed[tick,2])
            v=v-z*limit;a=c.adaptation_decay*a+z;f=c.feature_decay*f+(1-c.feature_decay)*z
            for k,s in enumerate((v,a,z,f)):
                error=float(np.max(np.abs(s.detach().numpy()-states[tick,k])))
                if k==3:feature_error=max(feature_error,error)
                else:state_error=max(state_error,error)
            if 16<=tick<=31:features.append(f)
            if tick==32:
                cache=torch.stack(features).mean(0)
                feature_error=max(feature_error,float(np.max(np.abs(cache.detach().numpy()-m.visual_feature))))
            if tick+1 in word_by_tick:
                index=word_by_tick[tick+1];w=words[index];pooled=.5*(cache+f)
                feature_error=max(feature_error,float(np.max(np.abs(pooled.detach().numpy()-w['feature']))))
                transformed=((pooled-mean)/scale).clamp(-5,5)
                logits=torch.as_tensor(w['weights'])@transformed+torch.as_tensor(w['bias'])
                logits=torch.where(torch.as_tensor(w['mask']),logits,torch.tensor(-float('inf')))
                term=-torch.log(torch.softmax(logits,0)[w['target']].clamp(min=1e-12));loss_terms.append(term)
                loss_error=max(loss_error,abs(float(term.detach())-losses[index]))
        if state_error>1e-4 or feature_error>1e-6 or loss_error>1e-4:raise ValueError(f'Visualcontrol graph parity: state={state_error}, feature={feature_error}, loss={loss_error}')
        torch.stack(loss_terms).mean().backward()
        gradients=[p.grad.detach().numpy().copy() for p in (wi,wr)]
        if not all(np.isfinite(g).all() for g in gradients):raise ValueError('Nonfinite isolated visualword gradient')
        norm=float(np.sqrt(sum(float(np.sum(g.astype(np.float64)**2)) for g in gradients)))
        self.last_gradient_norm=norm;self.zero_gradient_responses+=int(norm==0)
        if norm>1:gradients=[g/norm for g in gradients]
        self.max_graph_state_error=max(self.max_graph_state_error,state_error);self.max_graph_feature_error=max(self.max_graph_feature_error,feature_error)
        self.max_graph_loss_error=max(self.max_graph_loss_error,loss_error);self.backward_passes+=1;self.surrogate_forward_ticks+=len(inputs)
        return gradients

    def learn(self,question,image,target,*,train_hidden=True):
        if type(train_hidden) is not bool:raise ValueError('Explicit hiddenupdate policy')
        inputs,states,words,losses=self._observe(question,image,target)
        if train_hidden:
            gradients=self._gradient(inputs,states,words,losses);self.optimizer_steps+=1
            for name,g in zip(('in','rec'),gradients):
                moment=getattr(self,'m_'+name);variance=getattr(self,'v_'+name)
                moment*=.9;moment+=.1*g;variance*=.999;variance+=.001*g*g
                update=-.001*(moment/(1-.9**self.optimizer_steps))/(np.sqrt(variance/(1-.999**self.optimizer_steps))+1e-8)
                weights=getattr(self.reference.core,'w_'+name);weights+=np.clip(update,-.002,.002);np.clip(weights,-3,3,out=weights)
        return {'loss':float(np.mean(losses)),'tokens':len(losses)}

    def generate(self,question,image,**kwargs):
        r=self.reference.generate(question,image,**kwargs);r['scope']='ISOLATED visualword surrogate-BPTT CONTROL; never SCAR-X main';return r

    def dumps(self):
        raw=self.reference.dumps();state=io.BytesIO();np.savez(state,**{n:getattr(self,n) for n in ('m_in','v_in','m_rec','v_rec')});state=state.getvalue()
        meta={'format':'scarx-isolated-visual-word-bptt-control-v1','control_only':True,'reference_sha256':hashlib.sha256(raw).hexdigest(),
            'optimizer_sha256':hashlib.sha256(state).hexdigest(),**{n:getattr(self,n) for n in ('optimizer_steps','backward_passes','surrogate_forward_ticks',
            'max_graph_state_error','max_graph_feature_error','max_graph_loss_error','last_gradient_norm','zero_gradient_responses')}}
        meta['metadata_sha256']=hashlib.sha256(json.dumps(meta,sort_keys=True).encode()).hexdigest();out=io.BytesIO()
        with zipfile.ZipFile(out,'w') as z:
            z.writestr(zipfile.ZipInfo('reference.visualword'),raw);z.writestr(zipfile.ZipInfo('optimizer.npz'),state);z.writestr(zipfile.ZipInfo('control.json'),json.dumps(meta))
        return out.getvalue()

    @classmethod
    def loads(cls,payload):
        from .visual_words import VisualWords
        if len(payload)>4_000_000:raise ValueError('Oversized visualcontrol')
        with zipfile.ZipFile(io.BytesIO(payload)) as z:
            if set(z.namelist())!={'reference.visualword','optimizer.npz','control.json'} or len(z.namelist())!=3 or sum(i.file_size for i in z.infolist())>4_000_000 or z.getinfo('control.json').file_size>65536:raise ValueError('Visualcontrol members')
            raw=z.read('reference.visualword');state=z.read('optimizer.npz');meta=json.loads(z.read('control.json'))
        digest=meta.pop('metadata_sha256')
        if meta['format']!='scarx-isolated-visual-word-bptt-control-v1' or meta['control_only'] is not True or hashlib.sha256(raw).hexdigest()!=meta['reference_sha256'] or hashlib.sha256(state).hexdigest()!=meta['optimizer_sha256'] or hashlib.sha256(json.dumps(meta,sort_keys=True).encode()).hexdigest()!=digest:raise ValueError('Visualcontrol checksum/provenance')
        m=cls(VisualWords.loads(raw))
        with zipfile.ZipFile(io.BytesIO(state)) as z:
            if len(z.namelist())!=4 or len(set(z.namelist()))!=4 or sum(i.file_size for i in z.infolist())>400000:raise ValueError('Visualcontrol optimizerbound')
        with np.load(io.BytesIO(state),allow_pickle=False) as z:
            if set(z.files)!={'m_in','v_in','m_rec','v_rec'}:raise ValueError('Visualcontrol optimizerfields')
            for name in z.files:
                a=z[name];template=getattr(m,name)
                if a.shape!=template.shape or a.dtype!=template.dtype or not np.isfinite(a).all() or name.startswith('v_') and np.any(a<0):raise ValueError('Visualcontrol optimizerarray')
                template[:]=a
        for name in ('optimizer_steps','backward_passes','surrogate_forward_ticks','zero_gradient_responses'):
            if type(meta[name]) is not int or meta[name]<0:raise ValueError('Visualcontrol counter')
            setattr(m,name,meta[name])
        if m.optimizer_steps!=m.backward_passes or m.optimizer_steps>m.reference.experiences or m.zero_gradient_responses>m.optimizer_steps:raise ValueError('Visualcontrol counterconsistency')
        for name in ('max_graph_state_error','max_graph_feature_error','max_graph_loss_error','last_gradient_norm'):
            value=meta[name]
            if value is None and name=='last_gradient_norm':continue
            if type(value) not in (int,float) or not np.isfinite(value) or value<0:raise ValueError('Visualcontrol statistic')
            setattr(m,name,value)
        return m


class EligibilityLearnabilityControl:
    """Proposal30 CONTROL ONLY: full temporal gradients, local head unchanged."""
    def __init__(self,reference):
        from .readout_eligibility import ReadoutEligibility
        if not isinstance(reference,ReadoutEligibility) or reference.condition not in ('correct','frozen'):raise ValueError('Matched eligibility research reference required')
        self.reference=reference;self.backward_passes=0;self.surrogate_forward_ticks=0;self.max_graph_state_error=0.;self.max_graph_feature_error=0.

    def _direction(self,sequence,spikes,target):
        m=self.reference;c=m.core.config;wi=torch.tensor(m.core.w_in,requires_grad=True);wr=torch.tensor(m.core.w_rec,requires_grad=True)
        indices=torch.as_tensor(m.core.indices,dtype=torch.long);v,a,z,f=[torch.zeros(c.hidden) for _ in range(4)]
        for x,observed in zip(sequence,spikes):
            current=wi@torch.as_tensor(x)+(wr*z[indices]).sum(1);v=c.leak*v+current;theta=c.threshold+c.adaptation_strength*a
            z=WordTapeSurrogate.apply(v,theta,torch.as_tensor(observed));v=v-z*theta;a=c.adaptation_decay*a+z;f=c.feature_decay*f+(1-c.feature_decay)*z
        errors=[float(np.max(np.abs(s.detach().numpy()-getattr(m.core,n)))) for s,n in zip((v,a,z,f),('voltage','adaptation','spikes','features'))]
        if max(errors[:3])>1e-4 or errors[3]>1e-6:raise ValueError('Eligibility BPTT forward parity')
        self.max_graph_state_error=max(self.max_graph_state_error,*errors[:3]);self.max_graph_feature_error=max(self.max_graph_feature_error,errors[3])
        scale=torch.as_tensor(np.maximum(np.sqrt(m.head.m2/(m.head.count-1)),.05).astype(np.float32));raw=(f-torch.as_tensor(m.head.mean.astype(np.float32)))/scale
        # Same strictly-interior clipping sensitivity as candidate at endpoints.
        transformed=raw.detach().clamp(-5,5)+(raw-raw.detach())*(raw.detach().abs()<5)
        logits=torch.as_tensor(m.head.weights)@transformed+torch.as_tensor(m.head.bias);logits=torch.where(torch.as_tensor(m.output_mask),logits,torch.tensor(-float('inf')))
        loss=torch.logsumexp(logits,0)-logits[target];loss.backward()
        direction=np.concatenate([-p.grad.detach().numpy().copy() for p in (wi,wr)],axis=1)
        if not np.isfinite(direction).all():raise ValueError('Nonfinite matched BPTT direction')
        self.backward_passes+=1;self.surrogate_forward_ticks+=len(sequence);return direction

    def learn(self,sequence,target):
        m=self.reference
        if m.condition=='frozen':return m.learn(sequence,target)
        x=m._sequence(sequence,m.core.config.inputs)
        if not m.head.frozen or type(target) is not int or not 0<=target<m.core.config.outputs:raise ValueError('Control supervision')
        snapshot=__import__('copy').deepcopy(self.__dict__)
        try:
            m.begin(False);spikes=[]
            for row in x:m.step(row);spikes.append(m.core.spikes.copy())
            direction=self._direction(x,spikes,target);loss=m.accumulate(target);m.bank[:]=direction;m.finish();return loss
        except BaseException:
            self.__dict__.clear();self.__dict__.update(snapshot);raise

    def features_many(self,sequences,reset_each_tick=False):return self.reference.features_many(sequences,reset_each_tick)
    def predict(self,sequence):return self.reference.predict(sequence)

    def dumps(self):
        raw=self.reference.dumps();meta={'format':'scarx-eligibility-bptt-control-v1','control_only':True,'reference_sha256':hashlib.sha256(raw).hexdigest(),
            **{n:getattr(self,n) for n in ('backward_passes','surrogate_forward_ticks','max_graph_state_error','max_graph_feature_error')}}
        meta['metadata_sha256']=hashlib.sha256(json.dumps(meta,sort_keys=True).encode()).hexdigest();out=io.BytesIO()
        with zipfile.ZipFile(out,'w') as z:z.writestr(zipfile.ZipInfo('reference.eligibility'),raw);z.writestr(zipfile.ZipInfo('control.json'),json.dumps(meta))
        return out.getvalue()

    @classmethod
    def loads(cls,payload):
        from .readout_eligibility import ReadoutEligibility
        if len(payload)>5_000_000:raise ValueError('Matched eligibility control bound')
        with zipfile.ZipFile(io.BytesIO(payload)) as z:
            if set(z.namelist())!={'reference.eligibility','control.json'} or len(z.namelist())!=2 or sum(i.file_size for i in z.infolist())>5_000_000 or z.getinfo('control.json').file_size>65536:raise ValueError('Matched eligibility control members')
            raw=z.read('reference.eligibility');meta=json.loads(z.read('control.json'))
        checksum=meta.pop('metadata_sha256')
        if meta.get('format')!='scarx-eligibility-bptt-control-v1' or meta.get('control_only') is not True or hashlib.sha256(raw).hexdigest()!=meta['reference_sha256'] or hashlib.sha256(json.dumps(meta,sort_keys=True).encode()).hexdigest()!=checksum:raise ValueError('Matched eligibility control checksum')
        m=cls(ReadoutEligibility.loads(raw))
        for name in ('backward_passes','surrogate_forward_ticks'):
            if type(meta[name]) is not int or meta[name]<0:raise ValueError('Matched eligibility counter')
            setattr(m,name,meta[name])
        for name in ('max_graph_state_error','max_graph_feature_error'):
            v=meta[name]
            if type(v) not in (int,float) or not np.isfinite(v) or v<0:raise ValueError('Matched eligibility parity statistic')
            setattr(m,name,v)
        if m.backward_passes!=m.reference.hidden_updates:raise ValueError('Matched eligibility control update consistency')
        return m


class DirectWordLearnabilityControl:
    """Isolated mean-word BPTT with proposal30 wordforward/localhead/Adam."""
    def __init__(self,reference):
        from .direct_words import DirectWords
        if not isinstance(reference,DirectWords) or reference.learner.condition not in ('correct','frozen'):raise ValueError('Directword researchreference')
        self.reference=reference;self.backward_passes=0;self.surrogate_forward_ticks=0;self.max_graph_state_error=0.;self.max_graph_feature_error=0.;self.max_graph_loss_error=0.

    def _direction(self,inputs,states,words,losses):
        m=self.reference;c=m.core.config;wi=torch.tensor(m.core.w_in,requires_grad=True);wr=torch.tensor(m.core.w_rec,requires_grad=True)
        indices=torch.as_tensor(m.core.indices,dtype=torch.long);v,a,z,f=[torch.zeros(c.hidden) for _ in range(4)];terms=[];by_tick={w['tick']:i for i,w in enumerate(words)}
        scale=torch.as_tensor(np.maximum(np.sqrt(m.head.m2/(m.head.count-1)),.05).astype(np.float32));mean=torch.as_tensor(m.head.mean.astype(np.float32))
        state_error=feature_error=loss_error=0.
        for tick,x in enumerate(inputs,1):
            v=c.leak*v+wi@torch.as_tensor(x)+(wr*z[indices]).sum(1);theta=c.threshold+c.adaptation_strength*a
            z=WordTapeSurrogate.apply(v,theta,torch.as_tensor(states[tick-1][2]));v=v-z*theta;a=c.adaptation_decay*a+z;f=c.feature_decay*f+(1-c.feature_decay)*z
            for index,s in enumerate((v,a,z,f)):
                error=float(np.max(np.abs(s.detach().numpy()-states[tick-1][index])))
                if index==3:feature_error=max(feature_error,error)
                else:state_error=max(state_error,error)
            if tick in by_tick:
                i=by_tick[tick];w=words[i];raw=(f-mean)/scale;transformed=raw.detach().clamp(-5,5)+(raw-raw.detach())*(raw.detach().abs()<5)
                logits=torch.as_tensor(w['weights'])@transformed+torch.as_tensor(w['bias']);logits=torch.where(torch.as_tensor(w['mask']),logits,torch.tensor(-float('inf')))
                loss=-torch.log(torch.softmax(logits,0)[w['target']].clamp(min=1e-12));terms.append(loss);loss_error=max(loss_error,abs(float(loss.detach())-losses[i]))
        if state_error>1e-4 or feature_error>1e-6 or loss_error>1e-4:raise ValueError('Directword BPTT forward/loss parity')
        torch.stack(terms).mean().backward();direction=np.concatenate([-p.grad.detach().numpy().copy() for p in (wi,wr)],axis=1)
        if not np.isfinite(direction).all():raise ValueError('Nonfinite directword BPTT')
        self.backward_passes+=1;self.surrogate_forward_ticks+=len(inputs);self.max_graph_state_error=max(self.max_graph_state_error,state_error)
        self.max_graph_feature_error=max(self.max_graph_feature_error,feature_error);self.max_graph_loss_error=max(self.max_graph_loss_error,loss_error)
        return direction

    def learn(self,prompt,target):
        m=self.reference
        if m.learner.condition=='frozen':return m.learn(prompt,target)
        tokens=m.target_tokens(target)
        if not m.head.frozen:raise ValueError('Calibratefirst')
        snapshot=__import__('copy').deepcopy(self.__dict__);inputs=[];states=[];words=[];losses=[];learner=m.learner;original=learner.step
        def observe(x):
            inputs.append(np.asarray(x,np.float32).copy());out=original(x);states.append([getattr(m.core,n).copy() for n in ('voltage','adaptation','spikes','features')]);return out
        learner.step=observe
        try:
            m.prepare(prompt,False);previous=1
            for i,token in enumerate(tokens):
                m.word(previous,i);words.append({'tick':len(inputs),'weights':m.head.weights.copy(),'bias':m.head.bias.copy(),'mask':learner.output_mask.copy(),'target':token})
                losses.append(learner.accumulate(token));previous=token
            del learner.step
            learner.bank[:]=self._direction(inputs,states,words,losses)*len(tokens);learner.finish();m.responses_seen+=1
            return {'loss':float(np.mean(losses)),'tokens':len(tokens)}
        except BaseException:self.__dict__.clear();self.__dict__.update(snapshot);raise
        finally:
            if 'step' in learner.__dict__:del learner.step

    def generate(self,prompt,**kwargs):
        r=self.reference.generate(prompt,**kwargs);r['scope']='ISOLATED ALIF-BPTT directword CONTROL, not main/locallearning.';return r

    def dumps(self):
        raw=self.reference.dumps();meta={'format':'scarx-direct-word-bptt-control-v1','control_only':True,'reference_sha256':hashlib.sha256(raw).hexdigest(),
            **{n:getattr(self,n) for n in ('backward_passes','surrogate_forward_ticks','max_graph_state_error','max_graph_feature_error','max_graph_loss_error')}}
        meta['metadata_sha256']=hashlib.sha256(json.dumps(meta,sort_keys=True).encode()).hexdigest();out=io.BytesIO()
        with zipfile.ZipFile(out,'w') as z:z.writestr(zipfile.ZipInfo('reference.words'),raw);z.writestr(zipfile.ZipInfo('control.json'),json.dumps(meta))
        return out.getvalue()

    @classmethod
    def loads(cls,payload):
        from .direct_words import DirectWords
        if len(payload)>6_000_000:raise ValueError('Directwordcontrol bound')
        with zipfile.ZipFile(io.BytesIO(payload)) as z:
            if set(z.namelist())!={'reference.words','control.json'} or len(z.namelist())!=2 or sum(i.file_size for i in z.infolist())>6_000_000 or z.getinfo('control.json').file_size>65536:raise ValueError('Directwordcontrol members')
            raw=z.read('reference.words');meta=json.loads(z.read('control.json'))
        checksum=meta.pop('metadata_sha256')
        if meta.get('format')!='scarx-direct-word-bptt-control-v1' or meta.get('control_only') is not True or hashlib.sha256(raw).hexdigest()!=meta['reference_sha256'] or hashlib.sha256(json.dumps(meta,sort_keys=True).encode()).hexdigest()!=checksum:raise ValueError('Directwordcontrol checksum')
        m=cls(DirectWords.loads(raw))
        for name in ('backward_passes','surrogate_forward_ticks'):
            if type(meta[name]) is not int or meta[name]<0:raise ValueError('Directwordcontrol counter')
            setattr(m,name,meta[name])
        if m.backward_passes!=m.reference.learner.hidden_updates:raise ValueError('Directwordcontrol learningcounter mismatch')
        for name in ('max_graph_state_error','max_graph_feature_error','max_graph_loss_error'):
            value=meta[name]
            if type(value) not in (int,float) or not np.isfinite(value) or value<0:raise ValueError('Directwordcontrol statistic')
            setattr(m,name,value)
        return m


class DirectVisualLearnabilityControl:
    """Only global/temporal CONTROL, identical37forward/head/synapseAdam."""
    def __init__(self,reference):
        from .direct_visual import DirectVisualWords
        if not isinstance(reference,DirectVisualWords) or reference.method not in ('correct','frozen'):raise ValueError('Directvisual controlreference')
        self.reference=reference;self.backward_passes=0;self.surrogate_forward_ticks=0;self.max_graph_state_error=0.;self.max_graph_feature_error=0.;self.max_graph_loss_error=0.

    def _direction(self,inputs,states,words,losses):
        m=self.reference;c=m.core.config;wi=torch.tensor(m.core.w_in,requires_grad=True);wr=torch.tensor(m.core.w_rec,requires_grad=True)
        indices=torch.as_tensor(m.core.indices,dtype=torch.long);v,a,z,f=[torch.zeros(48) for _ in range(4)];cache=[];pooled_cache=None;terms=[];by_tick={r['tick']:i for i,r in enumerate(words)}
        scale=torch.as_tensor(np.maximum(np.sqrt(m.head.m2/(m.head.count-1)),.05).astype(np.float32));mean=torch.as_tensor(m.head.mean.astype(np.float32))
        se=fe=le=0.
        for tick,x in enumerate(inputs,1):
            v=c.leak*v+wi@torch.as_tensor(x)+(wr*z[indices]).sum(1);theta=c.threshold+c.adaptation_strength*a
            z=WordTapeSurrogate.apply(v,theta,torch.as_tensor(states[tick-1][2]));v=v-z*theta;a=c.adaptation_decay*a+z;f=c.feature_decay*f+(1-c.feature_decay)*z
            for k,s in enumerate((v,a,z,f)):
                e=float(np.max(np.abs(s.detach().numpy()-states[tick-1][k])))
                if k==3:fe=max(fe,e)
                else:se=max(se,e)
            if 17<=tick<=32:cache.append(f)
            if tick==33:pooled_cache=torch.stack(cache).mean(0);fe=max(fe,float(np.max(np.abs(pooled_cache.detach().numpy()-m.visual_feature))))
            if tick in by_tick:
                i=by_tick[tick];w=words[i];pooled=.5*(pooled_cache+f);fe=max(fe,float(np.max(np.abs(pooled.detach().numpy()-w['feature']))))
                raw=(pooled-mean)/scale;transformed=raw.detach().clamp(-5,5)+(raw-raw.detach())*(raw.detach().abs()<5)
                logits=torch.as_tensor(w['weights'])@transformed+torch.as_tensor(w['bias']);logits=torch.where(torch.as_tensor(w['mask']),logits,torch.tensor(-float('inf')))
                term=-torch.log(torch.softmax(logits,0)[w['target']].clamp(min=1e-12));terms.append(term);le=max(le,abs(float(term.detach())-losses[i]))
        if se>1e-4 or fe>1e-6 or le>1e-4:raise ValueError('Directvisualcontrol graphparity')
        torch.stack(terms).mean().backward();direction=np.concatenate([-p.grad.detach().numpy().copy() for p in (wi,wr)],axis=1)
        if not np.isfinite(direction).all():raise ValueError('Nonfinite directvisualcontrol direction')
        self.backward_passes+=1;self.surrogate_forward_ticks+=len(inputs);self.max_graph_state_error=max(self.max_graph_state_error,se);self.max_graph_feature_error=max(self.max_graph_feature_error,fe);self.max_graph_loss_error=max(self.max_graph_loss_error,le)
        return direction

    def learn(self,question,image,target):
        m=self.reference
        if m.method=='frozen':return m.learn(question,image,target)
        tokens=m.target_tokens(target)
        if not m.head.frozen:raise ValueError('Calibrate first')
        snapshot=__import__('copy').deepcopy(self.__dict__);inputs=[];states=[];words=[];losses=[];actual=m._tick
        def record(x):
            inputs.append(np.asarray(x,np.float32).copy());actual(x);states.append([getattr(m.core,n).copy() for n in ('voltage','adaptation','spikes','features')])
        m._tick=record
        try:
            m.training_trace=False;m.prepare(question,image);previous=1
            for i,token in enumerate(tokens):
                f=m.word_feature(previous,i);words.append({'tick':len(inputs),'feature':f.copy(),'weights':m.head.weights.copy(),'bias':m.head.bias.copy(),'mask':m.output_mask.copy(),'target':token})
                losses.append(m.accumulate_word(int(token)));previous=token
            del m._tick;m.learner.bank[:]=self._direction(inputs,states,words,losses)*len(tokens);m.learner.finish();m.experiences+=1
            return {'loss':float(np.mean(losses)),'tokens':len(tokens)}
        except BaseException:self.__dict__.clear();self.__dict__.update(snapshot);raise
        finally:
            if '_tick' in m.__dict__:del m._tick

    def generate(self,question,image,**kwargs):
        r=self.reference.generate(question,image,**kwargs);r['scope']='ISOLATEDdirectvisual ALIF-BPTT CONTROL, never main/locallearning';return r

    def dumps(self):
        raw=self.reference.dumps();meta={'format':'scarx-direct-visual-bptt-control-v1','control_only':True,'reference_sha256':hashlib.sha256(raw).hexdigest(),
            **{n:getattr(self,n) for n in ('backward_passes','surrogate_forward_ticks','max_graph_state_error','max_graph_feature_error','max_graph_loss_error')}}
        meta['metadata_sha256']=hashlib.sha256(json.dumps(meta,sort_keys=True).encode()).hexdigest();out=io.BytesIO()
        with zipfile.ZipFile(out,'w') as z:z.writestr(zipfile.ZipInfo('reference.visual'),raw);z.writestr(zipfile.ZipInfo('control.json'),json.dumps(meta))
        return out.getvalue()

    @classmethod
    def loads(cls,payload):
        from .direct_visual import DirectVisualWords
        if len(payload)>7_000_000:raise ValueError('Directvisualcontrol bound')
        with zipfile.ZipFile(io.BytesIO(payload)) as z:
            if set(z.namelist())!={'reference.visual','control.json'} or len(z.namelist())!=2 or sum(i.file_size for i in z.infolist())>7_000_000 or z.getinfo('control.json').file_size>65536:raise ValueError('Directvisualcontrol members')
            raw=z.read('reference.visual');meta=json.loads(z.read('control.json'))
        checksum=meta.pop('metadata_sha256')
        if meta.get('format')!='scarx-direct-visual-bptt-control-v1' or meta.get('control_only') is not True or hashlib.sha256(raw).hexdigest()!=meta['reference_sha256'] or hashlib.sha256(json.dumps(meta,sort_keys=True).encode()).hexdigest()!=checksum:raise ValueError('Directvisualcontrol checksum')
        m=cls(DirectVisualWords.loads(raw))
        for name in ('backward_passes','surrogate_forward_ticks'):
            if type(meta[name]) is not int or meta[name]<0:raise ValueError('Directvisualcontrol counter')
            setattr(m,name,meta[name])
        if m.backward_passes!=m.reference.learner.hidden_updates:raise ValueError('Directvisualcontrol counterconsistency')
        for name in ('max_graph_state_error','max_graph_feature_error','max_graph_loss_error'):
            v=meta[name]
            if type(v) not in (int,float) or not np.isfinite(v) or v<0:raise ValueError('Directvisualcontrol statistic')
            setattr(m,name,v)
        return m


if __name__ == "__main__":
    import sys
    print(run_phase21_control(Path.cwd()) if '--phase21' in sys.argv else run_control(Path.cwd()))
