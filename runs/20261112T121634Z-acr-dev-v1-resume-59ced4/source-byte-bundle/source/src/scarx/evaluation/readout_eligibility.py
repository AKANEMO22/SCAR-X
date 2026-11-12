"""Approved proposal30 RESEARCH ONLY: own-neuron forward sensitivities.

One learned readout error transport is explicitly non-local. No hidden-to-hidden
error transport, reverse-time tape, autograd, Transformer or serving integration.
"""
import copy,hashlib,io,json,zipfile
import numpy as np
from ..neural.core import Config,SpikingCore
from .readout_conditioning import ConditionedReadout


def spike_slope(margin):
    raw=np.asarray(margin)/.4;p=1/(1+np.exp(-np.clip(raw,-20,20)))
    return (p*(1-p)/.4*(np.abs(raw)<20)).astype(np.float32)


class ReadoutEligibility:
    conditions=('correct','shuffled','zero','frozen')
    array_names=('e_v','e_a','e_f','bank','moment','variance','output_mask')

    def __init__(self,config,condition='correct'):
        if condition not in self.conditions or config.neuron!='alif' or config.hidden!=48 or config.fan_in!=8 or not 1<=config.inputs<=454 or not 2<=config.outputs<=266:raise ValueError('Bounded ALIF research configuration')
        config.validate();self.core=SpikingCore(config);self.head=ConditionedReadout(self.core.w_out,self.core.bias,'normalized')
        self.core.w_out=self.head.weights;self.core.bias=self.head.bias;self.condition=condition
        for name in self.array_names[:-1]:setattr(self,name,np.zeros((config.hidden,config.inputs+config.fan_in),np.float32))
        self.output_mask=np.ones(config.outputs,bool);self.signal_rng=np.random.default_rng(config.seed+35100000)
        self.total_ticks=0;self.steps_in_segment=0;self.endpoints=0;self.hidden_updates=0;self.segments_learned=0
        self.segment_active=False;self.track=False;self.last_direction_norm=None

    def begin(self,track=True):
        if type(track) is not bool or self.segment_active and self.endpoints:raise ValueError('Finish/rollback pending supervised segment first')
        self.core.reset_state()
        for n in ('e_v','e_a','e_f','bank'):getattr(self,n).fill(0)
        self.steps_in_segment=0;self.endpoints=0;self.segment_active=True;self.track=track and self.condition!='frozen'

    def step(self,x):
        c=self.core.config;x=np.asarray(x,np.float32)
        if not self.segment_active or self.steps_in_segment>=256 or x.shape!=(c.inputs,) or not np.isfinite(x).all():raise ValueError('Active bounded finite input segment required')
        if self.track:
            old_z=self.core.spikes;theta=c.threshold+c.adaptation_strength*self.core.adaptation
            current=self.core.w_in@x+np.sum(self.core.w_rec*old_z[self.core.indices],axis=1)
            u=c.leak*self.core.voltage+current;psi=spike_slope(u-theta)
            pre=np.concatenate((np.broadcast_to(x,(c.hidden,c.inputs)),old_z[self.core.indices]),axis=1)
            u_e=c.leak*self.e_v+pre;a_old=self.e_a.copy();z_e=psi[:,None]*(u_e-c.adaptation_strength*a_old)
        self.core.step(x,track=False)
        if self.track:
            self.e_v[:]=u_e-theta[:,None]*z_e-self.core.spikes[:,None]*c.adaptation_strength*a_old
            self.e_a[:]=c.adaptation_decay*a_old+z_e
            self.e_f[:]=c.feature_decay*self.e_f+(1-c.feature_decay)*z_e
            if not all(np.isfinite(getattr(self,n)).all() for n in ('e_v','e_a','e_f')):raise ValueError('Nonfinite local sensitivity')
        self.total_ticks+=1;self.steps_in_segment+=1
        return self.core.features

    def accumulate(self,target):
        if not self.segment_active or not self.steps_in_segment or not self.head.frozen or type(target) is not int or not 0<=target<len(self.output_mask) or not self.output_mask[target]:raise ValueError('Calibrated supervised endpoint required')
        feature=self.core.features;p=self.head.probabilities(feature,self.output_mask);error=-p.copy();error[target]+=1
        scale=np.maximum(np.sqrt(self.head.m2/(self.head.count-1)),.05).astype(np.float32)
        raw=(feature-self.head.mean.astype(np.float32))/scale
        signal=(self.head.weights.T@error)*(np.abs(raw)<5)/scale
        permutation=self.signal_rng.permutation(self.core.config.hidden) # matched consumption for every condition
        if self.condition=='shuffled':signal=signal[permutation]
        if self.condition in ('correct','shuffled'):
            self.bank+=signal[:,None]*self.e_f
            if not np.isfinite(self.bank).all():raise ValueError('Nonfinite local direction')
        loss=self.head.learn(feature,target,self.output_mask);self.core.updates+=1;self.endpoints+=1
        return loss

    def _apply_direction(self,direction):
        if direction.shape!=self.bank.shape or not np.isfinite(direction).all():raise ValueError('Bounded finite synapse-local direction')
        self.last_direction_norm=float(np.linalg.norm(direction.astype(np.float64)));self.hidden_updates+=1
        self.moment*=.9;self.moment+=.1*direction;self.variance*=.999;self.variance+=.001*direction*direction
        update=.001*(self.moment/(1-.9**self.hidden_updates))/(np.sqrt(self.variance/(1-.999**self.hidden_updates))+1e-8)
        if not np.isfinite(update).all():raise ValueError('Nonfinite local Adam update')
        d=self.core.config.inputs
        for name,delta in (('w_in',update[:,:d]),('w_rec',update[:,d:])):
            w=getattr(self.core,name);w+=np.clip(delta,-.002,.002);np.clip(w,-3,3,out=w)

    def finish(self):
        if not self.segment_active or not self.endpoints:raise ValueError('No pending endpoint to finish')
        if self.condition in ('correct','shuffled'):self._apply_direction(self.bank/self.endpoints)
        self.segments_learned+=1;self.segment_active=False;self.track=False

    @staticmethod
    def _sequence(sequence,dimensions):
        x=np.asarray(sequence,np.float32)
        if x.ndim!=2 or not 1<=len(x)<=256 or x.shape[1]!=dimensions or not np.isfinite(x).all():raise ValueError('Finite1..256tick sequence')
        return x

    def features(self,sequence,reset_each_tick=False):
        x=self._sequence(sequence,self.core.config.inputs);self.begin(False)
        for row in x:
            if reset_each_tick:self.core.reset_state()
            self.step(row)
        self.segment_active=False;return self.core.features.copy()

    def features_many(self,sequences,reset_each_tick=False):return np.stack([self.features(x,reset_each_tick) for x in sequences])

    def calibrate(self,sequences):
        if len(sequences)!=128 or self.head.frozen:raise ValueError('Exactly128unlabeled calibration inputs')
        while self.head.count<128:self.head.observe(self.features(sequences[self.head.count]))
        self.head.freeze()

    def learn(self,sequence,target):
        x=self._sequence(sequence,self.core.config.inputs)
        if not self.head.frozen or type(target) is not int or not 0<=target<self.core.config.outputs:raise ValueError('Valid calibrated supervision')
        snapshot=copy.deepcopy(self.__dict__)
        try:
            self.begin(True)
            for row in x:self.step(row)
            loss=self.accumulate(target);self.finish();return loss
        except BaseException:
            self.__dict__.clear();self.__dict__.update(snapshot);raise

    def predict(self,sequence):return int(self.head.probabilities(self.features(sequence),self.output_mask).argmax())

    def memory_bytes(self):return self.core.memory_bytes()+self.head.mean.nbytes+self.head.m2.nbytes+sum(getattr(self,n).nbytes for n in self.array_names)

    def dumps(self):
        core=self.core.dumps();head=self.head.dumps();state=io.BytesIO();np.savez(state,**{n:getattr(self,n) for n in self.array_names});state=state.getvalue()
        meta={'format':'scarx-direct-readout-eligibility-v1','condition':self.condition,'signal_rng':self.signal_rng.bit_generator.state,
            **{n:getattr(self,n) for n in ('total_ticks','steps_in_segment','endpoints','hidden_updates','segments_learned','segment_active','track','last_direction_norm')},
            **{n+'_sha256':hashlib.sha256(p).hexdigest() for n,p in (('core',core),('head',head),('state',state))}}
        meta['metadata_sha256']=hashlib.sha256(json.dumps(meta,sort_keys=True).encode()).hexdigest();out=io.BytesIO()
        with zipfile.ZipFile(out,'w') as z:
            for n,p in [('core.npz',core),('head.npz',head),('state.npz',state),('metadata.json',json.dumps(meta))]:z.writestr(zipfile.ZipInfo(n),p)
        raw=out.getvalue()
        if len(raw)>4_000_000:raise MemoryError('Eligibility checkpoint exceeds4MB')
        return raw

    @classmethod
    def loads(cls,payload):
        if len(payload)>4_000_000:raise ValueError('Eligibility checkpoint bound')
        with zipfile.ZipFile(io.BytesIO(payload)) as z:
            if set(z.namelist())!={'core.npz','head.npz','state.npz','metadata.json'} or len(z.namelist())!=4 or sum(i.file_size for i in z.infolist())>4_000_000 or z.getinfo('metadata.json').file_size>65536:raise ValueError('Eligibility checkpoint members')
            blobs={n:z.read(n+'.npz') for n in ('core','head','state')};meta=json.loads(z.read('metadata.json'))
        checksum=meta.pop('metadata_sha256')
        if meta.get('format')!='scarx-direct-readout-eligibility-v1' or hashlib.sha256(json.dumps(meta,sort_keys=True).encode()).hexdigest()!=checksum or any(hashlib.sha256(p).hexdigest()!=meta[n+'_sha256'] for n,p in blobs.items()):raise ValueError('Eligibility checksum')
        for name,raw in blobs.items():
            with zipfile.ZipFile(io.BytesIO(raw)) as z:
                if len(z.namelist())!=len(set(z.namelist())) or sum(i.file_size for i in z.infolist())>2_000_000:raise ValueError('Eligibility inner bound')
                if name in ('core','head') and z.getinfo('metadata.npy').file_size>65536:raise ValueError('Eligibility metadata bound')
        with np.load(io.BytesIO(blobs['core']),allow_pickle=False) as z:
            config=Config(**json.loads(str(z['metadata']))['config'])
        m=cls(config,meta['condition']);loaded=SpikingCore.loads(blobs['core'])
        with np.load(io.BytesIO(blobs['head']),allow_pickle=False) as z:
            if set(z.files)!={'metadata','weights','bias','mean','m2'}:raise ValueError('Eligibility head fields')
            hm=json.loads(str(z['metadata']))
            if hm.get('mode')!='normalized' or type(hm.get('count')) is not int or not 0<=hm['count']<=128 or type(hm.get('updates')) is not int or hm['updates']<0 or type(hm.get('frozen')) is not bool or hm['frozen'] and hm['count']!=128:raise ValueError('Eligibility head policy')
            for name in ('weights','bias','mean','m2'):
                a=z[name];template=getattr(m.head,name)
                if a.shape!=template.shape or a.dtype!=template.dtype or not np.isfinite(a).all() or name=='m2' and np.any(a<0):raise ValueError('Eligibility head array')
        head=ConditionedReadout.loads(blobs['head'])
        if not np.array_equal(loaded.w_out,head.weights) or not np.array_equal(loaded.bias,head.bias) or loaded.updates!=head.updates:raise ValueError('Eligibility head/core mismatch')
        m.core=loaded;m.head=head;m.core.w_out=m.head.weights;m.core.bias=m.head.bias
        with np.load(io.BytesIO(blobs['state']),allow_pickle=False) as z:
            if set(z.files)!=set(m.array_names):raise ValueError('Eligibility state fields')
            for name in m.array_names:
                a=z[name];template=getattr(m,name)
                if a.shape!=template.shape or a.dtype!=template.dtype or not np.isfinite(a).all() or name=='variance' and np.any(a<0):raise ValueError('Eligibility state array')
                template[:]=a
        for name in ('total_ticks','steps_in_segment','endpoints','hidden_updates','segments_learned'):
            value=meta[name]
            if type(value) is not int or value<0:raise ValueError('Eligibility counter')
            setattr(m,name,value)
        if m.steps_in_segment>256 or m.endpoints>256 or m.endpoints>m.steps_in_segment or m.hidden_updates>m.segments_learned or m.segments_learned>m.core.updates or not m.output_mask.any():raise ValueError('Eligibility counter/mask consistency')
        for name in ('segment_active','track'):
            if type(meta[name]) is not bool:raise ValueError('Eligibility flag')
            setattr(m,name,meta[name])
        value=meta['last_direction_norm']
        if value is not None and (type(value) not in (int,float) or not np.isfinite(value) or value<0):raise ValueError('Eligibility direction norm')
        m.last_direction_norm=value;m.signal_rng.bit_generator.state=meta['signal_rng']
        if m.condition in ('zero','frozen') and (m.hidden_updates or np.any(m.moment) or np.any(m.variance)):raise ValueError('Inactive hidden optimizer')
        return m
