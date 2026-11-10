"""Experimental byte decoder around ONE existing deterministic ALIF core.

Not imported by serving. No target words looked up from memory or teacher model.
"""
import io,json,zipfile,hashlib
import numpy as np
from ..neural.core import SpikingCore,Config
from ..neural.input_events import InputEvents
from ..neural.byte_codec import UTF8Codec,UTF8Cursor
from ..neural.language import tokenize
from .readout_conditioning import ConditionedReadout


class ByteBackbone:
    def __init__(self,seed):
        self.core=SpikingCore(Config(inputs=454,outputs=268,hidden=48,fan_in=8,plastic_rows=12,seed=seed,
          leak=.65,threshold=.5,feature_decay=.25,adaptation_strength=.02,readout_lr=.08,hidden_lr=.0001,anchor_strength=0,weight_clip=20))
        self.core.w_in-=.15;self.core.stable_in[:]=self.core.w_in;self.ticks=0

    def tick(self,x=None,track=False):
        if x is None:x=np.zeros(454,np.float32)
        self.core.step(x,track);self.ticks+=1

    def boundary(self):
        for _ in range(12):self.tick()

    def byte_step(self,previous,context,position,track=False):
        x=np.zeros(454,np.float32)
        if previous>=2:x[previous-2]=.8
        elif previous!=1:raise ValueError('Previous token must be BOS or byte')
        x[256:384]=context*1.6
        x[-6:]=[1.,0.,-1. if previous==1 else .2,position/128.,.2*np.sin(position),.2*np.cos(position)]
        for _ in range(3):self.tick(x,track)


class ByteDialogue:
    def __init__(self,seed=6101,condition='byte_raw'):
        if condition not in ('byte_frozen','byte_raw','byte_normalized','byte_normalized_full'):raise ValueError('Unknown byte condition')
        self.backbone=ByteBackbone(seed);self.core=self.backbone.core;self.condition=condition;self.encoder=InputEvents(256)
        self.head=ConditionedReadout(self.core.w_out,self.core.bias,'normalized' if 'normalized' in condition else 'raw')
        self.core.w_out=self.head.weights;self.core.bias=self.head.bias;self.responses_seen=0
        self.order_rng=np.random.default_rng(seed);self.passes_done=0

    def _prepare(self,messages,track=False,image=None):
        events=self.encoder.dialogue(messages)
        if image is not None:
            events=np.concatenate((events,self.encoder.image(image)))
            if len(events)>256:raise ValueError('Input events exceed256')
        self.backbone.boundary()
        if track:
            c=self.core;c.elig_in.fill(0);c.elig_rec.fill(0);c.pre.fill(0);c.post.fill(0)
            c.rows[:]=(np.arange(12)+c.updates*12)%48
        summary=events[:,256:384].sum(0);summary/=max(1.,float(np.linalg.norm(summary)))
        for row in events:self.backbone.tick(row,track)
        return summary

    def calibrate(self,examples):
        if self.head.frozen:raise ValueError('Already calibrated')
        for example in examples:
            context=self._prepare(example['messages']);previous=1
            for position,token in enumerate(UTF8Codec.encode(example['target'])+[0]):
                self.backbone.byte_step(previous,context,position);self.head.observe(self.core.features);previous=token
        self.head.freeze()

    def learn(self,messages,target):
        tokens=UTF8Codec.encode(target)
        if not 1<=len(tokens)<=256:raise ValueError('Target1–256UTF8bytes')
        context=self._prepare(messages,True);previous=1;cursor=UTF8Cursor();loss=0.
        for position,token in enumerate(tokens+[0]):
            self.backbone.byte_step(previous,context,position,True);mask=cursor.mask(257-position)
            if not mask[token]:raise ValueError('Teacher byte violates UTF8 cursor')
            p=self.head.probabilities(self.core.features,mask);loss-=np.log(max(float(p[token]),1e-12))
            if self.condition!='byte_frozen':
                self.head.learn(self.core.features,token,mask)
                reward=float(p[token]);advantage=float(np.clip(reward-self.core.reward_mean,-1,1))
                self.core.reward_mean=.95*self.core.reward_mean+.05*reward
                if self.condition=='byte_normalized_full':
                    c=self.core
                    for name,trace in (('in',c.elig_in),('rec',c.elig_rec)):
                        weights=getattr(c,'w_'+name);weights[c.rows]+=np.clip(c.config.hidden_lr*advantage*trace,-c.config.update_clip,c.config.update_clip)
                        np.clip(weights,-c.config.weight_clip,c.config.weight_clip,out=weights)
                self.core.updates+=1
            if token:cursor.accept(token)
            previous=token
        self.responses_seen+=1
        return float(loss/(len(tokens)+1))

    def generate(self,messages,image=None,max_bytes=128):
        if type(max_bytes) is not int or not 1<=max_bytes<=256:raise ValueError('Budget1–256bytes')
        context=self._prepare(messages,image=image);previous=1;cursor=UTF8Cursor();tokens=[];stopped=False
        for position in range(max_bytes):
            self.backbone.byte_step(previous,context,position)
            token=int(np.argmax(self.head.probabilities(self.core.features,cursor.mask(max_bytes-position))))
            if token==0:stopped=True;break
            cursor.accept(token);tokens.append(token);previous=token
        if cursor.remaining:raise RuntimeError('Half codepoint escaped budget mask')
        text=UTF8Codec.decode(tokens)
        return {'text':text,'byte_tokens':tokens,'utf8_valid':True,'stopped':stopped,'stop_reason':'eos' if stopped else 'length',
          'characters':len(text),'bytes':len(tokens),'syntax_enforced_by_host':True}

    def score(self,examples):
        m=self.loads(self.dumps());outputs=[];nll=0.;count=0;correct=0
        for example in examples:
            generated=m.generate(example['messages']);target=UTF8Codec.encode(example['target'])
            text=generated['text'];outputs.append({'id':example['id'],'target':example['target'],'generated':text,
              'exact':tokenize(text)==tokenize(example['target']),'raw_exact':text==example['target'],
              'nonempty':bool(text.strip()),'repeated_character':any(text.count(c)>max(8,len(text)*.5) for c in set(text) if not c.isspace()),
              'bytes':generated['bytes'],'characters':generated['characters'],'stopped':generated['stopped'],'utf8_valid':generated['utf8_valid']})
            context=m._prepare(example['messages']);cursor=UTF8Cursor();previous=1
            for position,token in enumerate(target+[0]):
                m.backbone.byte_step(previous,context,position);p=m.head.probabilities(m.core.features,cursor.mask(257-position))
                nll-=np.log(max(float(p[token]),1e-12));correct+=int(np.argmax(p)==token);count+=1
                if token:cursor.accept(token)
                previous=token
        return {'exact':sum(x['exact'] for x in outputs)/len(outputs),'raw_exact':sum(x['raw_exact'] for x in outputs)/len(outputs),
          'utf8_valid_rate':sum(x['utf8_valid'] for x in outputs)/len(outputs),'nonempty_rate':sum(x['nonempty'] for x in outputs)/len(outputs),
          'repetition_rate':sum(x['repeated_character'] for x in outputs)/len(outputs),'byte_nll':float(nll/count),
          'teacher_forced_byte_accuracy':correct/count,'oov_bytes':0,'target_bytes_including_eos':count,'outputs':outputs}

    def dumps(self):
        core=self.core.dumps();head=self.head.dumps();meta={'format':'scarx-byte-dialogue-experiment-v1','condition':self.condition,
          'ticks':self.backbone.ticks,'responses_seen':self.responses_seen,'passes_done':self.passes_done,
          'order_rng':self.order_rng.bit_generator.state,'sha256':{'core.npz':hashlib.sha256(core).hexdigest(),'head.npz':hashlib.sha256(head).hexdigest()}}
        stream=io.BytesIO()
        with zipfile.ZipFile(stream,'w') as archive:
            for name,data in (('core.npz',core),('head.npz',head),('meta.json',json.dumps(meta).encode())):archive.writestr(zipfile.ZipInfo(name),data)
        return stream.getvalue()

    @classmethod
    def loads(cls,payload):
        if len(payload)>4000000:raise ValueError('Oversized byte prototype')
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            if len(archive.namelist())!=3 or set(archive.namelist())!={'core.npz','head.npz','meta.json'} or sum(i.file_size for i in archive.infolist())>4000000:raise ValueError('Byte checkpoint members')
            meta=json.loads(archive.read('meta.json'));core=archive.read('core.npz');head=archive.read('head.npz')
        if meta['format']!='scarx-byte-dialogue-experiment-v1':raise ValueError('Wrong byte format')
        for name,data in (('core.npz',core),('head.npz',head)):
            if hashlib.sha256(data).hexdigest()!=meta['sha256'][name]:raise ValueError('Byte checkpoint checksum')
        loaded=SpikingCore.loads(core);c=loaded.config
        if (c.inputs,c.hidden,c.outputs,c.fan_in,c.plastic_rows,c.neuron)!=(454,48,268,8,12,'alif'):raise ValueError('Unexpected byte topology')
        m=cls(c.seed,meta['condition']);m.core=loaded;m.backbone.core=loaded;m.head=ConditionedReadout.loads(head)
        if not np.array_equal(m.core.w_out,m.head.weights) or not np.array_equal(m.core.bias,m.head.bias):raise ValueError('Readout mismatch')
        if not np.isfinite(m.head.mean).all() or not np.isfinite(m.head.m2).all() or m.head.count<0:raise ValueError('Invalid normalization state')
        m.core.w_out=m.head.weights;m.core.bias=m.head.bias
        for key in ('responses_seen','passes_done','ticks'):
            if type(meta[key]) is not int or meta[key]<0:raise ValueError('Invalid byte counters')
        m.responses_seen=meta['responses_seen'];m.passes_done=meta['passes_done'];m.backbone.ticks=meta['ticks'];m.order_rng.bit_generator.state=meta['order_rng']
        return m
