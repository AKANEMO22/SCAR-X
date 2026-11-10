"""One-core held-activation working-memory control, not learned long-term memory."""
import io,json,zipfile,hashlib
import numpy as np
from .byte_dialogue import ByteDialogue
from .readout_conditioning import ConditionedReadout
from ..neural.byte_codec import UTF8Codec,UTF8Cursor
from ..neural.language import tokenize


class WorkingByteDialogue(ByteDialogue):
    def __init__(self,seed=9103,condition='snapshot_raw'):
        if condition not in ('no_snapshot','snapshot_raw','snapshot_normalized'):raise ValueError('Unknown held-state condition')
        super().__init__(seed,'byte_raw');self.condition=condition
        weights=np.concatenate((self.core.w_out.copy(),np.zeros_like(self.core.w_out)),axis=1)
        self.head=ConditionedReadout(weights,self.core.bias,'normalized' if condition=='snapshot_normalized' else 'raw')
        self.core.w_out=self.head.weights[:,:48];self.core.bias=self.head.bias
        self.held=np.zeros(48,np.float32);self.ablation=None;self.override=None

    def _prepare(self,messages,track=False,image=None):
        context=super()._prepare(messages,track,image)
        self.held[:]=self.core.features
        if self.condition=='no_snapshot' or self.ablation=='zero':self.held.fill(0)
        if self.override is not None:self.held[:]=self.override
        return context

    def features(self):return np.concatenate((self.core.features,self.held))

    def calibrate(self,examples):
        if self.head.frozen:raise ValueError('Already calibrated')
        for example in examples:
            context=self._prepare(example['messages']);previous=1
            for position,token in enumerate(UTF8Codec.encode(example['target'])+[0]):
                self.backbone.byte_step(previous,context,position);self.head.observe(self.features());previous=token
        self.head.freeze()

    def learn(self,messages,target):
        tokens=UTF8Codec.encode(target)
        if not 1<=len(tokens)<=256:raise ValueError('Target1–256bytes')
        context=self._prepare(messages,True);previous=1;cursor=UTF8Cursor();loss=0.
        for position,token in enumerate(tokens+[0]):
            self.backbone.byte_step(previous,context,position,True);mask=cursor.mask(257-position)
            if not mask[token]:raise ValueError('Teacher token violatesUTF8')
            loss+=self.head.learn(self.features(),token,mask);self.core.updates+=1
            if token:cursor.accept(token)
            previous=token
        self.responses_seen+=1;return loss/(len(tokens)+1)

    def generate(self,messages,image=None,max_bytes=128):
        if type(max_bytes) is not int or not 1<=max_bytes<=256:raise ValueError('Budget1–256bytes')
        context=self._prepare(messages,image=image);previous=1;cursor=UTF8Cursor();tokens=[];stopped=False
        for position in range(max_bytes):
            self.backbone.byte_step(previous,context,position)
            token=int(np.argmax(self.head.probabilities(self.features(),cursor.mask(max_bytes-position))))
            if token==0:stopped=True;break
            cursor.accept(token);tokens.append(token);previous=token
        if cursor.remaining:raise RuntimeError('Incomplete codepoint')
        text=UTF8Codec.decode(tokens)
        return {'text':text,'byte_tokens':tokens,'utf8_valid':True,'stopped':stopped,'bytes':len(tokens),'characters':len(text),'syntax_enforced_by_host':True}

    def score(self,examples,ablation=None):
        if ablation not in (None,'zero','swap'):raise ValueError('Unknown memory ablation')
        m=self.loads(self.dumps());m.ablation='zero' if ablation=='zero' else None
        outputs=[];nll=0.;correct=0;count=0
        for index,example in enumerate(examples):
            if ablation=='swap':
                other=self.loads(self.dumps());other._prepare(examples[(index+1)%len(examples)]['messages']);m.override=other.held.copy()
            generated=m.generate(example['messages']);text=generated['text'];target=UTF8Codec.encode(example['target'])
            outputs.append({'id':example['id'],'target':example['target'],'generated':text,'exact':tokenize(text)==tokenize(example['target']),
              'raw_exact':text==example['target'],'stopped':generated['stopped'],'utf8_valid':True,'bytes':generated['bytes'],
              'repeated_character':any(text.count(c)>max(8,len(text)*.5) for c in set(text) if not c.isspace())})
            context=m._prepare(example['messages']);cursor=UTF8Cursor();previous=1
            for position,token in enumerate(target+[0]):
                m.backbone.byte_step(previous,context,position);p=m.head.probabilities(m.features(),cursor.mask(257-position))
                nll-=np.log(max(float(p[token]),1e-12));correct+=int(np.argmax(p)==token);count+=1
                if token:cursor.accept(token)
                previous=token
        return {'exact':sum(r['exact'] for r in outputs)/len(outputs),'raw_exact':sum(r['raw_exact'] for r in outputs)/len(outputs),
          'byte_nll':float(nll/count),'teacher_forced_byte_accuracy':correct/count,'repetition_rate':sum(r['repeated_character'] for r in outputs)/len(outputs),
          'utf8_valid_rate':1.,'outputs':outputs}

    def dumps(self):
        core=self.core.dumps();head=self.head.dumps();held=self.held.astype('<f4').tobytes()
        meta={'format':'scarx-working-byte-experiment-v2','condition':self.condition,'ticks':self.backbone.ticks,'responses_seen':self.responses_seen,
          'passes_done':self.passes_done,'order_rng':self.order_rng.bit_generator.state,
          'sha256':{n:hashlib.sha256(a).hexdigest() for n,a in (('core.npz',core),('head.npz',head),('held.bin',held))}}
        stream=io.BytesIO()
        with zipfile.ZipFile(stream,'w') as z:
            for n,a in (('core.npz',core),('head.npz',head),('held.bin',held),('meta.json',json.dumps(meta).encode())):z.writestr(zipfile.ZipInfo(n),a)
        return stream.getvalue()

    @classmethod
    def loads(cls,payload):
        from ..neural.core import SpikingCore
        if len(payload)>4000000:raise ValueError('Oversized working-memory checkpoint')
        with zipfile.ZipFile(io.BytesIO(payload)) as z:
            if len(z.namelist())!=4 or set(z.namelist())!={'core.npz','head.npz','held.bin','meta.json'} or sum(i.file_size for i in z.infolist())>4000000:raise ValueError('Bad working-state members')
            meta=json.loads(z.read('meta.json'));items={n:z.read(n) for n in ('core.npz','head.npz','held.bin')}
        if meta['format']!='scarx-working-byte-experiment-v2':raise ValueError('Wrong working-state format')
        if any(hashlib.sha256(a).hexdigest()!=meta['sha256'][n] for n,a in items.items()):raise ValueError('Working-state checksum')
        core=SpikingCore.loads(items['core.npz']);head=ConditionedReadout.loads(items['head.npz'])
        if (core.config.inputs,core.config.hidden,core.config.outputs,core.config.fan_in,core.config.plastic_rows)!=(454,48,268,8,12) or head.weights.shape!=(268,96):raise ValueError('Wrong working-state topology')
        m=cls(core.config.seed,meta['condition']);m.core=core;m.backbone.core=core;m.head=head
        if not np.array_equal(core.w_out,head.weights[:,:48]) or not np.array_equal(core.bias,head.bias):raise ValueError('Working-state readout mismatch')
        held=np.frombuffer(items['held.bin'],dtype='<f4')
        if held.shape!=(48,) or not np.isfinite(held).all() or not np.isfinite(head.mean).all() or not np.isfinite(head.m2).all():raise ValueError('Invalid working activation/statistics')
        m.held[:]=held;m.core.w_out=head.weights[:,:48];m.core.bias=head.bias
        for name in ('responses_seen','passes_done','ticks'):
            if type(meta[name]) is not int or meta[name]<0:raise ValueError('Invalid working counter')
        m.responses_seen=meta['responses_seen'];m.passes_done=meta['passes_done'];m.backbone.ticks=meta['ticks'];m.order_rng.bit_generator.state=meta['order_rng']
        return m
