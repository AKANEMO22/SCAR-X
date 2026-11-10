"""One fixed-size ALIF recurrent core for words and digits. No pretrained weights.

The two output slices are readouts of the SAME hidden neurons/recurrence. Hidden
learning remains a heuristic scalar-modulated local eligibility rule, not proof
of useful representation learning. Runtime never resets its neuron state.
"""
import hashlib
import io
import json
import zipfile
from dataclasses import replace
import numpy as np
from .core import Config, SpikingCore
from .language import tokenize, render


class UnifiedCore:
    context_size = 128
    settle_ticks = 12

    def __init__(self, vocabulary, seed=17, encoding=None):
        self.vocabulary = list(vocabulary)
        if any(not isinstance(t,str) or not 1<=len(t)<=64 for t in self.vocabulary):raise ValueError('Vocabulary tokens need1–64characters')
        if self.vocabulary[:2] != ['<eos>', '<bos>'] or not 3 <= len(vocabulary) <= 256 or len(set(vocabulary)) != len(vocabulary):
            raise ValueError('Invalid vocabulary')
        self.ids = {word: i for i, word in enumerate(vocabulary)}
        self.encoding = dict(encoding or {'kind':'legacy'})
        if self.encoding.get('kind') not in ('legacy','lexical'):raise ValueError('Unknown encoder')
        if self.encoding.get('surface_normalization','none') not in ('none','terminal-punctuation-v1'):raise ValueError('Unknown surface normalization')
        if not isinstance(self.encoding.get('idf',{}),dict) or len(self.encoding.get('idf',{}))>1024:raise ValueError('Invalid IDF')
        if any(not isinstance(k,str) or not np.isfinite(v) or not 0<v<=20 for k,v in self.encoding.get('idf',{}).items()):raise ValueError('Invalid IDF weights')
        if any(not isinstance(self.encoding.get(k,d),(int,float)) or not np.isfinite(self.encoding.get(k,d)) or not 0<self.encoding.get(k,d)<=8 for k,d in [('context_gain',1.6),('previous_gain',.8)]):raise ValueError('Invalid encoding gain')
        self.image_start = len(vocabulary) + self.context_size
        self.core = SpikingCore(Config(inputs=self.image_start+64+6, hidden=48,
            outputs=len(vocabulary)+10, fan_in=8, plastic_rows=12, seed=seed,
            leak=.65, threshold=.5, feature_decay=.25, adaptation_strength=.02,
            readout_lr=.025, hidden_lr=.0001, anchor_strength=0, weight_clip=20.))
        self.core.w_in -= .15
        self.core.stable_in[:] = self.core.w_in
        self.ticks = 0
        self.on_tick = None
        self.active_count = 48
        self.max_neurons = 128
        self.capacity_rng = np.random.default_rng(seed+421)
        self.last_active = np.zeros(48,np.int64)
        self.capacity_events = []

    @staticmethod
    def lexical_encoding(prompts):
        from collections import Counter
        prompts=list(prompts);counts=Counter(t for p in prompts for t in set(tokenize(p)))
        return {'kind':'lexical','idf':{t:float(np.log((1+len(prompts))/(1+n))+1) for t,n in counts.items()},'context_gain':2.4,'previous_gain':.4}

    def context(self, prompt):
        if not isinstance(prompt, str) or not 1 <= len(prompt.strip()) <= 2048:
            raise ValueError('Câu hỏi cần 1–2048 ký tự.')
        words = tokenize(prompt)
        if self.encoding.get('surface_normalization')=='terminal-punctuation-v1':
            end=len(words)
            while end and words[end-1] in ('.','!','?'):end-=1
            if end:words=words[:end]
        normalized = ' '.join(words)
        if self.encoding['kind']=='lexical':
            def hashed(parts, weighted=False):
                out=np.zeros(self.context_size,np.float32)
                for part in parts:
                    code=int.from_bytes(hashlib.blake2b(part.encode(),digest_size=8).digest(),'little')
                    weight=self.encoding.get('idf',{}).get(part,1.) if weighted else 1.
                    out[code%self.context_size]+=weight*(1 if (code>>8)&1 else -1)
                return out/max(1.,float(np.linalg.norm(out)))
            vector=.8*hashed(words,True)+.2*hashed([normalized[i:i+3] for i in range(max(0,len(normalized)-2))])
            return vector/max(1.,float(np.linalg.norm(vector)))
        features = words + [normalized[i:i+3] for i in range(max(0, len(normalized)-2))]
        result = np.zeros(self.context_size, np.float32)
        for part in features:
            code = int.from_bytes(hashlib.blake2b(part.encode(), digest_size=8).digest(), 'little')
            result[code % self.context_size] += 1 if (code >> 8) & 1 else -1
        return result / max(1., float(np.linalg.norm(result)))

    def tick(self, x=None, track=False, kind='idle', source=None):
        if x is None:
            x = np.zeros(self.core.config.inputs, np.float32)
        if self.core.config.hidden==48 and self.active_count==48:
            self.core.step(x, track=track)
        else:self._active_step(x,track)
        self.ticks += 1
        self.last_active[:self.active_count]=self.ticks
        if self.on_tick is not None:
            self.on_tick(kind, source)

    def boundary(self):
        # Neural dynamics continue; no reset or independent per-modality core.
        for _ in range(self.settle_ticks):
            self.tick(kind='settle')

    def word_step(self, previous, context, position, track=False, source=None):
        x = np.zeros(self.core.config.inputs, np.float32)
        x[previous] = float(self.encoding.get('previous_gain',.8))
        x[len(self.vocabulary):self.image_start] = context * float(self.encoding.get('context_gain',1.6))
        x[-6:] = [1., 0., .2, position/24, .2*np.sin(position), .2*np.cos(position)]
        for _ in range(3):
            self.tick(x, track, 'language', source)
        return self.probabilities('language', position)

    def probabilities(self, modality, position=0):
        c = self.core
        logits = c.w_out[:,:self.active_count] @ c.features[:self.active_count] + c.bias
        mask = np.zeros(c.config.outputs, bool)
        if modality == 'language':
            mask[:len(self.vocabulary)] = True
            mask[1] = False
            if position == 0:
                mask[0] = False
                for i, token in enumerate(self.vocabulary[2:], 2):
                    if not any(ch.isalnum() for ch in token):
                        mask[i] = False
        else:
            mask[len(self.vocabulary):] = True
        p = np.zeros(c.config.outputs, np.float32)
        values = np.exp(logits[mask] - np.max(logits[mask]))
        p[mask] = values / values.sum()
        return p

    def learn(self, target, modality, position=0, mode='full'):
        if mode not in ('full','readout'):raise ValueError('Unknown local learning mode')
        c = self.core
        p = self.probabilities(modality, position)
        if not 0 <= target < len(p) or p[target] <= 0:
            raise ValueError('Target outside active readout')
        loss = -float(np.log(max(float(p[target]), 1e-12)))
        error = -p.copy(); error[target] += 1
        c.w_out[:,:self.active_count] += np.clip(c.config.readout_lr * error[:, None] * c.features[None, :self.active_count], -c.config.update_clip, c.config.update_clip)
        c.bias += np.clip(c.config.readout_lr * error, -c.config.update_clip, c.config.update_clip)
        reward = float(p[target]); advantage = float(np.clip(reward-c.reward_mean, -1, 1))
        c.reward_mean = .95*c.reward_mean + .05*reward
        for name, trace in (('in', c.elig_in), ('rec', c.elig_rec)):
            if mode=='readout':continue
            weights = getattr(c, 'w_'+name)
            weights[c.rows] += np.clip(c.config.hidden_lr*advantage*trace, -c.config.update_clip, c.config.update_clip)
            np.clip(weights, -c.config.weight_clip, c.config.weight_clip, out=weights)
        np.clip(c.w_out, -c.config.weight_clip, c.config.weight_clip, out=c.w_out)
        np.clip(c.bias, -c.config.weight_clip, c.config.weight_clip, out=c.bias)
        c.updates += 1
        return loss

    def train_pair(self, prompt, answer, mode='full'):
        tokens = tokenize(answer)
        if not 1 <= len(tokens) <= 24 or any(t not in self.ids for t in tokens):
            raise ValueError('Câu trả lời cần 1–24 token trong vocabulary hiện tại.')
        self.boundary()
        # Clear only learning traces at the start of a labeled segment. Neuron
        # voltage/adaptation/spikes/features keep evolving without a reset.
        self.core.elig_in.fill(0); self.core.elig_rec.fill(0)
        self.core.pre.fill(0); self.core.post.fill(0)
        start = (self.core.updates*self.core.config.plastic_rows) % self.active_count
        self.core.rows[:] = (np.arange(12)+start) % self.active_count
        previous = 1; loss = 0.
        context = self.context(prompt)
        for position, token in enumerate([self.ids[t] for t in tokens]+[0]):
            self.word_step(previous, context, position, True, prompt)
            loss += self.learn(token, 'language', position,mode)
            previous = token
        return loss / (len(tokens)+1)

    def generate(self, prompt, max_words=24, stop_at_sentence=False):
        context = self.context(prompt)
        if type(max_words) is not int or not 1 <= max_words <= 32:
            raise ValueError('Giới hạn sinh 1–32 token.')
        self.boundary(); previous = 1; words = []; scores = []; stopped = False; reason = 'length'
        for position in range(max_words):
            p = self.word_step(previous, context, position, source=prompt)
            token = int(np.argmax(p)); scores.append(float(p[token]))
            if token == 0:
                stopped = True; reason = 'eos'; break
            words.append(self.vocabulary[token]); previous = token
            if stop_at_sentence and self.vocabulary[token] in ('.','!','?'):
                stopped = True; reason = 'sentence_boundary'; break
        return {'text': render(words), 'tokens': words, 'stopped': stopped, 'stop_reason': reason,
                'mean_probability': float(np.mean(scores)), 'source': 'shared ALIF neural word readout; no answer lookup'}

    def image(self, pixels, train_label=None, mode='full'):
        pixels = np.asarray(pixels, np.float32)
        if pixels.size != 64 or not np.isfinite(pixels).all() or pixels.min() < 0 or pixels.max() > 16:
            raise ValueError('Ảnh neural cần 64 giá trị trong [0,16].')
        self.boundary()
        x = np.zeros(self.core.config.inputs, np.float32)
        x[self.image_start:self.image_start+64] = pixels.reshape(-1)/16-.3
        x[-6:] = [0., 1., .2, 0., 0., 0.]
        if train_label is not None:
            if type(train_label) is not int or not 0 <= train_label <= 9: raise ValueError('Nhãn chữ số0–9.')
            self.core.elig_in.fill(0); self.core.elig_rec.fill(0)
            self.core.pre.fill(0); self.core.post.fill(0)
            self.core.rows[:] = (np.arange(12)+self.core.updates*12) % self.active_count
        for _ in range(4):
            self.tick(x, train_label is not None, 'vision', '8x8 digit')
        p = self.probabilities('vision')[len(self.vocabulary):]
        result = {'digit': int(np.argmax(p)), 'probabilities': p.tolist(),
                  'scope': 'handwritten_digits_only; not screenshot/OCR/visualQA'}
        if train_label is not None:
            result['loss'] = self.learn(len(self.vocabulary)+train_label, 'vision',mode=mode)
        return result

    def dumps(self):
        payload = self.core.dumps()
        meta = {'format': 'scarx-generation1-core-v2', 'vocabulary': self.vocabulary,
                'ticks': self.ticks, 'sha256': hashlib.sha256(payload).hexdigest(),
                'encoding':self.encoding,'active_count':self.active_count,'max_neurons':self.max_neurons,
                'capacity_rng':self.capacity_rng.bit_generator.state,'last_active':self.last_active.tolist(),
                'capacity_events':self.capacity_events[-32:],'neuron_ids':list(range(self.core.config.hidden))}
        output = io.BytesIO()
        with zipfile.ZipFile(output, 'w') as archive:
            archive.writestr(zipfile.ZipInfo('core.npz'), payload)
            archive.writestr(zipfile.ZipInfo('metadata.json'), json.dumps(meta, ensure_ascii=False))
        return output.getvalue()

    @classmethod
    def loads(cls, payload):
        if len(payload) > 2_000_000: raise ValueError('Oversized generation checkpoint')
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            if len(archive.namelist()) != 2 or set(archive.namelist()) != {'core.npz', 'metadata.json'} or sum(x.file_size for x in archive.infolist()) > 2_000_000:
                raise ValueError('Invalid generation checkpoint')
            meta = json.loads(archive.read('metadata.json')); data = archive.read('core.npz')
        if meta.get('format') not in ('scarx-generation1-core','scarx-generation1-core-v2') or hashlib.sha256(data).hexdigest() != meta['sha256']:
            raise ValueError('Checkpoint checksum/format')
        with zipfile.ZipFile(io.BytesIO(data)) as inner:
            if sum(i.file_size for i in inner.infolist()) > 2_000_000 or len(inner.namelist()) != len(set(inner.namelist())):
                raise ValueError('Oversized or duplicate neural arrays')
        model = cls(meta['vocabulary'],encoding=meta.get('encoding')); core = SpikingCore.loads(data)
        if core.config.inputs != model.core.config.inputs or core.config.outputs != model.core.config.outputs or core.config.neuron!='alif' or not 48<=core.config.hidden<=128 or (core.config.fan_in, core.config.plastic_rows)!=(8,12):
            raise ValueError('Incompatible generation topology')
        model.core = core; model.ticks = int(meta['ticks'])
        if model.ticks < 0: raise ValueError('Invalid tick counter')
        model.active_count=int(meta.get('active_count',core.config.hidden));model.max_neurons=int(meta.get('max_neurons',128))
        if not 48<=model.active_count<=core.config.hidden<=model.max_neurons<=128:raise ValueError('Invalid capacity bounds')
        model.last_active=np.asarray(meta.get('last_active',[model.ticks]*core.config.hidden),dtype=np.int64)
        if model.last_active.shape!=(core.config.hidden,) or np.any(model.last_active<0) or np.any(model.last_active>model.ticks):raise ValueError('Invalid neuron clocks')
        if meta.get('neuron_ids',list(range(core.config.hidden)))!=list(range(core.config.hidden)):raise ValueError('Invalid neuron IDs')
        if 'capacity_rng' in meta:model.capacity_rng.bit_generator.state=meta['capacity_rng']
        model.capacity_events=list(meta.get('capacity_events',[]))[-32:]
        return model

    def select_capacity(self, requested):
        """Actual bounded allocation, not a claim of task difficulty or intelligence.

        Old48neurons/indices/weights remain intact; new readout columns start0.
        New neurons need learning. Dormancy retains state with lazy decay.
        """
        if type(requested) is not int or not 48<=requested<=self.max_neurons:raise ValueError('Neuron budget48–128')
        if requested==self.active_count:return
        old=self.core;changed=requested!=self.active_count
        if requested>old.config.hidden:
            old_h=old.config.hidden;new_h=requested
            new=SpikingCore(replace(old.config,hidden=new_h,seed=int(self.capacity_rng.integers(1,2**30))))
            new.w_in-=.15;new.stable_in[:]=new.w_in
            for name in ('w_in','stable_in','w_rec','stable_rec','indices'):
                getattr(new,name)[:old_h]=getattr(old,name)
            for name in ('w_out','stable_out'):
                getattr(new,name).fill(0);getattr(new,name)[:,:old_h]=getattr(old,name)
            for name in ('bias','stable_bias','rows','elig_in','elig_rec'):getattr(new,name)[:]=getattr(old,name)
            for name in ('voltage','adaptation','spikes','post','features'):getattr(new,name)[:old_h]=getattr(old,name)
            new.pre[:old.config.inputs]=old.pre[:old.config.inputs]
            new.pre[new.config.inputs:new.config.inputs+old_h]=old.pre[old.config.inputs:]
            new.updates=old.updates;new.reward_mean=old.reward_mean
            old.__dict__.update(new.__dict__)
            self.last_active=np.pad(self.last_active,(0,new_h-old_h),constant_values=self.ticks)
            self.capacity_events.append({'tick':self.ticks,'from':old_h,'to':new_h,'new_readout_columns':'zero; untrained'})
        self.active_count=requested
        self.core.rows[:]=(np.arange(12)+self.core.updates*12)%requested
        if changed:self.core.elig_in.fill(0);self.core.elig_rec.fill(0)

    def expand_vocabulary(self,answer):
        """Explicit labeled lessons only; old token IDs and hidden neurons persist."""
        additions=sorted(set(tokenize(answer))-set(self.vocabulary))
        if any(len(t)>64 for t in additions):raise ValueError('Một token đầu ra tối đa64ký tự.')
        if not additions:return 0
        if len(self.vocabulary)+len(additions)>256:raise ValueError('Targetvocabularylimit256; cần protocol/codec mới trước khi mở rộng.')
        old=self.core;v=len(self.vocabulary);new_v=v+len(additions);shift=len(additions)
        new=SpikingCore(replace(old.config,inputs=old.config.inputs+shift,outputs=old.config.outputs+shift))
        for name in ('w_in','stable_in'):
            source=getattr(old,name);dest=getattr(new,name)
            dest[:,:v]=source[:,:v];dest[:,new_v:]=source[:,v:]
            dest[:,v:new_v]-=.15
        for name in ('w_out','stable_out'):
            source=getattr(old,name);dest=getattr(new,name);dest.fill(0)
            dest[:v]=source[:v];dest[new_v:]=source[v:]
        for name in ('bias','stable_bias'):
            source=getattr(old,name);dest=getattr(new,name)
            dest[:v]=source[:v];dest[new_v:]=source[v:];dest[v:new_v]=-6.
        for name in ('w_rec','stable_rec','indices','voltage','adaptation','spikes','post','features','rows','elig_rec'):
            getattr(new,name)[:]=getattr(old,name)
        new.pre[:v]=old.pre[:v];new.pre[new_v:]=old.pre[v:]
        new.elig_in[:,:v]=old.elig_in[:,:v];new.elig_in[:,new_v:]=old.elig_in[:,v:]
        new.updates=old.updates;new.reward_mean=old.reward_mean
        old.__dict__.update(new.__dict__)
        self.vocabulary+=additions;self.ids={word:i for i,word in enumerate(self.vocabulary)}
        self.image_start=new_v+self.context_size
        return shift

    def _active_step(self,x,track):
        c=self.core;cfg=c.config;n=self.active_count;now=self.ticks+1
        elapsed=np.minimum(now-self.last_active[:n],100000)
        old=np.zeros(cfg.hidden,np.float32);old[:n]=np.where(elapsed==1,c.spikes[:n],0)
        current=c.w_in[:n]@x+np.sum(c.w_rec[:n]*old[c.indices[:n]],axis=1)
        c.voltage[:n]=np.power(cfg.leak,elapsed)*c.voltage[:n]+current
        limit=cfg.threshold+cfg.adaptation_strength*np.power(cfg.adaptation_decay,elapsed-1)*c.adaptation[:n]
        c.spikes[:n]=(c.voltage[:n]>=limit).astype(np.float32);c.voltage[:n]-=c.spikes[:n]*limit
        c.adaptation[:n]=np.power(cfg.adaptation_decay,elapsed)*c.adaptation[:n]+c.spikes[:n]
        c.features[:n]=np.power(cfg.feature_decay,elapsed)*c.features[:n]+(1-cfg.feature_decay)*c.spikes[:n]
        if track:
            r=c.rows;c.elig_in*=cfg.trace_decay;c.elig_rec*=cfg.trace_decay
            c.elig_in+=c.spikes[r,None]*c.pre[None,:cfg.inputs]-.5*c.post[r,None]*x[None,:]
            c.elig_rec+=c.spikes[r,None]*c.pre[cfg.inputs:][c.indices[r]]-.5*c.post[r,None]*old[c.indices[r]]
            np.clip(c.elig_in,-5,5,out=c.elig_in);np.clip(c.elig_rec,-5,5,out=c.elig_rec)
            c.pre*=cfg.trace_decay;c.pre[:cfg.inputs]+=x;c.pre[cfg.inputs:]+=old
            c.post*=cfg.trace_decay;c.post[:n]+=c.spikes[:n]
