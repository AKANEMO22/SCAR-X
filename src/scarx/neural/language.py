"""Small vocabulary autoregressive ALIF language pilot; no response lookup.

Prompt hashes are input features, not semantic labels. Next word/punctuation
is sampled from neural probabilities. Corpus answers exist only in training.
"""
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import re
import unicodedata
import zipfile
import numpy as np
from .core import Config,SpikingCore


def tokenize(text):
    return re.findall(r'\w+(?:-\w+)*|[^\w\s]',unicodedata.normalize('NFKC',text).lower(),re.UNICODE)


def render(tokens):
    text=' '.join(tokens)
    text=re.sub(r'\s+([,.!?;:])',r'\1',text)
    return text[:1].upper()+text[1:]


class WordLanguageCore:
    def __init__(self,vocabulary,seed=17,config_overrides=None):
        self.vocabulary=list(vocabulary)
        if self.vocabulary[:2]!=['<eos>','<bos>'] or len(set(vocabulary))!=len(vocabulary) or not 3<=len(vocabulary)<=256:
            raise ValueError('Invalid pilot vocabulary')
        self.ids={word:i for i,word in enumerate(vocabulary)}
        self.context_size=128
        parameters=dict(inputs=len(vocabulary)+self.context_size+3,hidden=48,outputs=len(vocabulary),fan_in=8,plastic_rows=12,seed=seed,
                 leak=.65,threshold=.5,feature_decay=.25,adaptation_strength=.02,readout_lr=.2,hidden_lr=.0003,anchor_strength=0)
        if config_overrides: parameters.update(config_overrides)
        c=Config(**parameters)
        self.core=SpikingCore(c)
        self.core.w_in-=.15; self.core.stable_in[:]=self.core.w_in

    @classmethod
    def vocabulary_from(cls,pairs):
        return ['<eos>','<bos>']+sorted({word for _,answer in pairs for word in tokenize(answer)})

    def context(self,prompt):
        normalized=unicodedata.normalize('NFKC',prompt).lower().strip()
        if len(normalized)>2048: raise ValueError('Pilot prompt limit 2048 characters')
        result=np.zeros(self.context_size,np.float32)
        features=tokenize(normalized)+[normalized[i:i+3] for i in range(max(0,len(normalized)-2))]
        for part in features:
            h=int.from_bytes(hashlib.blake2b(part.encode(),digest_size=8).digest(),'little')
            result[h%self.context_size]+=1 if (h>>8)&1 else -1
        result/=max(1,float(np.linalg.norm(result)))
        return result

    def step(self,previous,context,position,track=False):
        x=np.zeros(self.core.config.inputs,np.float32)
        x[previous]=1
        x[len(self.vocabulary):len(self.vocabulary)+self.context_size]=context*.8
        x[-3:]=[.2,position/32,.2*np.sin(position)]
        for _ in range(3): p=self.core.step(x,track=track)
        return p

    def train_pair(self,prompt,answer,mode='full'):
        context=self.context(prompt); target=[self.ids[t] for t in tokenize(answer)]+[0]
        self.core.reset_state(); previous=1; loss=0.; correct=0
        for position,token in enumerate(target):
            p=self.step(previous,context,position,track=True)
            correct+=int(np.argmax(p)==token)
            loss+=self.core.learn(token,mode=mode)
            previous=token
        return loss,correct,len(target)

    def teacher_score(self,prompt,answer):
        context=self.context(prompt); self.core.reset_state(); previous=1; loss=0.; correct=0
        target=[self.ids[t] for t in tokenize(answer)]+[0]
        for position,token in enumerate(target):
            p=self.step(previous,context,position)
            loss-=float(np.log(max(float(p[token]),1e-12))); correct+=int(np.argmax(p)==token); previous=token
        return loss,correct,len(target)

    def generate(self,prompt,max_words=32):
        context=self.context(prompt); self.core.reset_state(); previous=1; words=[]; probabilities=[]; stopped=False
        for position in range(max_words):
            p=self.step(previous,context,position).copy(); p[1]=0; p/=p.sum()
            token=int(np.argmax(p)); probabilities.append(float(p[token]))
            if token==0: stopped=True; break
            words.append(self.vocabulary[token]); previous=token
        return {'text':render(words),'tokens':words,'stopped':stopped,'mean_probability':float(np.mean(probabilities)) if probabilities else 0.,
                'source':'autoregressive learned ALIF readout; no answer retrieval','scope':'small-vocabulary conversational pilot'}

    def generate_beam(self,prompt,max_words=32,width=4,length_penalty=.6):
        """Generic neural sequence search; never compares against corpus answers."""
        if not 1<=width<=8 or not 1<=max_words<=64: raise ValueError('Invalid decoder budget')
        context=self.context(prompt); self.core.reset_state()
        names=('voltage','adaptation','spikes','features')
        def state():return {n:getattr(self.core,n).copy() for n in names}
        def restore(saved):
            for n,value in saved.items():setattr(self.core,n,value.copy())
        def rank(item):return item['score']/max(1,len(item['ids']))**length_penalty
        beams=[{'ids':[],'score':0.,'stopped':False,'state':state()}]
        for position in range(max_words):
            candidates=[]
            for beam in beams:
                if beam['stopped']:
                    candidates.append(beam);continue
                restore(beam['state'])
                previous=beam['ids'][-1] if beam['ids'] else 1
                p=self.step(previous,context,position).copy();p[1]=0;p/=p.sum()
                after=state()
                for token in np.argsort(p)[-width:]:
                    token=int(token)
                    candidates.append({'ids':beam['ids']+[token],'score':beam['score']+float(np.log(max(float(p[token]),1e-12))),
                                       'stopped':token==0,'state':after})
            beams=sorted(candidates,key=rank,reverse=True)[:width]
            if all(beam['stopped'] for beam in beams):break
        completed=[beam for beam in beams if beam['stopped']]
        best=max(completed or beams,key=rank)
        words=[self.vocabulary[i] for i in best['ids'] if i!=0]
        return {'text':render(words),'tokens':words,'stopped':best['stopped'],'sequence_log_probability':best['score'],
                'source':'beam search of learned ALIF word probabilities; no answer retrieval','beam_width':width,
                'scope':'small-vocabulary conversational pilot'}

    def save(self,directory):
        directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
        self.core.save(directory/'core.npz')
        metadata={'format':'scarx-word-language-v1','vocabulary':self.vocabulary,'sha256':hashlib.sha256((directory/'core.npz').read_bytes()).hexdigest()}
        (directory/'language.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')

    def dumps(self):
        payload=self.core.dumps(); stream=io.BytesIO()
        metadata={'format':'scarx-word-language-v1','vocabulary':self.vocabulary,'sha256':hashlib.sha256(payload).hexdigest()}
        with zipfile.ZipFile(stream,'w') as z:
            z.writestr(zipfile.ZipInfo('core.npz'),payload)
            z.writestr(zipfile.ZipInfo('language.json'),json.dumps(metadata,ensure_ascii=False))
        return stream.getvalue()

    @classmethod
    def loads(cls,payload):
        if len(payload)>8_000_000:raise ValueError('Language checkpoint too large')
        with zipfile.ZipFile(io.BytesIO(payload)) as z:
            if set(z.namelist())!={'core.npz','language.json'} or len(z.namelist())!=2 or sum(i.file_size for i in z.infolist())>8_000_000:raise ValueError('Invalid language bundle')
            metadata=json.loads(z.read('language.json'));core_payload=z.read('core.npz')
        if metadata.get('format')!='scarx-word-language-v1' or hashlib.sha256(core_payload).hexdigest()!=metadata['sha256']:raise ValueError('Language checksum mismatch')
        model=cls(metadata['vocabulary']);core=SpikingCore.loads(core_payload)
        if core.config.inputs!=model.core.config.inputs or core.config.outputs!=len(model.vocabulary) or core.config.hidden!=48:raise ValueError('Incompatible language core')
        model.core=core
        return model

    def expand_vocabulary(self,answer):
        additions=sorted(set(tokenize(answer))-set(self.vocabulary))
        if not additions:return 0
        if len(self.vocabulary)+len(additions)>256:raise ValueError('Vocabulary pilot giới hạn256token; không tự tăng model.')
        old=self.core; old_size=len(self.vocabulary)
        overrides={k:v for k,v in asdict(old.config).items() if k not in ('inputs','outputs')}
        new=WordLanguageCore(self.vocabulary+additions,old.config.seed,overrides)
        for name in ('w_in','stable_in'):
            getattr(new.core,name)[:,:old_size]=getattr(old,name)[:,:old_size]
            getattr(new.core,name)[:,len(new.vocabulary):]=getattr(old,name)[:,old_size:]
        for name in ('w_out','stable_out','bias','stable_bias'):getattr(new.core,name)[:old_size]=getattr(old,name)
        for name in ('w_rec','stable_rec','indices'):getattr(new.core,name)[:]=getattr(old,name)
        new.core.updates=old.updates;new.core.reward_mean=old.reward_mean;new.core.reset_state()
        self.core,self.vocabulary,self.ids=new.core,new.vocabulary,new.ids
        return len(additions)

    @classmethod
    def load(cls,directory):
        directory=Path(directory); metadata=json.loads((directory/'language.json').read_text(encoding='utf-8'))
        if metadata['format']!='scarx-word-language-v1' or hashlib.sha256((directory/'core.npz').read_bytes()).hexdigest()!=metadata['sha256']:
            raise ValueError('Invalid language checkpoint')
        core=SpikingCore.load(directory/'core.npz')
        model=cls(metadata['vocabulary'],core.config.seed)
        if core.config.inputs!=model.core.config.inputs or core.config.outputs!=len(model.vocabulary) or core.config.hidden!=48:
            raise ValueError('Incompatible language checkpoint')
        model.core=core
        return model
