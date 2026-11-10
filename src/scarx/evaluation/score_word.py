"""Experimental next-word adapter of locked local score rule; not serving."""
import hashlib,io,json,zipfile
import numpy as np
from ..neural.core import Config
from ..neural.masked_score import MaskedScoreCore
from ..neural.unified import UnifiedCore
from ..neural.language import tokenize,render
from .fixed_word_codes import validate as validate_encoding,context as fixed_context


class ScoreWord:
    context_size=128

    def __init__(self,vocabulary,seed,condition,encoding=None):
        if condition not in ('frozen','full','shuffled') or vocabulary[:2]!=['<eos>','<bos>'] or not 3<=len(vocabulary)<=256 or any(not isinstance(t,str) or not 1<=len(t)<=64 for t in vocabulary) or len(set(vocabulary))!=len(vocabulary):raise ValueError('Invalid word pilot')
        self.vocabulary=list(vocabulary);self.ids={t:i for i,t in enumerate(vocabulary)};self.encoding={'kind':'legacy'} if encoding is None or encoding=={'kind':'legacy'} else validate_encoding(encoding);self.condition=condition
        d=len(vocabulary)+128+64+6;o=len(vocabulary)+10
        self.core=MaskedScoreCore(Config(inputs=d,outputs=o,hidden=48,fan_in=8,plastic_rows=12,seed=seed,
            leak=.65,threshold=.5,feature_decay=.25,adaptation_strength=.02,readout_lr=.08,hidden_lr=.002,anchor_strength=0),
            'independent','adam',False,condition=='frozen',condition=='shuffled')
        self.core.w_in-=.15;self.core.stable_in[:]=self.core.w_in
        self.decoder_rng=np.random.default_rng(seed+10300000);self.responses_seen=0;self.evaluation_ticks=0;self.mask(0)
        if not self.core.output_mask.any():raise ValueError('No first-word output')

    def context(self,prompt):
        return UnifiedCore.context(self,prompt) if self.encoding['kind']=='legacy' else fixed_context(self.encoding,prompt)

    def mask(self,position):
        m=self.core.output_mask;m.fill(False);m[:len(self.vocabulary)]=True;m[1]=False
        if position==0:
            m[0]=False
            for i,t in enumerate(self.vocabulary[2:],2):
                if not any(ch.isalnum() for ch in t):m[i]=False

    def sequence(self,prompt,prefix):
        if len(prefix)>23 or any(type(t) is not int or not 0<=t<len(self.vocabulary) for t in prefix):raise ValueError('Invalid bounded word prefix')
        context=self.context(prompt);events=[np.zeros(self.core.config.inputs,np.float32) for _ in range(12)]
        for position,previous in enumerate([1]+prefix):
            x=np.zeros(self.core.config.inputs,np.float32);x[previous]=.8
            x[len(self.vocabulary):len(self.vocabulary)+128]=1.6*context
            x[-6:]=[1.,0.,.2,position/24,.2*np.sin(position),.2*np.cos(position)]
            events.extend((x.copy(),x.copy(),x.copy()))
        return np.stack(events)

    def calibrate(self,prompts):
        prompts=list(prompts);rng=np.random.default_rng(self.core.config.seed+10100000);f=[]
        for _ in range(128):
            prompt=prompts[int(rng.integers(len(prompts)))];length=int(rng.integers(6))
            prefix=[int(i) for i in rng.integers(2,len(self.vocabulary),length)]
            sequence=self.sequence(prompt,prefix);f.append(self.core.features_many(sequence[None,:,:],int(rng.integers(2**30)),samples=8)[0])
        self.core.calibrate(np.stack(f))

    def learn(self,prompt,target):
        tokens=tokenize(target)
        if not tokens or len(tokens)>23 or any(t not in self.ids for t in tokens):raise ValueError('Target outside bounded word vocabulary')
        prefix=[];losses=[];trajectory_losses=[]
        for position,token in enumerate([self.ids[t] for t in tokens]+[0]):
            self.mask(position);result=self.core.learn_example(self.sequence(prompt,prefix),int(token))
            losses.append(result['loss']);trajectory_losses.append(result['trajectory_loss'])
            if token:prefix.append(int(token))
        self.responses_seen+=1
        return {'loss':float(np.mean(losses)),'trajectory_loss':float(np.mean(trajectory_losses)),'tokens':len(losses)}

    def distribution(self,prompt,prefix,seed):
        self.mask(len(prefix));sequence=self.sequence(prompt,prefix)
        features=self.core.features_many(sequence[None,:,:],seed,samples=8)[0]
        self.evaluation_ticks+=len(sequence)*8
        return self.core.probabilities(features)

    def generate(self,prompt,seed=None,max_words=12):
        if seed is None:seed=int(self.decoder_rng.integers(2**30))
        if type(max_words) is not int or not 1<=max_words<=12:raise ValueError('Word output budget1..12')
        prefix=[];probabilities=[];stopped=False
        for position in range(max_words):
            p=self.distribution(prompt,prefix,seed+position*101);token=int(p.argmax());probabilities.append(float(p[token]))
            if token==0:stopped=True;break
            prefix.append(token)
        return {'text':render([self.vocabulary[i] for i in prefix]),'tokens':prefix,'stopped':stopped,
            'mean_probability':float(np.mean(probabilities)),'scope':'one experimental local-score ALIF core, MC8prefix replay, not serving'}

    def teacher_metrics(self,prompt,target,seed=991):
        tokens=[self.ids[t] for t in tokenize(target)]+[0];prefix=[];loss=0.;correct=0;first=False
        for position,token in enumerate(tokens):
            p=self.distribution(prompt,prefix,seed+position*101);loss-=np.log(max(float(p[token]),1e-12));correct+=int(p.argmax()==token)
            if position==0:first=bool(p.argmax()==token)
            if token:prefix.append(token)
        return {'nll':float(loss/len(tokens)),'token_accuracy':correct/len(tokens),'first_token_correct':first}

    def dumps(self):
        payload=self.core.dumps();meta={'format':'scarx-score-word-pilot-v1','vocabulary':self.vocabulary,'condition':self.condition,
            'responses_seen':self.responses_seen,'evaluation_ticks':self.evaluation_ticks,'decoder_rng':self.decoder_rng.bit_generator.state,'core_sha256':hashlib.sha256(payload).hexdigest()}
        if self.encoding['kind']!='legacy':
            meta['encoding']=self.encoding;meta['encoding_sha256']=hashlib.sha256(json.dumps(self.encoding,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
        out=io.BytesIO()
        with zipfile.ZipFile(out,'w') as z:z.writestr(zipfile.ZipInfo('core.npz'),payload);z.writestr(zipfile.ZipInfo('metadata.json'),json.dumps(meta))
        return out.getvalue()

    @classmethod
    def loads(cls,payload):
        if len(payload)>20000000:raise ValueError('Oversized scoreword checkpoint')
        with zipfile.ZipFile(io.BytesIO(payload)) as z:
            if set(z.namelist())!={'core.npz','metadata.json'} or len(z.namelist())!=2 or sum(i.file_size for i in z.infolist())>20000000 or z.getinfo('metadata.json').file_size>65536:raise ValueError('Word checkpoint members')
            core=z.read('core.npz');meta=json.loads(z.read('metadata.json'))
        if meta['format']!='scarx-score-word-pilot-v1' or hashlib.sha256(core).hexdigest()!=meta['core_sha256']:raise ValueError('Word checkpoint checksum/format')
        if 'encoding' in meta and hashlib.sha256(json.dumps(meta['encoding'],ensure_ascii=False,sort_keys=True).encode()).hexdigest()!=meta.get('encoding_sha256'):raise ValueError('Input encoding checksum')
        model=cls(meta['vocabulary'],0,meta['condition'],meta.get('encoding'));loaded=MaskedScoreCore.loads(core)
        if loaded.config.inputs!=len(model.vocabulary)+128+64+6 or loaded.config.outputs!=len(model.vocabulary)+10 or loaded.config.hidden!=48:raise ValueError('Word core dimensions')
        if loaded.frozen!=(meta['condition']=='frozen') or loaded.shuffle!=(meta['condition']=='shuffled') or loaded.pairing!='independent' or loaded.optimizer!='adam' or loaded.clipped:raise ValueError('Word core policy mismatch')
        if type(meta['responses_seen']) is not int or meta['responses_seen']<0:raise ValueError('Word counter')
        if type(meta.get('evaluation_ticks',0)) is not int or meta.get('evaluation_ticks',0)<0:raise ValueError('Word evaluation counter')
        model.core=loaded;model.responses_seen=meta['responses_seen'];model.evaluation_ticks=meta.get('evaluation_ticks',0);model.decoder_rng.bit_generator.state=meta['decoder_rng'];return model
