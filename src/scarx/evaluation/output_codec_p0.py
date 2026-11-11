"""User-approved isolated output-codec engineering prototype, not serving.

Unchanged ALIF/STDP equations. Legacy arithmetic blocks preserve operation order
under vocabulary layout remapping. No dataset trainer, lookup, proxy or autograd.
"""
from dataclasses import replace
from pathlib import Path
import copy,hashlib,io,json,math,os,re,tempfile,unicodedata,zipfile

import numpy as np

from ..neural.core import Config,SpikingCore
from ..neural.unified import UnifiedCore
from ..neural.language import tokenize as legacy_tokenize,render as legacy_render

VERSIONS=('legacy-word-v1','numeric-atom-v2')
MODEL_FORMAT='scarx-output-codec-p0-model-v1'
SESSION_FORMAT='scarx-output-codec-p0-session-v1'
MAX_PAYLOAD=4_000_000
BASE_MODEL_FIELDS={'active_count','capacity_events','capacity_rng','core','encoding','ids','image_start','last_active','max_neurons','on_tick','ticks','vocabulary'}


def _hash(raw):return hashlib.sha256(raw).hexdigest()
def _json(value):return json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False).encode('utf-8')
def _normalized(text):return ' '.join(unicodedata.normalize('NFKC',text).casefold().split())


class AnswerCodec:
    def __init__(self,version):
        if version not in VERSIONS:raise ValueError('Unknown output codec')
        self.version=version

    def encode(self,text):
        if not isinstance(text,str) or not 1<=len(text)<=2048:raise ValueError('Bounded answer text required')
        try:text.encode('utf-8')
        except UnicodeEncodeError as exc:raise ValueError('Invalid Unicode answer') from exc
        text=unicodedata.normalize('NFKC',text).lower()
        tokens=legacy_tokenize(text) if self.version=='legacy-word-v1' else re.findall(r'[+-]?[0-9]+(?:\.[0-9]+)?|\w+(?:-\w+)*|[^\w\s]',text,re.UNICODE)
        if any(len(t)>64 for t in tokens):raise ValueError('Output token exceeds64characters')
        return tokens

    def render(self,tokens):
        if not isinstance(tokens,(list,tuple)) or any(not isinstance(t,str) or not 1<=len(t)<=64 for t in tokens):
            raise ValueError('Bounded string tokens required')
        return legacy_render(tokens)

    def roundtrip_matches(self,text):return _normalized(self.render(self.encode(text)))==_normalized(text)


class _CompatCore(SpikingCore):
    """Same neural equations; preserve legacy GEMV grouping after column growth."""
    def set_layout(self,legacy_vocabulary,full_vocabulary):
        self.legacy_vocabulary=legacy_vocabulary
        self.full_vocabulary=full_vocabulary
        self.legacy_inputs=np.r_[np.arange(legacy_vocabulary),np.arange(full_vocabulary,self.config.inputs)]
        self.added_inputs=np.arange(legacy_vocabulary,full_vocabulary)

    def step(self,x,track=True):
        c=self.config;x=np.asarray(x,np.float32)
        if x.shape!=(c.inputs,) or not np.isfinite(x).all():raise ValueError('Invalid input vector')
        old=self.spikes.copy()
        w=np.ascontiguousarray(self.w_in[:,self.legacy_inputs]);v=np.ascontiguousarray(x[self.legacy_inputs])
        current=w@v+np.sum(self.w_rec*old[self.indices],axis=1)
        if np.any(x[self.added_inputs]):
            current+=np.ascontiguousarray(self.w_in[:,self.added_inputs])@np.ascontiguousarray(x[self.added_inputs])
        self.voltage=c.leak*self.voltage+current
        limit=c.threshold+c.adaptation_strength*self.adaptation
        self.spikes=(self.voltage>=limit).astype(np.float32)
        self.voltage-=self.spikes*limit
        self.adaptation=c.adaptation_decay*self.adaptation+self.spikes
        self.features=c.feature_decay*self.features+(1-c.feature_decay)*self.spikes
        if track:
            r=self.rows
            self.elig_in*=c.trace_decay;self.elig_rec*=c.trace_decay
            self.elig_in+=self.spikes[r,None]*self.pre[None,:c.inputs]-.5*self.post[r,None]*x[None,:]
            self.elig_rec+=self.spikes[r,None]*self.pre[c.inputs:][self.indices[r]]-.5*self.post[r,None]*old[self.indices[r]]
            np.clip(self.elig_in,-5,5,out=self.elig_in);np.clip(self.elig_rec,-5,5,out=self.elig_rec)
            self.pre*=c.trace_decay;self.pre[:c.inputs]+=x;self.pre[c.inputs:]+=old
            self.post=c.trace_decay*self.post+self.spikes
        return self.probabilities()


def _strict_json(raw):
    def pairs(items):
        out={}
        for key,value in items:
            if key in out:raise ValueError('Duplicate JSON key')
            out[key]=value
        return out
    if len(raw)>65536:raise ValueError('Oversized metadata')
    result=json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda value:(_ for _ in ()).throw(ValueError('Nonfinite JSON')))
    if not isinstance(result,dict):raise ValueError('Metadata object required')
    return result


def _zip(payload,expected,limit=MAX_PAYLOAD):
    if not isinstance(payload,bytes) or len(payload)>limit:raise ValueError('Checkpoint size bound')
    z=zipfile.ZipFile(io.BytesIO(payload))
    if len(z.infolist())!=len(expected) or set(z.namelist())!=set(expected) or sum(i.file_size for i in z.infolist())>limit:
        z.close();raise ValueError('Checkpoint members/bounds')
    if any(i.compress_type not in (zipfile.ZIP_STORED,zipfile.ZIP_DEFLATED) for i in z.infolist()):
        z.close();raise ValueError('Unsupported compression')
    return z


def _npy_header(z,name,shape=None,dtype=None,metadata=False):
    info=z.getinfo(name)
    with z.open(name) as f:
        version=np.lib.format.read_magic(f)
        if version==(1,0):found,fortran,typ=np.lib.format.read_array_header_1_0(f,max_header_size=8192)
        elif version==(2,0):found,fortran,typ=np.lib.format.read_array_header_2_0(f,max_header_size=8192)
        else:raise ValueError('Unsupported NPY version')
        header_end=f.tell()
    if fortran or typ.hasobject:raise ValueError('NPY layout/object rejected')
    if metadata:
        if found!=() or typ.kind!='U' or typ.itemsize>65536:raise ValueError('NPY metadata shape/type')
    elif found!=shape or typ!=np.dtype(dtype):raise ValueError('NPY declared shape/type rejected before allocation')
    if header_end+math.prod(found)*typ.itemsize!=info.file_size:raise ValueError('NPY body/header mismatch')


def _preflight_core(payload,vocabulary_size):
    if len(payload)>2_000_000:raise ValueError('Core byte bound')
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        names=z.namelist()
        if len(names)!=len(set(names)) or sum(i.file_size for i in z.infolist())>2_000_000:raise ValueError('Core duplicate/bounds')
        _npy_header(z,'metadata.npy',metadata=True)
        meta=_strict_json(str(np.load(io.BytesIO(z.read('metadata.npy')),allow_pickle=False)).encode('utf-8'))
        if set(meta)!={'format','config','updates','reward_mean','sha256'} or type(meta['format']) is not int or meta['format']!=1:
            raise ValueError('Unknown core metadata state/format')
        if type(meta['updates']) is not int or meta['updates']<0 or type(meta['reward_mean']) not in (int,float) or not math.isfinite(meta['reward_mean']):
            raise ValueError('Invalid core metadata learning state')
        if set(meta['config'])!=set(Config.__dataclass_fields__):raise ValueError('Core metadata config state')
        c=Config(**meta['config'])
        if (c.hidden,c.fan_in,c.plastic_rows,c.neuron)!=(48,8,12,'alif') or c.inputs!=vocabulary_size+198 or c.outputs!=vocabulary_size+10:
            raise ValueError('P0 supports public H48 layout only')
        if any(type(getattr(c,k)) is not int for k in ('hidden','fan_in','plastic_rows','inputs','outputs')):raise ValueError('Integer layout dimensions required')
        c.validate()
        h,d,k,o,r=c.hidden,c.inputs,c.fan_in,c.outputs,c.plastic_rows
        shapes={'indices':((h,k),'int32'),'w_in':((h,d),'float32'),'w_rec':((h,k),'float32'),'w_out':((o,h),'float32'),
            'bias':((o,),'float32'),'stable_in':((h,d),'float32'),'stable_rec':((h,k),'float32'),
            'stable_out':((o,h),'float32'),'stable_bias':((o,),'float32'),'voltage':((h,),'float32'),
            'adaptation':((h,),'float32'),'spikes':((h,),'float32'),'pre':((d+h,),'float32'),
            'post':((h,),'float32'),'features':((h,),'float32'),'rows':((r,),'int32'),
            'elig_in':((r,d),'float32'),'elig_rec':((r,k),'float32')}
        if set(names)!={n+'.npy' for n in shapes}|{'metadata.npy'}:raise ValueError('Core field set')
        if not isinstance(meta['sha256'],dict) or set(meta['sha256'])!=set(shapes):raise ValueError('Core metadata checksum fields')
        for name,(shape,dtype) in shapes.items():_npy_header(z,name+'.npy',shape,dtype)


def _archive(items):
    out=io.BytesIO()
    with zipfile.ZipFile(out,'w') as z:
        for name,payload in items:z.writestr(zipfile.ZipInfo(name),payload)
    raw=out.getvalue()
    if len(raw)>MAX_PAYLOAD:raise ValueError('Serialized P0 payload too large')
    return raw


class CodecCoreP0(UnifiedCore):
    """Isolated copied public-H48 model; inherited input encoder/learning rule."""
    def __init__(self,*args,**kwargs):raise ValueError('Use explicit migrate or validated loads for the research prototype')
    @classmethod
    def migrate(cls,source,answer_tokens,version='numeric-atom-v2'):
        if type(source) is not UnifiedCore or type(source.core) is not SpikingCore or source.core.config.hidden!=48 or source.active_count!=48:
            raise ValueError('Only unwrapped public-H48 UnifiedCore supported; do not strip foreign learner state')
        c=source.core.config
        if (c.fan_in,c.plastic_rows,c.neuron)!=(8,12,'alif') or c.inputs!=len(source.vocabulary)+198 or c.outputs!=len(source.vocabulary)+10:raise ValueError('Source layout is not public UnifiedCore')
        expected={'config','indices','updates','reward_mean'}|set(SpikingCore.weight_names)|set(SpikingCore.state_names)
        if set(source.core.__dict__)!=expected or source.on_tick is not None:raise ValueError('Unknown core/callback state cannot be silently discarded')
        if set(source.__dict__)!=BASE_MODEL_FIELDS or len(source.capacity_events)>32:raise ValueError('Unknown/unbounded model state cannot be silently discarded')
        if source.ids!={t:i for i,t in enumerate(source.vocabulary)} or source.image_start!=len(source.vocabulary)+source.context_size:
            raise ValueError('Inconsistent source vocabulary/layout state')
        codec=AnswerCodec(version)
        if not isinstance(answer_tokens,(list,tuple,set)) or len(answer_tokens)>256:raise ValueError('Bounded requested token inventory required')
        if any(not isinstance(t,str) or not 1<=len(t)<=64 or t in ('<eos>','<bos>') for t in answer_tokens):raise ValueError('Invalid requested token')
        additions=sorted(set(answer_tokens)-set(source.vocabulary));v=len(source.vocabulary);nv=v+len(additions)
        if nv>256:raise ValueError('Output vocabulary cap256')
        model=copy.deepcopy(source);old=model.core
        new=_CompatCore(replace(old.config,inputs=old.config.inputs+len(additions),outputs=old.config.outputs+len(additions)))
        for name in ('w_in','stable_in'):
            target=getattr(new,name);prior=getattr(old,name);target.fill(0);target[:,:v]=prior[:,:v];target[:,nv:]=prior[:,v:]
        for name in ('w_out','stable_out'):
            target=getattr(new,name);prior=getattr(old,name);target.fill(0);target[:v]=prior[:v];target[nv:]=prior[v:]
        for name in ('bias','stable_bias'):
            target=getattr(new,name);prior=getattr(old,name);target.fill(-6);target[:v]=prior[:v];target[nv:]=prior[v:]
        for name in ('w_rec','stable_rec','indices','voltage','adaptation','spikes','post','features','rows','elig_rec'):
            getattr(new,name)[:]=getattr(old,name)
        new.pre.fill(0);new.pre[:v]=old.pre[:v];new.pre[nv:]=old.pre[v:]
        new.elig_in.fill(0);new.elig_in[:,:v]=old.elig_in[:,:v];new.elig_in[:,nv:]=old.elig_in[:,v:]
        new.updates=old.updates;new.reward_mean=old.reward_mean;new.set_layout(v,nv)
        model.__class__=cls;model.core=new;model.vocabulary+=additions;model.ids={t:i for i,t in enumerate(model.vocabulary)}
        model.image_start=nv+model.context_size;model.codec_version=codec.version;model.legacy_vocab_size=v
        model.enabled_words=np.zeros(nv,bool);model.enabled_words[:v]=True;model.enabled_words[1]=False
        return model

    def select_capacity(self,requested):
        if requested!=48:raise ValueError('P0 does not certify capacity growth')

    def expand_vocabulary(self,answer):
        raise ValueError('P0 requires explicit migration; legacy in-place expansion is unsupported')

    def train_pair(self,*args,**kwargs):
        raise ValueError('P0 permits bounded CodecFixtureSession state fixtures only')

    def image(self,pixels,train_label=None,mode='full'):
        if train_label is not None:raise ValueError('P0 does not permit untracked image training')
        return super().image(pixels,mode=mode)

    def enable_tokens(self,tokens):
        if not isinstance(tokens,(list,tuple,set)) or len(tokens)>256:raise ValueError('Bounded token collection required')
        for token in tokens:
            if token not in self.ids or token=='<bos>':raise ValueError('Unknown/BOS output token')
        for token in tokens:self.enabled_words[self.ids[token]]=True

    def probabilities(self,modality,position=0):
        c=self.core;v=self.legacy_vocab_size;nv=len(self.vocabulary)
        indices=np.r_[np.arange(v),np.arange(nv,nv+10)]
        old_logits=np.ascontiguousarray(c.w_out[indices,:self.active_count])@c.features[:self.active_count]+c.bias[indices]
        logits=np.zeros(c.config.outputs,np.float32);logits[indices]=old_logits
        if nv>v:logits[v:nv]=c.w_out[v:nv,:self.active_count]@c.features[:self.active_count]+c.bias[v:nv]
        mask=np.zeros(c.config.outputs,bool)
        if modality=='language':
            mask[:nv]=self.enabled_words;mask[1]=False
            if position==0:
                mask[0]=False
                for i,token in enumerate(self.vocabulary[2:],2):
                    if not any(ch.isalnum() for ch in token):mask[i]=False
        elif modality=='vision':mask[nv:]=True
        else:raise ValueError('Unknown modality')
        if not mask.any():raise ValueError('No enabled outputs')
        values=np.exp(logits[mask]-np.max(logits[mask]));p=np.zeros(c.config.outputs,np.float32);p[mask]=values/values.sum()
        return p

    def generate(self,*args,**kwargs):
        result=super().generate(*args,**kwargs)
        result['text']=AnswerCodec(self.codec_version).render(result['tokens'])
        result['source']='research-only output codec on one copied shared ALIF core; no lookup'
        return result

    def dumps(self):
        if set(self.__dict__)!=BASE_MODEL_FIELDS|{'codec_version','legacy_vocab_size','enabled_words'} or self.on_tick is not None:
            raise ValueError('Unknown research model state cannot be discarded')
        core_fields={'config','indices','updates','reward_mean','legacy_vocabulary','full_vocabulary','legacy_inputs','added_inputs'}|set(SpikingCore.weight_names)|set(SpikingCore.state_names)
        if type(self.core) is not _CompatCore or set(self.core.__dict__)!=core_fields:raise ValueError('Unknown research core state cannot be discarded')
        raw=UnifiedCore.dumps(self)
        with zipfile.ZipFile(io.BytesIO(raw)) as z:core=z.read('core.npz');meta=json.loads(z.read('metadata.json'))
        meta.update(format=MODEL_FORMAT,codec_version=self.codec_version,legacy_vocab_size=self.legacy_vocab_size,enabled_words=self.enabled_words.tolist())
        meta['metadata_sha256']=_hash(_json(meta))
        return _archive([('core.npz',core),('metadata.json',_json(meta))])

    @classmethod
    def loads(cls,payload):
        try:
            with _zip(payload,{'core.npz','metadata.json'},2_000_000) as z:
                meta=_strict_json(z.read('metadata.json'));core=z.read('core.npz')
            if meta.get('format') in ('scarx-generation1-core','scarx-generation1-core-v2'):
                if 'codec_version' in meta:raise ValueError('Legacy payload cannot silently select a new codec')
                allowed={'format','vocabulary','ticks','sha256','encoding','active_count','max_neurons','capacity_rng','last_active','capacity_events','neuron_ids'}
                if set(meta)-allowed:raise ValueError('Unknown legacy metadata state')
                if not isinstance(meta.get('capacity_events',[]),list) or len(meta.get('capacity_events',[]))>32:raise ValueError('Legacy capacity history bound')
                vocab=meta['vocabulary']
                if not isinstance(vocab,list) or not 3<=len(vocab)<=256:raise ValueError('Legacy vocabulary bounds')
                _preflight_core(core,len(vocab))
                return cls.migrate(UnifiedCore.loads(payload),[],version='legacy-word-v1')
            checksum=meta.pop('metadata_sha256')
            if meta.get('format')!=MODEL_FORMAT or _hash(_json(meta))!=checksum or meta['sha256']!=_hash(core):raise ValueError('Model metadata/checksum')
            vocab=meta['vocabulary'];nv=len(vocab)
            if not 3<=nv<=256 or vocab[:2]!=['<eos>','<bos>'] or len(set(vocab))!=nv:raise ValueError('Vocabulary bounds')
            if any(not isinstance(t,str) or not 1<=len(t)<=64 for t in vocab):raise ValueError('Vocabulary token')
            codec=AnswerCodec(meta['codec_version']);v=meta['legacy_vocab_size'];mask=meta['enabled_words']
            expected={'format','vocabulary','ticks','sha256','encoding','active_count','max_neurons','capacity_rng','last_active','capacity_events','neuron_ids','codec_version','legacy_vocab_size','enabled_words'}
            if set(meta)!=expected:raise ValueError('Unknown/missing model metadata state')
            if type(v) is not int or not 3<=v<=nv or not isinstance(mask,list) or len(mask)!=nv or any(type(b) is not bool for b in mask) or mask[1] or not mask[0]:raise ValueError('Codec mask/layout')
            if type(meta['ticks']) is not int or meta['ticks']<0 or meta.get('active_count')!=48:raise ValueError('P0 clocks/capacity')
            if type(meta['active_count']) is not int or type(meta['max_neurons']) is not int or not 48<=meta['max_neurons']<=128:
                raise ValueError('P0 capacity metadata')
            if not isinstance(meta['capacity_events'],list) or len(meta['capacity_events'])>32:raise ValueError('P0 capacity event bound')
            clocks=meta.get('last_active')
            if not isinstance(clocks,list) or len(clocks)!=48 or any(type(t) is not int or not 0<=t<=meta['ticks'] for t in clocks):raise ValueError('P0 neuron clocks')
            _preflight_core(core,nv)
            original=dict(meta);original['format']='scarx-generation1-core-v2'
            for name in ('codec_version','legacy_vocab_size','enabled_words'):original.pop(name)
            model=UnifiedCore.loads(_archive([('core.npz',core),('metadata.json',_json(original))]))
            model.__class__=cls;model.core.__class__=_CompatCore;model.core.set_layout(v,nv)
            model.codec_version=codec.version;model.legacy_vocab_size=v;model.enabled_words=np.asarray(mask,bool)
            return model
        except (KeyError,TypeError,OverflowError,UnicodeError,json.JSONDecodeError,zipfile.BadZipFile) as exc:
            raise ValueError('Malformed research codec checkpoint') from exc


class CodecFixtureSession:
    """Bounded state/rollback fixture controller; not a dataset/idle trainer."""
    def __init__(self,model,seed=913):
        if type(model) is not CodecCoreP0:raise ValueError('Research codec model required')
        self.model=model;self.rng=np.random.default_rng(seed);self.replay=[];self.fixture_updates=0
        self.controller={'phase':'idle','cursor':0,'replay_cursor':0,'prompt':'','targets':[],'position':0,'previous':1,'mode':'readout'}

    def add_replay_fixture(self,prompt,answer):
        if len(self.replay)>=4 or not isinstance(prompt,str) or len(prompt)>256:raise ValueError('Bounded replay fixture')
        tokens=AnswerCodec(self.model.codec_version).encode(answer)
        if not 1<=len(tokens)<=24 or any(t not in self.model.ids for t in tokens):raise ValueError('Replay fixture target coverage')
        self.replay.append({'prompt':prompt,'answer':answer,'codec_version':self.model.codec_version})

    def begin_fixture(self,prompt,answer,mode='readout'):
        if self.controller['phase']!='idle' or not isinstance(prompt,str) or len(prompt)>256 or mode not in ('full','readout'):raise ValueError('Fixture controller state')
        try:self.model.context(prompt)
        except UnicodeError as exc:raise ValueError('Invalid Unicode fixture prompt') from exc
        tokens=AnswerCodec(self.model.codec_version).encode(answer)
        if not 1<=len(tokens)<=24 or any(t not in self.model.ids or not self.model.enabled_words[self.model.ids[t]] for t in tokens):raise ValueError('Masked/OOV fixture target')
        if not any(c.isalnum() for c in tokens[0]) or self.fixture_updates+len(tokens)+1>32:raise ValueError('Fixture start token/update budget')
        self.model.boundary();c=self.model.core
        c.elig_in.fill(0);c.elig_rec.fill(0);c.pre.fill(0);c.post.fill(0)
        start=c.updates*c.config.plastic_rows%48;c.rows[:]=(np.arange(12)+start)%48
        self.controller.update(phase='teaching',prompt=prompt,targets=[self.model.ids[t] for t in tokens]+[0],position=0,previous=1,mode=mode)

    def advance_fixture(self):
        state=self.controller
        if state['phase']!='teaching' or self.fixture_updates>=32:raise ValueError('No fixture step or update budget exhausted')
        target=state['targets'][state['position']]
        self.model.word_step(state['previous'],self.model.context(state['prompt']),state['position'],True)
        loss=self.model.learn(target,'language',state['position'],state['mode']);self.fixture_updates+=1
        state['previous']=target;state['position']+=1
        if state['position']==len(state['targets']):state.update(phase='idle',cursor=state['cursor']+1,prompt='',targets=[],position=0,previous=1)
        return loss

    def next_replay_fixture(self):
        if not self.replay:raise ValueError('No replay fixtures')
        index=int(self.rng.integers(len(self.replay)));self.controller['replay_cursor']+=1
        return copy.deepcopy(self.replay[index])

    def dumps(self):
        if set(self.__dict__)!={'model','rng','replay','fixture_updates','controller'}:raise ValueError('Unknown fixture learner/controller state cannot be discarded')
        model=self.model.dumps();state={'format':SESSION_FORMAT,'model_sha256':_hash(model),'rng':self.rng.bit_generator.state,
            'replay':self.replay,'fixture_updates':self.fixture_updates,'controller':self.controller}
        state['metadata_sha256']=_hash(_json(state))
        return _archive([('model.zip',model),('session.json',_json(state))])

    @classmethod
    def loads(cls,payload):
        try:
            with _zip(payload,{'model.zip','session.json'}) as z:model=z.read('model.zip');state=_strict_json(z.read('session.json'))
            checksum=state.pop('metadata_sha256')
            if state.get('format')!=SESSION_FORMAT or _hash(_json(state))!=checksum or state['model_sha256']!=_hash(model):raise ValueError('Session checksum/format')
            if set(state)!={'format','model_sha256','rng','replay','fixture_updates','controller'}:raise ValueError('Unknown session metadata state')
            result=cls(CodecCoreP0.loads(model));result.rng.bit_generator.state=state['rng']
            if type(state['fixture_updates']) is not int or not 0<=state['fixture_updates']<=32 or not isinstance(state['replay'],list) or len(state['replay'])>4:raise ValueError('Fixture state bounds')
            for row in state['replay']:
                if set(row)!={'prompt','answer','codec_version'} or row['codec_version']!=result.model.codec_version:raise ValueError('Replay codec/schema')
                result.add_replay_fixture(row['prompt'],row['answer'])
            c=state['controller'];keys={'phase','cursor','replay_cursor','prompt','targets','position','previous','mode'}
            if set(c)!=keys or c['phase'] not in ('idle','teaching') or c['mode'] not in ('readout','full'):raise ValueError('Controller schema')
            for k in ('cursor','replay_cursor','position','previous'):
                if type(c[k]) is not int or c[k]<0:raise ValueError('Controller integer')
            nv=len(result.model.vocabulary)
            if not isinstance(c['prompt'],str) or len(c['prompt'])>256 or not isinstance(c['targets'],list) or len(c['targets'])>25 or any(type(t) is not int or not 0<=t<nv or t==1 for t in c['targets']) or not 0<=c['previous']<nv:raise ValueError('Controller target bounds')
            if c['phase']=='teaching' and (not c['targets'] or c['targets'][-1]!=0 or not c['position']<len(c['targets'])):raise ValueError('Pending fixture cursor')
            if c['phase']=='teaching':
                if c['targets'].count(0)!=1 or c['previous']!=(1 if c['position']==0 else c['targets'][c['position']-1]):raise ValueError('Pending previous token')
                if any(not result.model.enabled_words[t] for t in c['targets']):raise ValueError('Pending target disabled')
                if state['fixture_updates']+len(c['targets'])-c['position']>32:raise ValueError('Pending fixture update budget')
                try:result.model.context(c['prompt'])
                except UnicodeError as exc:raise ValueError('Invalid pending prompt') from exc
            if c['phase']=='idle' and (c['targets'] or c['position'] or c['prompt'] or c['previous']!=1):raise ValueError('Idle fixture cursor')
            result.controller=c;result.fixture_updates=state['fixture_updates'];return result
        except (KeyError,TypeError,OverflowError,UnicodeError,json.JSONDecodeError,zipfile.BadZipFile) as exc:raise ValueError('Malformed fixture session') from exc

    def restore(self,payload):
        replacement=self.loads(payload)
        self.__dict__.clear();self.__dict__.update(replacement.__dict__)

    def guarded_fixture(self,prompt,answer,guard,mode='readout'):
        snapshot=self.dumps()
        try:
            self.begin_fixture(prompt,answer,mode)
            while self.controller['phase']=='teaching':self.advance_fixture()
            if not guard(self):self.restore(snapshot);return False
            return True
        except BaseException:
            self.restore(snapshot);raise

    def save(self,path):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);payload=self.dumps()
        fd,name=tempfile.mkstemp(dir=path.parent,prefix=path.name+'.',suffix='.tmp')
        try:
            with os.fdopen(fd,'wb') as stream:stream.write(payload);stream.flush();os.fsync(stream.fileno())
            os.replace(name,path)
        finally:
            if os.path.exists(name):os.unlink(name)

    @classmethod
    def load(cls,path):
        path=Path(path)
        if path.stat().st_size>MAX_PAYLOAD:raise ValueError('Session file bound')
        return cls.loads(path.read_bytes())
