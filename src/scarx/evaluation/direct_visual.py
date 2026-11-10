"""Approved30 research on UNCHANGED VisualWords32 forward/cache.

Own-neuron cache sensitivities computed forward only. One readout feedbackstep,
no hidden-to-hidden gradients, autograd, alternateencoder or serving integration.
"""
import copy,hashlib,io,json,zipfile
import numpy as np
from .visual_words import VisualWords
from .readout_eligibility import ReadoutEligibility
from ..neural.language import tokenize


class DirectVisualWords(VisualWords):
    methods=('correct','shuffled','zero','frozen','no_image','no_question')

    def __init__(self,vocabulary,seed,method='correct'):
        if method not in self.methods:raise ValueError('Registered directvisual method')
        super().__init__(vocabulary,seed,method if method in ('no_image','no_question') else 'full')
        self.method=method;self.learner=ReadoutEligibility(self.core.config,method if method in ('shuffled','zero','frozen') else 'correct')
        self.learner.core=self.core;self.learner.head=self.head
        self.cache_e=np.zeros_like(self.learner.e_f);self.pooled_e=np.zeros_like(self.cache_e);self.pooled_feature=np.zeros(48,np.float32)
        self.pending_inputs=np.zeros((33,self.core.config.inputs),np.float32);self.feature_sum=np.zeros(48,np.float32)
        self.prepare_cursor=33;self.cache_count=0;self.training_trace=False;self.recording_cache=False

    def _tick(self,x):
        self.learner.step(x);self.ticks+=1

    def begin_prepare(self,question,image,ablation=None):
        # Same33numericinputs as32, buffered to resume an interruptedimage.
        # This is inputdata, not a reverse-time neuralstate/gradient tape.
        if ablation not in (None,'blank_image','blank_question'):raise ValueError('Unknown causalablation')
        events=self.encoder.image(image)
        if not isinstance(question,str) or not 1<=len(question.strip())<=2048:raise ValueError('Question1..2048chars')
        words=tokenize(question)
        if not 1<=len(words)<=64 or any(len(t)>64 for t in words):raise ValueError('Questiontokens bound')
        q=sum((self.encoder.token_vector(t) for t in words),np.zeros(128,np.float32));q/=max(1.,float(np.linalg.norm(q)))
        if self.condition=='no_question' or ablation=='blank_question':q.fill(0)
        self.learner.begin(self.training_trace);self.question_feature[:]=q
        for n in ('visual_feature','cache_e','pooled_e','pooled_feature','feature_sum','pending_inputs'):getattr(self,n).fill(0)
        self.pending_inputs[12:15]=self._word_input(1,0)
        for index,event in enumerate(events):
            x=event.copy();x[len(self.vocabulary):self.encoder.image_start]=1.6*q;x[-6]=1.
            if self.condition=='no_image' or ablation=='blank_image':x[self.encoder.image_start:self.encoder.image_start+48]=0
            self.pending_inputs[15+index]=x
        self.prepare_cursor=0;self.cache_count=0;self.recording_cache=True;self.prepared=False

    def prepare_step(self):
        if not self.recording_cache or not self.learner.segment_active or not 0<=self.prepare_cursor<33:raise ValueError('No pendingimageinput')
        self._tick(self.pending_inputs[self.prepare_cursor]);self.prepare_cursor+=1
        if 17<=self.prepare_cursor<=32:
            self.cache_e+=self.learner.e_f;self.feature_sum+=self.core.features;self.cache_count+=1
        if self.prepare_cursor==33:
            if self.cache_count!=16:raise ValueError('Imagecache count')
            self.cache_e/=16;self.visual_feature[:]=self.feature_sum/16;self.images_seen+=1;self.prepared=True;self.recording_cache=False

    def prepare(self,question,image,ablation=None):
        self.begin_prepare(question,image,ablation)
        while self.prepare_cursor<33:self.prepare_step()

    def word_feature(self,previous,position):
        feature=super().word_feature(previous,position);self.pooled_feature[:]=feature
        self.pooled_e[:]=.5*(self.cache_e+self.learner.e_f);self.learner.output_mask[:]=self.output_mask
        return feature

    def accumulate_word(self,target):
        """Reuse35 feedback/head math on its ACTUAL pooledpresynapticfeature.

        Substitute only for the endpoint calculation, then restore the neuronal
        dynamics/own filtered eligibility before any nexttick. No hiddenerror
        travels through the cache; its local sensitivity was already accumulated.
        """
        dynamic=self.core.features;eligibility=self.learner.e_f
        try:
            self.core.features=self.pooled_feature;self.learner.e_f=self.pooled_e
            return self.learner.accumulate(target)
        finally:self.core.features=dynamic;self.learner.e_f=eligibility

    def learn(self,question,image,target):
        tokens=self.target_tokens(target)
        if not self.head.frozen:raise ValueError('TRAINcalibration required')
        snapshot=copy.deepcopy(self.__dict__)
        try:
            self.training_trace=True;self.prepare(question,image);previous=1;losses=[]
            for position,token in enumerate(tokens):self.word_feature(previous,position);losses.append(self.accumulate_word(int(token)));previous=token
            self.learner.finish();self.experiences+=1;return {'loss':float(np.mean(losses)),'tokens':len(tokens)}
        except BaseException:self.__dict__.clear();self.__dict__.update(snapshot);raise
        finally:self.training_trace=False

    def generate(self,question,image,max_words=8,ablation=None):
        self.training_trace=False;r=super().generate(question,image,max_words,ablation);self.learner.segment_active=False
        r['scope']='Proposal30directvisual research, actual32forward/neuralwords; NOT main/normalvision.';return r

    def memory_bytes(self):
        return self.learner.memory_bytes()+sum(getattr(self,n).nbytes for n in ('question_feature','visual_feature','output_mask','cache_e','pooled_e','pooled_feature','pending_inputs','feature_sum'))

    def dumps(self):
        visual=VisualWords.dumps(self);learner=self.learner.dumps();state=io.BytesIO();np.savez(state,**{n:getattr(self,n) for n in ('cache_e','pooled_e','pooled_feature','pending_inputs','feature_sum')});state=state.getvalue()
        meta={'format':'scarx-direct-visual-bridge-v1','method':self.method,'training_trace':self.training_trace,'recording_cache':self.recording_cache,'prepare_cursor':self.prepare_cursor,'cache_count':self.cache_count,
            **{k+'_sha256':hashlib.sha256(v).hexdigest() for k,v in [('visual',visual),('learner',learner),('state',state)]}}
        meta['metadata_sha256']=hashlib.sha256(json.dumps(meta,sort_keys=True).encode()).hexdigest();out=io.BytesIO()
        with zipfile.ZipFile(out,'w') as z:
            for name,raw in [('visual.words',visual),('learner.eligibility',learner),('state.npz',state),('metadata.json',json.dumps(meta))]:z.writestr(zipfile.ZipInfo(name),raw)
        raw=out.getvalue()
        if len(raw)>6_000_000:raise MemoryError('Directvisualcheckpoint6MB')
        return raw

    @classmethod
    def loads(cls,payload):
        if len(payload)>6_000_000:raise ValueError('Directvisualcheckpoint bound')
        with zipfile.ZipFile(io.BytesIO(payload)) as z:
            if set(z.namelist())!={'visual.words','learner.eligibility','state.npz','metadata.json'} or len(z.namelist())!=4 or sum(i.file_size for i in z.infolist())>6_000_000 or z.getinfo('metadata.json').file_size>65536:raise ValueError('Directvisualcheckpoint members')
            blobs={k:z.read(n) for k,n in [('visual','visual.words'),('learner','learner.eligibility'),('state','state.npz')]};meta=json.loads(z.read('metadata.json'))
        checksum=meta.pop('metadata_sha256')
        if meta.get('format')!='scarx-direct-visual-bridge-v1' or hashlib.sha256(json.dumps(meta,sort_keys=True).encode()).hexdigest()!=checksum or any(hashlib.sha256(v).hexdigest()!=meta[k+'_sha256'] for k,v in blobs.items()):raise ValueError('Directvisualcheckpoint checksum')
        visual=VisualWords.loads(blobs['visual']);learned=ReadoutEligibility.loads(blobs['learner']);m=cls(visual.vocabulary,visual.core.config.seed,meta['method'])
        if visual.condition!=m.condition or learned.condition!=m.learner.condition or learned.core.dumps()!=visual.core.dumps() or learned.head.dumps()!=visual.head.dumps():raise ValueError('Directvisual sharedcore/state mismatch')
        m.__dict__.update(visual.__dict__);m.learner=learned;m.core=learned.core;m.head=learned.head
        with zipfile.ZipFile(io.BytesIO(blobs['state'])) as z:
            if len(z.namelist())!=5 or len(set(z.namelist()))!=5 or sum(i.file_size for i in z.infolist())>500000:raise ValueError('Directvisual statebound')
        with np.load(io.BytesIO(blobs['state']),allow_pickle=False) as z:
            if set(z.files)!={'cache_e','pooled_e','pooled_feature','pending_inputs','feature_sum'}:raise ValueError('Directvisual statefields')
            for name in z.files:
                a=z[name];template=getattr(m,name)
                if a.shape!=template.shape or a.dtype!=template.dtype or not np.isfinite(a).all():raise ValueError('Directvisual statearray')
                template[:]=a
        for name in ('training_trace','recording_cache'):
            if type(meta[name]) is not bool:raise ValueError('Directvisual flag')
            setattr(m,name,meta[name])
        if type(meta['prepare_cursor']) is not int or not 0<=meta['prepare_cursor']<=33 or type(meta['cache_count']) is not int or not 0<=meta['cache_count']<=16:raise ValueError('Directvisual preparecounter')
        m.prepare_cursor=meta['prepare_cursor'];m.cache_count=meta['cache_count']
        if m.recording_cache and (m.prepared or m.prepare_cursor>=33 or m.cache_count!=max(0,min(16,m.prepare_cursor-16))):raise ValueError('Directvisual pendingimage consistency')
        if learned.total_ticks!=m.ticks or learned.segments_learned!=m.experiences:raise ValueError('Directvisual counters')
        return m
