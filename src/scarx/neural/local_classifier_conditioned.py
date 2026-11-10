"""Final local-classifier adjustment: frozen train-only feature conditioning.

Same ALIF/eligibility as v1, with an explicit local classifier chain factor.
No task-readout/global/time-reversed gradient. Research only, not serving.
"""
from dataclasses import asdict
import hashlib,io,json,zipfile
import numpy as np
from .core import Config
from .local_classifier import LocalClassifierCore,softmax


class ConditionedLocalClassifierCore(LocalClassifierCore):
    extra_names=LocalClassifierCore.extra_names+('classifier_mean','classifier_scale')

    def __init__(self,config):
        super().__init__(config)
        self.classifier_mean=np.zeros(config.hidden,np.float32)
        self.classifier_scale=np.ones(config.hidden,np.float32)
        self.conditioning_frozen=False

    def condition(self,mean,scale):
        if self.conditioning_frozen or self.aux_updates:raise ValueError('Auxiliary calibration already frozen/used')
        mean=np.asarray(mean,np.float32);scale=np.asarray(scale,np.float32)
        if mean.shape!=(self.config.hidden,) or scale.shape!=mean.shape or not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale<.05):raise ValueError('Invalid classifier statistics')
        self.classifier_mean[:]=mean;self.classifier_scale[:]=scale;self.conditioning_frozen=True

    def auxiliary_signal(self,target):
        if not self.conditioning_frozen:raise ValueError('Freeze train-only classifier statistics')
        if type(target) not in (int,np.int32,np.int64) or not 0<=target<self.config.outputs:raise ValueError('Invalid target')
        raw=(self.features-self.classifier_mean)/self.classifier_scale
        p=softmax(self.fixed_classifier@np.clip(raw,-5,5));error=-p.copy();error[target]+=1
        signal=(self.fixed_classifier.T@error)*(np.abs(raw)<5)/self.classifier_scale
        return signal,-float(np.log(max(float(p[target]),1e-12)))

    def dumps(self):
        arrays=self.arrays();meta={'format':'scarx-conditioned-local-classifier-v2','config':asdict(self.config),
            'updates':self.updates,'aux_updates':self.aux_updates,'reward_mean':self.reward_mean,
            'conditioning_frozen':self.conditioning_frozen,'rng':self.rng.bit_generator.state,
            'sha256':{k:hashlib.sha256(a.tobytes()).hexdigest() for k,a in arrays.items()}}
        stream=io.BytesIO();np.savez(stream,metadata=np.array(json.dumps(meta)),**arrays);return stream.getvalue()

    @classmethod
    def loads(cls,payload):
        if len(payload)>20000000:raise ValueError('Oversized conditioned classifier checkpoint')
        with zipfile.ZipFile(io.BytesIO(payload)) as z:
            if sum(i.file_size for i in z.infolist())>20000000 or z.getinfo('metadata.npy').file_size>65536 or len(z.namelist())!=len(set(z.namelist())):raise ValueError('Invalid classifier archive')
        with np.load(io.BytesIO(payload),allow_pickle=False) as z:
            meta=json.loads(str(z['metadata']))
            if meta['format']!='scarx-conditioned-local-classifier-v2':raise ValueError('Classifier format')
            m=cls(Config(**meta['config']));expected=m.arrays()
            if set(z.files)!=set(expected)|{'metadata'}:raise ValueError('Classifier fields')
            for name,template in expected.items():
                a=z[name]
                if a.shape!=template.shape or a.dtype!=template.dtype or not np.isfinite(a).all() or hashlib.sha256(a.tobytes()).hexdigest()!=meta['sha256'][name]:raise ValueError('Corrupt classifier array')
                template[:]=a
            for name in ('updates','aux_updates'):
                if type(meta[name]) is not int or meta[name]<0:raise ValueError('Invalid counter')
                setattr(m,name,meta[name])
            if type(meta['conditioning_frozen']) is not bool or np.any(m.classifier_scale<.05):raise ValueError('Invalid classifier calibration')
            m.conditioning_frozen=meta['conditioning_frozen'];m.reward_mean=meta['reward_mean'];m.rng.bit_generator.state=meta['rng']
            if not np.isfinite(m.reward_mean) or np.any(m.indices<0) or np.any(m.indices>=m.config.hidden) or np.any(m.rows<0) or np.any(m.rows>=m.config.hidden):raise ValueError('Invalid classifier state')
        return m
