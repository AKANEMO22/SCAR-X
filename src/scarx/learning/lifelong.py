"""Task-agnostic consolidation clocks and output-local Synaptic Intelligence.

No task boundary or clean label is provided to these learners. SI is an
adaptation limited to the readout, not a reproduction of whole-network SI.
"""
from pathlib import Path
import json
import io
import os
import tempfile
import zipfile
import numpy as np
from ..neural.plasticity import PlasticCore


class LifelongLearner:
    modes=("readout_ema","score_ema","score_meta","meta_decay","selective","random_reject","si_readout")

    def __init__(self,config,mode):
        if mode not in self.modes: raise ValueError(mode)
        self.mode=mode
        rule="readout" if mode in ("readout_ema","si_readout") else "score"
        consolidation="ema" if mode in ("readout_ema","score_ema") else "meta"
        self.model=PlasticCore(config,rule,consolidation)
        self.si={}
        for key,name in (("out","w_out"),("bias","bias")):
            a=getattr(self.model,name)
            for prefix in ("work","omega"): self.si[prefix+'_'+key]=np.zeros_like(a)
            self.si['anchor_'+key]=a.copy()

    def update(self,sequence,label):
        m=self.model; m.forward(sequence,track=True)
        if self.mode!='si_readout':
            loss=m.learn(int(label))
        else:
            p=m.probabilities(); loss=-float(np.log(max(float(p[label]),1e-9)))
            error=-p.copy(); error[label]+=1
            for key,name,signal in (("out","w_out",error[:,None]*m.features[None,:]),("bias","bias",error)):
                weights=getattr(m,name); omega=self.si['omega_'+key]; anchor=self.si['anchor_'+key]
                delta=np.clip(m.config.readout_lr*(signal-.1*omega*(weights-anchor)),-m.config.update_clip,m.config.update_clip)
                before=weights.copy(); weights+=delta
                np.clip(weights,-m.config.weight_clip,m.config.weight_clip,out=weights)
                self.si['work_'+key]+=signal*(weights-before)
            m.updates+=1
        if m.updates%32==0 and self.mode in ('meta_decay','selective','random_reject'):
            for key in ('in','rec','out','bias'): getattr(m,'importance_'+key)[:]*=.995
        if m.updates%64==0:
            if self.mode=='si_readout':
                for key,name in (("out","w_out"),("bias","bias")):
                    movement=getattr(m,name)-self.si['anchor_'+key]
                    self.si['omega_'+key][:]=np.minimum(20,self.si['omega_'+key]+np.maximum(0,self.si['work_'+key])/(movement**2+.1))
                    self.si['anchor_'+key][:]=getattr(m,name); self.si['work_'+key].fill(0)
            else: m.consolidate()
        return loss

    def save(self,path):
        path=Path(path); path.mkdir(parents=True,exist_ok=True)
        fd,tmp=tempfile.mkstemp(dir=path,suffix='.tmp')
        try:
            with os.fdopen(fd,'wb') as stream:
                stream.write(self.dumps()); stream.flush(); os.fsync(stream.fileno())
            os.replace(tmp,path/'learner.zip')
        finally:
            if os.path.exists(tmp): os.unlink(tmp)

    def dumps(self):
        state=io.BytesIO(); np.savez(state,mode=np.array(self.mode),**self.si)
        output=io.BytesIO()
        with zipfile.ZipFile(output,'w') as z:
            z.writestr(zipfile.ZipInfo('model.npz'),self.model.dumps()); z.writestr(zipfile.ZipInfo('learner.npz'),state.getvalue())
        return output.getvalue()

    @classmethod
    def loads(cls,payload):
        if len(payload)>100_000_000: raise ValueError('Oversized learner')
        with zipfile.ZipFile(io.BytesIO(payload)) as z:
            if set(z.namelist())!={'model.npz','learner.npz'} or len(z.namelist())!=2 or sum(i.file_size for i in z.infolist())>100_000_000:
                raise ValueError('Invalid learner bundle')
            return cls._decode(z.read('model.npz'),z.read('learner.npz'))

    @classmethod
    def load(cls,path):
        path=Path(path)
        if (path/'learner.zip').exists():
            if (path/'learner.zip').stat().st_size>100_000_000: raise ValueError('Oversized learner')
            return cls.loads((path/'learner.zip').read_bytes())
        # Read historical two-file checkpoints without rewriting research data.
        if sum((path/name).stat().st_size for name in ('model.npz','learner.npz'))>100_000_000: raise ValueError('Oversized learner')
        return cls._decode((path/'model.npz').read_bytes(),(path/'learner.npz').read_bytes())

    @classmethod
    def _decode(cls,model_payload,learner_payload):
        model=PlasticCore.loads(model_payload)
        with zipfile.ZipFile(io.BytesIO(learner_payload)) as archive:
            if sum(i.file_size for i in archive.infolist())>10_000_000: raise ValueError('Oversized learner arrays')
        with np.load(io.BytesIO(learner_payload),allow_pickle=False) as z:
            obj=cls(model.config,str(z['mode'])); obj.model=model
            if set(z.files)!=set(obj.si)|{'mode'}: raise ValueError('Invalid learner fields')
            for k in obj.si:
                a=z[k]
                if a.shape!=obj.si[k].shape or a.dtype!=obj.si[k].dtype or not np.isfinite(a).all(): raise ValueError('Invalid SI state')
                obj.si[k][:]=a
        return obj
