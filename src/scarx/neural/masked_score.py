"""Outputmask adapter for the unchanged local paired-score learning equations."""
import numpy as np
from .paired_score import PairedScoreCore


class MaskedScoreCore(PairedScoreCore):
    checkpoint_format='scarx-masked-score-word-v1'
    extra_names=PairedScoreCore.extra_names+('output_mask',)

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.output_mask=np.ones(self.config.outputs,bool)

    def probabilities(self,features=None):
        f=self.features if features is None else features
        logits=self.transform(f)@self.w_out.T+self.bias;logits=np.where(self.output_mask,logits,-np.inf)
        values=np.exp(logits-logits.max(axis=-1,keepdims=True));return values/values.sum(axis=-1,keepdims=True)

    def learn_example(self,sequence,target):
        if type(target) not in (int,np.int32,np.int64) or not 0<=target<self.config.outputs or not self.output_mask[target]:raise ValueError('Target outside mask')
        return super().learn_example(sequence,target)

    @classmethod
    def loads(cls,payload):
        model=super().loads(payload)
        if not model.output_mask.any():raise ValueError('Empty output mask')
        return model
