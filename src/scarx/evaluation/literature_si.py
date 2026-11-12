"""Small mathematical SI reference; NOT connected to an SCAR-X learner.

Paper: Zenke, Poole & Ganguli (ICML2017), equations3–5.
Source: ganguli-lab/pathint @86c5e07c5603d6a44c2f53585c19ee70773ef15c
(MIT), protocols.PATH_INT_PROTOCOL and utils.ema. Independently implemented
from those equations in float64; no TensorFlow/Keras code is imported/copied.

Explicit consolidate() calls are boundary-assisted, even without a task-ID
argument. A fixed-window/task-agnostic schedule would be a separate adaptation.
An approximate local credit vector is NOT the original loss gradient. This
helper cannot grant the SI paper's loss-attribution interpretation to it.
"""
import math
import numpy as np

UPSTREAM_SHA='86c5e07c5603d6a44c2f53585c19ee70773ef15c'
MAX_PARAMETERS=1_000_000


def _array(value,shape=None):
    a=np.asarray(value)
    if a.dtype.kind not in 'fiu' or not 1<=a.size<=MAX_PARAMETERS or a.ndim>4:
        raise ValueError('Bounded real parameter array required')
    if shape is not None and a.shape!=shape:raise ValueError('Parameter shape mismatch')
    a=a.astype(np.float64,copy=True)
    if not np.isfinite(a).all():raise ValueError('Finite parameters required')
    return a


class SynapticIntelligenceState:
    """Per-parameter work/importance with actual deltas and explicit boundaries."""
    format='scarx-si-math-reference-v1'

    def __init__(self,weights,xi=.1):
        if type(xi) not in (int,float) or not math.isfinite(xi) or xi<=0:
            raise ValueError('Positive finite SI damping required')
        w=_array(weights);self.xi=float(xi)
        self.anchor=w.copy();self.previous=w.copy()
        self.work=np.zeros_like(w);self.importance=np.zeros_like(w)
        self.steps=0;self.segments=0

    def observe(self,before,after,gradient_unregularized):
        """Record g=dL_unreg/dw BEFORE update and physical after-before delta.

        The caller must include clipping, regularization, and optimizer effects
        in `after`. Raw proposed steps or post-normalization updates are not g.
        Validation precedes all state mutation.
        """
        a=_array(before,self.anchor.shape);b=_array(after,self.anchor.shape)
        g=_array(gradient_unregularized,self.anchor.shape)
        if not np.array_equal(a,self.previous):raise ValueError('Unobserved weight change')
        candidate=self.work-g*(b-a)
        if not np.isfinite(candidate).all():raise ValueError('SI work overflow')
        self.work[:]=candidate;self.previous[:]=b;self.steps+=1

    def consolidate(self,decay='sum'):
        """Match upstream: rectify AFTER sum/EMA, not each contribution first."""
        if decay!='sum' and (type(decay) not in (int,float) or not math.isfinite(decay) or not 0<=decay<=1):
            raise ValueError('Decay is sum or a finite fraction')
        contribution=self.work/((self.previous-self.anchor)**2+self.xi)
        combined=self.importance+contribution if decay=='sum' else decay*self.importance+(1-decay)*contribution
        updated=np.maximum(0,combined)
        if not np.isfinite(updated).all():raise ValueError('SI importance overflow')
        self.importance[:]=updated;self.anchor[:]=self.previous
        self.work.fill(0);self.segments+=1

    def regularizer(self,weights,strength=1.):
        """Return lambda*sum(Omega*(w-anchor)^2) and its DESCENT gradient."""
        if type(strength) not in (int,float) or not math.isfinite(strength) or strength<0:
            raise ValueError('Nonnegative finite regularizer strength')
        delta=_array(weights,self.anchor.shape)-self.anchor
        penalty=float(strength*np.sum(self.importance*delta**2))
        gradient=2*strength*self.importance*delta
        if not math.isfinite(penalty) or not np.isfinite(gradient).all():raise ValueError('SI penalty overflow')
        return penalty,gradient

    def memory_bytes(self):return sum(getattr(self,k).nbytes for k in ('anchor','previous','work','importance'))

    def state_dict(self):
        """Copy data only; a production file/checkpoint adapter is NOT supplied."""
        return {'format':self.format,'upstream_sha':UPSTREAM_SHA,'xi':self.xi,'steps':self.steps,
            'segments':self.segments,'arrays':{k:getattr(self,k).copy() for k in ('anchor','previous','work','importance')}}

    @classmethod
    def from_state(cls,state):
        if type(state) is not dict or set(state)!={'format','upstream_sha','xi','steps','segments','arrays'}:
            raise ValueError('SI state schema')
        if state['format']!=cls.format or state['upstream_sha']!=UPSTREAM_SHA:raise ValueError('SI reference identity')
        if any(type(state[k]) is not int or not 0<=state[k]<2**63 for k in ('steps','segments')):
            raise ValueError('SI counter bounds')
        arrays=state['arrays'];names={'anchor','previous','work','importance'}
        if type(arrays) is not dict or set(arrays)!=names:raise ValueError('SI array schema')
        result=cls(arrays['anchor'],state['xi'])
        values={k:_array(arrays[k],result.anchor.shape) for k in names}
        if np.any(values['importance']<0):raise ValueError('SI importance must be nonnegative')
        for k,v in values.items():getattr(result,k)[:]=v
        result.steps=state['steps'];result.segments=state['segments']
        return result
