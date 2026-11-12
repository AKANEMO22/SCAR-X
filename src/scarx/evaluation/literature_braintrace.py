"""BrainTrace D-RTRL equation adapted to SCAR-X's unchanged ALIF forward.

Research-only NumPy adapter, NOT an execution/port of the BrainTrace compiler.
Upstream: chaobrain/braintrace fafa25753b743a2e2d5b47209c4ee947abc02cfb,
Apache-2.0; D_RTRL/ParamDimVjpAlgorithm parameter-dimensional recurrence.
The explicit local block below is derived for SCAR-X's post-spike reset,
adaptation and filtered readout, retaining coupled within-neuron states while
cutting recurrent mixing, including self connections, as upstream's default
diagonal recurrence scope does. No BPTT, autograd or full RTRL in the learner.

The isolated full_surrogate_sensitivity function is an oracle/control ONLY.
It is not called from the learner, app, server, or primary prediction path.
"""
import copy
import numpy as np
from .r8_matched_credit import MatchedLocalCredit
from .readout_eligibility import spike_slope

UPSTREAM_SHA='fafa25753b743a2e2d5b47209c4ee947abc02cfb'
REFERENCE_KIND='braintrace-drtrl-scarx-v1'
CONDITIONS=('learned_mtsg','shuffled_signal','zero_signal','readout_only','frozen_hidden')


def local_blocks(config,theta,spike,psi):
    """D_j = d(v',a',f')/d(v,a,f), B_j = d(v',a',f')/dI_j.

    Surrogate derivatives are defined by the supplied psi; they are not finite
    differences of a hard Heaviside. The injection into each owned synapse is
    B_j[:,None] * presynaptic_activity[None,:].
    """
    theta=np.asarray(theta,np.float32);z=np.asarray(spike,np.float32);p=np.asarray(psi,np.float32)
    if theta.ndim!=1 or theta.shape!=z.shape or theta.shape!=p.shape or not np.isfinite(theta+z+p).all():
        raise ValueError('Finite same-width local blocks required')
    c=config;n=len(theta);d=np.zeros((n,3,3),np.float32)
    b=np.column_stack((1-theta*p,p,(1-c.feature_decay)*p)).astype(np.float32)
    d[:,0,0]=c.leak*(1-theta*p)
    d[:,0,1]=c.adaptation_strength*(theta*p-z)
    d[:,1,0]=c.leak*p;d[:,1,1]=c.adaptation_decay-c.adaptation_strength*p
    d[:,2,0]=(1-c.feature_decay)*c.leak*p
    d[:,2,1]=-(1-c.feature_decay)*c.adaptation_strength*p
    d[:,2,2]=c.feature_decay
    return d,b


class BrainTraceDRTRL(MatchedLocalCredit):
    """Explicit block-D-RTRL adapter; source readout/optimizer remain inherited.

    e_v/e_a/e_f now mean sensitivities of voltage/adaptation/actual readout
    feature. e_f is not filtered a second time. There is no sensory gate.
    No new persistent arrays: three traces replace the three baseline traces.
    """
    def __init__(self,config,condition='learned_mtsg'):
        if condition not in CONDITIONS:raise ValueError('Unregistered reference condition')
        if (config.hidden,config.inputs,config.outputs,config.fan_in,config.plastic_rows,config.neuron)!=(48,8,2,8,12,'alif'):
            raise ValueError('Reference adapter fixed to H48 D8 O2 K8 R12 ALIF')
        arm={'learned_mtsg':'multi_gated','shuffled_signal':'multi_gated_shuffle',
             'zero_signal':'multi_gated_zero','readout_only':'multi_gated_readout_only',
             'frozen_hidden':'multi_gated_frozen'}[condition]
        super().__init__(config,arm)
        self.reference_kind=REFERENCE_KIND
        self.trace_kind='slow';self.gate_mode='none'
        self.gamma_e_vec.fill(config.feature_decay)

    def step(self,x,track=True):
        c=self.config;row=np.asarray(x,np.float32)
        if row.shape!=(8,) or not np.isfinite(row).all():raise ValueError('Invalid input vector')
        if track:
            old_z=self.core.spikes
            theta=c.threshold+c.adaptation_strength*self.core.adaptation
            i_in=self.core.w_in@row;i_rec=np.sum(self.core.w_rec*old_z[self.core.indices],axis=1)
            # Keep the paired estimators' source grouping. Core itself still
            # uses leak*v+(i_in+i_rec); the real-valued equation is identical,
            # but this inherited floating-point convention can differ by 1ULP.
            u=(c.leak*self.core.voltage+i_rec)+i_in
            psi=spike_slope(u-theta)
            pre=np.concatenate((np.broadcast_to(row,(48,8)),old_z[self.core.indices]),axis=1)
            previous=np.stack((self.e_v,self.e_a,self.e_f),axis=1)
        self.core.step(row,track=False)  # one unchanged hard-spike forward
        if track:
            d,b=local_blocks(c,theta,self.core.spikes,psi)
            current=np.einsum('hab,hbk->hak',d,previous)+b[:,:,None]*pre[:,None,:]
            self.e_v[:]=current[:,0];self.e_a[:]=current[:,1];self.e_f[:]=current[:,2]
        self.total_ticks+=1
        return self.core.features

    @classmethod
    def loads(cls,payload):
        import io,zipfile
        from .forensic.r8_snapshot import metadata,loads,MAX_BYTES
        from ..neural.core import Config
        if not isinstance(payload,bytes) or len(payload)>MAX_BYTES:raise ValueError('Snapshot bound')
        try:
            with zipfile.ZipFile(io.BytesIO(payload)) as z:
                if len(z.namelist())!=len(set(z.namelist())) or sum(i.file_size for i in z.infolist())>MAX_BYTES:
                    raise ValueError('Snapshot member bound')
                if z.getinfo('metadata.json').file_size>65536:raise ValueError('Metadata bound')
                meta=metadata(z.read('metadata.json'))
            c=Config(**meta['config'])
            return loads(cls(c,meta['settings']['condition']),payload)[0]
        except (KeyError,TypeError,zipfile.BadZipFile) as exc:raise ValueError('Malformed reference snapshot') from exc


def full_surrogate_sensitivity(core,sequence):
    """CONTROL ONLY: full forward sensitivity of f to all hidden parameters.

    Frozen parameters, no learning, no targets, no reverse-time tape. State
    sensitivities retain recurrent spike paths that diagonal D-RTRL cuts.
    Uses <=48neurons, <=8inputs/fan-in, <=256ticks: well below2GB. This oracle
    is NOT a locally learned candidate and cannot be used as a serving core.
    Parameter order: row-major [input weights | sparse recurrent weights].
    """
    c=core.config
    if not (1<=c.hidden<=48 and 1<=c.inputs<=8 and 1<=c.fan_in<=8):raise ValueError('Bounded oracle only')
    seq=np.asarray(sequence,np.float32)
    if seq.ndim!=2 or seq.shape[1]!=c.inputs or not 1<=len(seq)<=256 or not np.isfinite(seq).all():
        raise ValueError('Bounded finite oracle sequence')
    m=copy.deepcopy(core);m.reset_state();h=c.hidden;k=c.inputs+c.fan_in;pcount=h*k
    ev=np.zeros((h,pcount),np.float64);ea=ev.copy();ez=ev.copy();ef=ev.copy()
    for row in seq:
        old_z=m.spikes
        theta=c.threshold+c.adaptation_strength*m.adaptation
        i_in=m.w_in@row;i_rec=np.sum(m.w_rec*old_z[m.indices],axis=1)
        u=(c.leak*m.voltage+i_rec)+i_in;psi=spike_slope(u-theta).astype(np.float64)
        pre=np.concatenate((np.broadcast_to(row,(h,c.inputs)),old_z[m.indices]),axis=1)
        injection=np.zeros((h,pcount),np.float64)
        for j in range(h):injection[j,j*k:(j+1)*k]=pre[j]
        ue=c.leak*ev+np.einsum('hk,hkp->hp',m.w_rec.astype(np.float64),ez[m.indices])+injection
        ze=psi[:,None]*(ue-c.adaptation_strength*ea)
        m.step(row,track=False)
        ev=ue-theta[:,None]*ze-m.spikes[:,None]*c.adaptation_strength*ea
        ea=c.adaptation_decay*ea+ze
        ef=c.feature_decay*ef+(1-c.feature_decay)*ze;ez=ze
    return ef


def direction_metrics(estimate,reference):
    a=np.asarray(estimate,np.float64).ravel();b=np.asarray(reference,np.float64).ravel()
    if a.shape!=b.shape or not np.isfinite(a+b).all():raise ValueError('Finite gradient vectors required')
    na=float(np.linalg.norm(a));nb=float(np.linalg.norm(b))
    return {'cosine':float(np.dot(a,b)/(na*nb)) if na*nb else None,
            'sign_agreement':float(np.mean(np.sign(a)==np.sign(b))),
            'norm_ratio':na/nb if nb else None,'estimate_norm':na,'reference_norm':nb,
            'scope':'Frozen-parameter surrogate-gradient oracle; not task accuracy or full-gradient local learning'}
