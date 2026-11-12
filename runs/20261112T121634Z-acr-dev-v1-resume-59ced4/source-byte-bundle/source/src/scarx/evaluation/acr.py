"""Research-only dual eligibility on ONE unchanged ALIF forward. No autograd.

Mixing is after filtering, with endpoint copies for bitwise reproduction.
State routing is a fixed local heuristic; labels arrive only at episode end.
"""
import copy
import numpy as np
from .r8_matched_credit import MatchedLocalCredit
from .readout_eligibility import spike_slope

AB_PROFILES=('static_0','static_25','static_50','static_75','static_100',
             'linear_up','linear_down','groups_up','groups_down','intrinsic')
C_PROFILES=('state','state_shuffle','random_alpha','state_signal_shuffle')
PROFILES=AB_PROFILES+C_PROFILES+('readout_only',)
EXTRA_ARRAYS=('e_f_own','e_v_stop','e_a_stop','e_f_stop','alpha','router_q',
              'router_bias','last_u','last_theta','last_a','last_signal','last_delta')


def sigmoid(x):
    return (1/(1+np.exp(-np.clip(x,-30,30)))).astype(np.float32)


class AdaptiveCreditRouting(MatchedLocalCredit):
    def __init__(self,config,profile='static_50',condition='learned_mtsg'):
        if profile not in PROFILES:raise ValueError('Unregistered ACR profile')
        if condition not in ('learned_mtsg','shuffled_signal','readout_only','frozen_hidden'):
            raise ValueError('Unregistered ACR signal')
        if profile=='readout_only':condition='readout_only'
        if profile=='state_signal_shuffle':condition='shuffled_signal'
        arm={'learned_mtsg':'multi_gated','shuffled_signal':'multi_gated_shuffle',
             'readout_only':'multi_gated_readout_only','frozen_hidden':'multi_gated_frozen'}[condition]
        super().__init__(config,arm)
        if (config.inputs,config.outputs,config.plastic_rows)!=(8,2,12):raise ValueError('ACR fixed layout')
        self.profile=profile
        for name in EXTRA_ARRAYS[:4]:setattr(self,name,np.zeros_like(self.e_f))
        for name in EXTRA_ARRAYS[4:-1]:setattr(self,name,np.zeros(config.hidden,np.float32))
        self.last_delta=np.zeros_like(self.e_f)
        gamma=self.gamma_e_vec[:,0]
        self.router_bias[:]=4*(gamma-(.25+.92)/2)/(.92-.25)
        self.router_rng=np.random.default_rng(config.seed+43100000)
        self.alpha[:]=self.fixed_alpha()

    def fixed_alpha(self):
        p=self.profile;h=self.config.hidden
        if p.startswith('static_'):return np.full(h,int(p[7:])/100,np.float32)
        if p.startswith('linear_'):
            a=np.linspace(.1,.9,h,dtype=np.float32);return a if p=='linear_up' else a[::-1].copy()
        if p.startswith('groups_'):
            a=np.r_[np.full(h//2,.25,np.float32),np.full(h-h//2,.75,np.float32)]
            return a if p=='groups_up' else a[::-1].copy()
        if p=='readout_only':return np.zeros(h,np.float32)
        return sigmoid(self.router_bias)

    def reset_episode(self):
        super().reset_episode()
        for name in ('e_f_own','e_v_stop','e_a_stop','e_f_stop','router_q'):getattr(self,name).fill(0)

    def route(self,u,theta,z,a):
        # No target, loss, task name, accuracy, or test state enters this function.
        if self.profile not in C_PROFILES:return self.fixed_alpha()
        own=np.linalg.norm(self.e_f_own,axis=1);stop=np.linalg.norm(self.e_f_stop,axis=1)
        self.router_q[:]=.9*self.router_q+.1*(stop-own)/(stop+own+1e-6)
        f=np.column_stack((np.clip(np.abs(u-theta)/(theta+1e-6),0,4),z,a/(1+a),self.router_q))
        perm=self.router_rng.permutation(self.config.hidden)  # same draws for all C
        if self.profile=='state_shuffle':f=f[perm]
        result=sigmoid(self.router_bias-1+f@np.asarray([-.5,1.5,1,1],np.float32))
        if self.profile=='random_alpha':result=result[perm]
        return result

    def mix(self):
        a=self.alpha[:,None]
        self.e_f[:]=(1-a)*self.e_f_own+a*self.e_f_stop
        # Endpoint copies avoid signed-zero/rounding changes, including per-row endpoints.
        self.e_f[self.alpha==0]=self.e_f_own[self.alpha==0]
        self.e_f[self.alpha==1]=self.e_f_stop[self.alpha==1]

    def step(self,x,track=True):
        c=self.config;row=np.asarray(x,np.float32)
        if row.shape!=(c.inputs,) or not np.isfinite(row).all():raise ValueError('Invalid input vector')
        if track:
            old_z=self.core.spikes
            theta=c.threshold+c.adaptation_strength*self.core.adaptation
            i_in=self.core.w_in@row;i_rec=np.sum(self.core.w_rec*old_z[self.core.indices],axis=1)
            u=(c.leak*self.core.voltage+i_rec)+i_in  # preserve PINNED grouping
            psi=spike_slope(u-theta)
            pre=np.concatenate((np.broadcast_to(row,(c.hidden,c.inputs)),old_z[self.core.indices]),axis=1)
            a_pre=self.core.adaptation.copy()
            own_u=c.leak*self.e_v+pre;own_a=self.e_a.copy()
            stop_u=c.leak*self.e_v_stop+pre;stop_a=self.e_a_stop.copy()
            own_z=psi[:,None]*(own_u-c.adaptation_strength*own_a)
            stop_z=psi[:,None]*(stop_u-c.adaptation_strength*stop_a)
        self.core.step(row,track=False)  # exactly ONE shared neural forward
        if track:
            self.e_v[:]=own_u-theta[:,None]*own_z-self.core.spikes[:,None]*c.adaptation_strength*own_a
            self.e_v_stop[:]=stop_u
            self.e_a[:]=c.adaptation_decay*own_a+own_z
            self.e_a_stop[:]=c.adaptation_decay*stop_a+stop_z
            gate=1.0 if float(np.max(np.abs(row)))>self.sensory_threshold else .15
            own_z[:,c.inputs:]*=gate;stop_z[:,c.inputs:]*=gate
            self.e_f_own[:]=self.gamma_e_vec*self.e_f_own+(1-self.gamma_e_vec)*own_z
            self.e_f_stop[:]=self.gamma_e_vec*self.e_f_stop+(1-self.gamma_e_vec)*stop_z
            self.alpha[:]=self.route(u,theta,self.core.spikes,a_pre)
            self.mix()
            self.last_u[:]=u;self.last_theta[:]=theta;self.last_a[:]=a_pre
        self.total_ticks+=1
        return self.core.features

    def learn_episode(self,sequence,target,stat_rate=.01):
        # Delegate the actual optimizer verbatim to the pinned baseline. Capture
        # its PRE-update learning signal without an extra forward or RNG draw.
        before=np.concatenate((self.core.w_in,self.core.w_rec),axis=1).astype(np.float64)
        wout=self.core.w_out.copy();bias=self.core.bias.copy()
        rng=copy.deepcopy(self.signal_rng)
        loss=super().learn_episode(sequence,target,stat_rate)
        fhat,raw,scale=self.standardized_features(self.core.features)
        logits=wout@fhat+bias;logits-=np.max(logits)
        p=np.exp(logits);p/=np.sum(p);err=-p.copy();err[int(target)]+=1
        signal=(wout.T@err)*(np.abs(raw)<5)/scale
        perm=rng.permutation(self.config.hidden)
        if self.condition=='shuffled_signal':signal=signal[perm]
        self.last_signal[:]=signal
        self.last_delta[:]=(np.concatenate((self.core.w_in,self.core.w_rec),axis=1).astype(np.float64)-before).astype(np.float32)
        return loss

    def memory_bytes(self):
        return self.core.memory_bytes()+sum(v.nbytes for v in vars(self).values() if isinstance(v,np.ndarray))

    def dumps(self):
        from .forensic.r8_snapshot import dumps
        return dumps(self)

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
            config=Config(**meta['config'])
            if (config.hidden,config.inputs,config.outputs,config.fan_in,config.plastic_rows)!=(48,8,2,8,12):raise ValueError('ACR layout')
            return loads(cls(config,meta['settings']['profile'],meta['settings']['condition']),payload)[0]
        except (KeyError,TypeError,zipfile.BadZipFile) as exc:raise ValueError('Malformed ACR snapshot') from exc


def hidden_matrix(m):return np.concatenate((m.core.w_in,m.core.w_rec),axis=1).copy()


class NormMismatch(ValueError):pass


def ray_to_radius(before,candidate,radius,tolerance=1e-5):
    """Physical float32 ray, bounded search; never relax tolerance to rescue data."""
    w=np.asarray(before);c=np.asarray(candidate)
    if w.shape!=c.shape or w.dtype!=np.float32 or c.dtype!=np.float32 or not np.isfinite(w).all() or not np.isfinite(c).all():
        raise NormMismatch('Invalid physical hidden arrays')
    delta=c.astype(np.float64)-w.astype(np.float64);raw=float(np.linalg.norm(delta))
    if not np.isfinite(radius) or not 0<=radius<=raw:raise NormMismatch('Infeasible radius')
    if radius==0:return w.copy()
    lo=0.;hi=1.;best=None;error=float('inf')
    for _ in range(64):
        scale=(lo+hi)/2
        trial=(w.astype(np.float64)+scale*delta).astype(np.float32)
        norm=float(np.linalg.norm(trial.astype(np.float64)-w.astype(np.float64)))
        if abs(norm-radius)<error:best=trial;error=abs(norm-radius)
        if error<=radius*tolerance:return best
        if norm<radius:lo=scale
        else:hi=scale
    raise NormMismatch('Positive radius not realizable within locked physical tolerance')


def matched_update(a,b,sequence,target):
    """Evaluation control only; restore BOTH whole states if physical matching fails."""
    olda=copy.deepcopy(a);oldb=copy.deepcopy(b);wa=hidden_matrix(a);wb=hidden_matrix(b)
    try:
        la=a.learn_episode(sequence,target);lb=b.learn_episode(sequence,target)
        ca=hidden_matrix(a);cb=hidden_matrix(b)
        na=float(np.linalg.norm(ca.astype(np.float64)-wa));nb=float(np.linalg.norm(cb.astype(np.float64)-wb))
        radius=.5*min(na,nb)
        ma=ray_to_radius(wa,ca,radius);mb=ray_to_radius(wb,cb,radius)
        for m,w in ((a,ma),(b,mb)):
            m.core.w_in[:]=w[:,:m.config.inputs];m.core.w_rec[:]=w[:,m.config.inputs:]
            m.last_delta[:]=(w.astype(np.float64)-(wa if m is a else wb)).astype(np.float32)
        return la,lb,radius,na,nb
    except Exception:
        a.__dict__.clear();a.__dict__.update(olda.__dict__)
        b.__dict__.clear();b.__dict__.update(oldb.__dict__)
        raise
