"""Approved H64/H128 capacity diagnostic; original ALIF/learning inherited.

Only constructor dimensions are generalized. Own step and source optimizer,
and groups_up dual eligibility step, are the pinned research implementations.
No production core changes, autograd or global recurrent credit propagation.
"""
import io,zipfile
import numpy as np
from ..neural.core import SpikingCore,Config
from .r8_matched_credit import MatchedLocalCredit
from .acr import AdaptiveCreditRouting,EXTRA_ARRAYS

HEIGHTS=(48,64,128)
CONDITIONS=('learned_mtsg','readout_only','shuffled_signal')

def initialize(m,config,condition):
    if (config.inputs,config.outputs,config.fan_in,config.plastic_rows,config.neuron)!=(8,2,8,12,'alif') or config.hidden not in HEIGHTS:
        raise ValueError('Approved capacity layout48/64/128 D8 O2 K8 R12 ALIF')
    if condition not in CONDITIONS:raise ValueError('Unregistered capacity condition')
    config.validate();m.config=config;m.condition=condition;m.sensory_threshold=.6;m.core=SpikingCore(config)
    shape=(config.hidden,config.inputs+config.fan_in)
    for name in ('e_v','e_a','e_f','syn_trace'):setattr(m,name,np.zeros(shape,np.float32))
    m.syn_energy=np.full(shape,1e-4,np.float32)
    m.f_mean=np.zeros(config.hidden,np.float32);m.f_var=np.full(config.hidden,.01,np.float32)
    m.w_in_0=m.core.w_in.copy();m.w_rec_0=m.core.w_rec.copy()
    m.gamma_e_vec=np.linspace(.25,.92,config.hidden,dtype=np.float32)[:,None]
    m.signal_rng=np.random.default_rng(config.seed+35100000)
    m.calibrated=False;m.episodes_learned=0;m.hidden_updates=0;m.total_ticks=0
    m.trace_kind='multi';m.gate_mode='activity'

def decode(cls,payload):
    from .forensic.r8_snapshot import metadata,loads,MAX_BYTES
    if not isinstance(payload,bytes) or len(payload)>MAX_BYTES:raise ValueError('Capacity snapshot bound')
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as z:
            if len(z.namelist())!=len(set(z.namelist())) or sum(i.file_size for i in z.infolist())>MAX_BYTES or z.getinfo('metadata.json').file_size>65536:
                raise ValueError('Capacity snapshot members/bounds')
            meta=metadata(z.read('metadata.json'))
        return loads(cls(Config(**meta['config']),meta['settings']['condition']),payload)[0]
    except (KeyError,TypeError,zipfile.BadZipFile) as exc:raise ValueError('Malformed capacity snapshot') from exc

class CapacityOwn(MatchedLocalCredit):
    def __init__(self,config,condition='learned_mtsg'):initialize(self,config,condition)
    @classmethod
    def loads(cls,payload):return decode(cls,payload)

class CapacityACR(AdaptiveCreditRouting):
    def __init__(self,config,condition='learned_mtsg'):
        initialize(self,config,condition);self.profile='groups_up'
        for name in EXTRA_ARRAYS[:4]:setattr(self,name,np.zeros_like(self.e_f))
        for name in EXTRA_ARRAYS[4:-1]:setattr(self,name,np.zeros(config.hidden,np.float32))
        self.last_delta=np.zeros_like(self.e_f);gamma=self.gamma_e_vec[:,0]
        self.router_bias[:]=4*(gamma-(.25+.92)/2)/(.92-.25)
        self.router_rng=np.random.default_rng(config.seed+43100000)
        self.alpha[:]=self.fixed_alpha()
    @classmethod
    def loads(cls,payload):return decode(cls,payload)
