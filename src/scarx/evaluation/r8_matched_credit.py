"""Isolated matched-controls repair of R8, not an app/serving learner.

Pinned peer forward/local learning are inherited. Only trace-policy control
matching is repaired, before a registered DEV comparison. No autograd.
"""
from pathlib import Path
import importlib.util,subprocess,sys
import numpy as np
from .readout_eligibility import spike_slope

PEER_SHA='607ffca7c9222c5e3fd9741c3ebdf5939e10bedf'
ROOT=Path(__file__).resolve().parents[3]
PEER=ROOT.parent/'review-gemini-r8-607ffca'


def peer_module(filename):
    if filename not in ('r8_mtsg_eprop','r8_bf_cascade','r8_benchmark_integrity'):raise ValueError('Explicit pinned module required')
    if subprocess.check_output(['git','-C',str(PEER),'rev-parse','HEAD'],text=True).strip()!=PEER_SHA:raise ValueError('Peer source moved')
    package='evaluation' if filename=='r8_benchmark_integrity' else 'neural'
    name='scarx.'+package+'._peer_607ffca_'+filename
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,PEER/'src/scarx'/package/(filename+'.py'))
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return sys.modules[name]


Base=peer_module('r8_mtsg_eprop').MultiTimescaleSensoryGatedEProp
ARMS={
    'fast_ungated':('fast','none','learned_mtsg'),
    'slow_ungated':('slow','none','learned_mtsg'),
    'multi_ungated':('multi','none','learned_mtsg'),
    'multi_gated':('multi','activity','learned_mtsg'),
    'multi_gated_frozen':('multi','activity','frozen_hidden'),
    'multi_gated_zero':('multi','activity','zero_signal'),
    'multi_gated_shuffle':('multi','activity','shuffled_signal'),
    'multi_gated_readout_only':('multi','activity','readout_only'),
}


class MatchedLocalCredit(Base):
    def __init__(self,config,arm='multi_gated'):
        if arm not in ARMS:raise ValueError('Unregistered matched arm')
        trace,gate,condition=ARMS[arm]
        super().__init__(config,condition=condition,sensory_threshold=.6)
        self.trace_kind=trace;self.gate_mode=gate
        self.gamma_e_vec[:]=.25 if trace=='fast' else .92 if trace=='slow' else np.linspace(.25,.92,48,dtype=np.float32)[:,None]

    def step(self,x,track=True):
        c=self.config;row=np.asarray(x,dtype=np.float32)
        if row.shape!=(c.inputs,) or not np.isfinite(row).all():raise ValueError('Invalid input vector')
        # All controls compute identical trace policy BEFORE the learning signal.
        if track:
            old_z=self.core.spikes
            theta=c.threshold+c.adaptation_strength*self.core.adaptation
            i_in=self.core.w_in@row;i_rec=np.sum(self.core.w_rec*old_z[self.core.indices],axis=1)
            u_carry=c.leak*self.core.voltage+i_rec;u=u_carry+i_in
            psi=spike_slope(u-theta)
            pre=np.concatenate((np.broadcast_to(row,(c.hidden,c.inputs)),old_z[self.core.indices]),axis=1)
            u_e=c.leak*self.e_v+pre;a_old=self.e_a.copy()
            z_e=psi[:,None]*(u_e-c.adaptation_strength*a_old)
        self.core.step(row,track=False)
        if track:
            self.e_v[:]=u_e-theta[:,None]*z_e-self.core.spikes[:,None]*c.adaptation_strength*a_old
            self.e_a[:]=c.adaptation_decay*a_old+z_e
            z_e_gated=z_e.copy()
            if self.gate_mode=='activity':
                g_rec=1.0 if float(np.max(np.abs(row)))>self.sensory_threshold else .15
                z_e_gated[:,c.inputs:]*=g_rec
            self.e_f[:]=self.gamma_e_vec*self.e_f+(1.0-self.gamma_e_vec)*z_e_gated
        self.total_ticks+=1;return self.core.features

    def dumps(self):
        from .forensic.r8_snapshot import dumps
        return dumps(self)

    @classmethod
    def loads(cls,payload):
        from .forensic.r8_snapshot import metadata,loads,MAX_BYTES
        from ..neural.core import Config
        import io,zipfile
        # Config is bounded to H48/D8/O2 by the adapter before arrays are loaded.
        if not isinstance(payload,bytes) or len(payload)>MAX_BYTES:raise ValueError('Snapshot byte bound')
        with zipfile.ZipFile(io.BytesIO(payload)) as z:
            names=z.namelist()
            if len(names)!=len(set(names)) or sum(i.file_size for i in z.infolist())>MAX_BYTES or z.getinfo('metadata.json').file_size>65536:
                raise ValueError('Snapshot member bound')
            meta=metadata(z.read('metadata.json'))
        config=Config(**meta['config']);settings=meta['settings']
        if (config.hidden,config.inputs,config.outputs,config.fan_in,config.plastic_rows)!=(48,8,2,8,12):raise ValueError('Matched snapshot layout')
        matches=[arm for arm,parts in ARMS.items() if parts==(settings['trace_kind'],settings['gate_mode'],settings['condition'])]
        if len(matches)!=1:raise ValueError('Unregistered matched checkpoint policy')
        return loads(cls(config,matches[0]),payload)[0]
