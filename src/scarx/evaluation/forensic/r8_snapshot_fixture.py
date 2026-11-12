"""Pinned technical fixtures only, no benchmark or algorithm selection."""
from pathlib import Path
import hashlib,importlib.util,subprocess,sys
import numpy as np
from . import r8_snapshot as S
from ...neural.core import Config
from ..research_workspace import peer_checkout

ROOT=Path(__file__).resolve().parents[4]
PEER=peer_checkout(ROOT,'review-gemini-r8-fe472b3')


def fixture_model(kind):
    if kind in ('matched','stopped','peer607_mtsg','peer607_bf'):
        from ..r8_matched_credit import MatchedLocalCredit,peer_module
        from ..r8_stopped_reset import StoppedResetCredit
        cfg=Config(inputs=8,outputs=2,hidden=48,fan_in=8,plastic_rows=12,seed=42)
        if kind=='matched':model=MatchedLocalCredit(cfg,'multi_gated_shuffle')
        elif kind=='stopped':model=StoppedResetCredit(cfg,'multi_gated_shuffle')
        elif kind=='peer607_mtsg':model=peer_module('r8_mtsg_eprop').MultiTimescaleSensoryGatedEProp(cfg,'shuffled_signal',sensory_threshold=.123)
        else:model=peer_module('r8_bf_cascade').BennaFusiMetaplasticCascade(cfg,use_context_gating=True,kappa1=.003)
        model.calibrated=True
        if hasattr(model,'signal_rng'):model.signal_rng.permutation(48)
        model.core.spikes[:]=1;model.step(np.full(8,.03,np.float32),True)
        return model
    if kind not in ('mtsg','bf'):raise ValueError('Explicit fixture kind required')
    if subprocess.check_output(['git','-C',str(PEER),'rev-parse','HEAD'],text=True).strip()!=S.SOURCE:
        raise ValueError('Pinned fixture source moved')
    filename='r8_mtsg_eprop' if kind=='mtsg' else 'r8_bf_cascade';module_name='scarx.neural._peer_fe472b3_'+filename
    if module_name not in sys.modules:
        spec=importlib.util.spec_from_file_location(module_name,PEER/'src/scarx/neural'/(filename+'.py'))
        module=importlib.util.module_from_spec(spec);sys.modules[module_name]=module;spec.loader.exec_module(module)
    module=sys.modules[module_name];cfg=Config(inputs=8,outputs=2,hidden=48,fan_in=8,plastic_rows=12,seed=42)
    if kind=='mtsg':model=module.MultiTimescaleSensoryGatedEProp(cfg,'shuffled_signal',sensory_threshold=.123);model.signal_rng.permutation(48)
    else:model=module.BennaFusiMetaplasticCascade(cfg,use_context_gating=True,kappa1=.003,kappa2=.0007,sensory_threshold=.123)
    model.calibrated=True;model.core.spikes[:]=1;model.step(np.full(8,.03,np.float32),True)
    return model


def continuation(model,controller):
    model.step(np.full(8,.031,np.float32),True)
    forward_state=digest_state(model,controller)
    seq=np.full((3,8),.03,np.float32);seq[0,0]=1.5
    loss=model.learn_episode(seq,1)
    rng=np.random.default_rng();rng.bit_generator.state=controller['rng'];draw=int(rng.integers(10000))
    controller['rng']=rng.bit_generator.state;controller['position']+=1
    return {'next_forward_state_sha256':forward_state,'next_fixture_loss':loss,'controller_rng_draw':draw,
            'state_sha256':digest_state(model,controller),'hidden_updates':model.hidden_updates,
            'scope':'One technical fixture update, not acquisition/retention or a live session'}


def digest_state(model,controller):return hashlib.sha256(S.dumps(model,controller)).hexdigest()
