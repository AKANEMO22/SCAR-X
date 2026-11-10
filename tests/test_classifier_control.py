import numpy as np
import pytest
import importlib.util,os,subprocess,sys
from pathlib import Path


def _exercise_isolated_control():
    from scarx.evaluation.controls import ClassifierLearnabilityControl,torch
    from scarx.neural.core import Config,SpikingCore
    from scarx.neural.local_target import LocalTargetCore
    from scarx.evaluation.readout_conditioning import ConditionedReadout
    torch.set_num_threads(1)
    c=Config(hidden=48,outputs=2,fan_in=8,plastic_rows=12,seed=181)
    base=SpikingCore(c);initial=LocalTargetCore(c)
    head=ConditionedReadout(base.w_out,base.bias,'normalized')
    seq=np.random.default_rng(11).normal(0,.3,(12,8)).astype(np.float32)
    for delta in (0.,.01,.02):base.forward(seq+delta);head.observe(base.features)
    head.freeze();control=ClassifierLearnabilityControl(c,head,True)
    base.forward(seq);np.testing.assert_array_equal(control.features(seq).numpy(),base.features)
    before=control.parameters[0].detach().numpy().copy();control.learn(seq,1)
    np.testing.assert_array_equal(control.parameters[0].detach().numpy(),before)
    clone=ClassifierLearnabilityControl(c,ConditionedReadout.loads(head.dumps()),False)
    old=clone.parameters[0].detach().numpy().copy();loss=clone.learn(seq,0)
    assert np.isfinite(loss) and np.count_nonzero(clone.parameters[0].detach().numpy()!=old)>0
    copied=clone.copied_core()
    assert np.count_nonzero(copied.spikes)==np.count_nonzero(clone.last_state[2].detach().numpy())
    for name in ('w_in','w_rec','w_out','bias'):getattr(initial,name)[:]=getattr(copied,name)
    batch=np.stack((seq,seq+.01,seq+.02))
    np.testing.assert_array_equal(clone.features(batch).detach().numpy(),initial.features_many(batch))
    assert clone.hidden_updates==1 and clone.optimizer_bytes()>0
    for optimizer in ('adam','sgd'):
        probe=ClassifierLearnabilityControl(c,ConditionedReadout.loads(head.dumps()),False,optimizer,'rotating')
        first=probe.parameters[0].detach().numpy().copy();probe.learn(seq,1)
        active=probe.last_rows;inactive=np.setdiff1d(np.arange(48),active)
        np.testing.assert_array_equal(probe.parameters[0].detach().numpy()[inactive],first[inactive])
        assert np.count_nonzero(probe.parameters[0].detach().numpy()[active]!=first[active])>0
        first_rows=probe.parameters[0].detach().numpy()[active].copy()
        saved_moments={k:v[active].detach().numpy().copy() for k,v in probe.optimizer.state[probe.parameters[0]].items() if hasattr(v,'shape') and v.shape==probe.parameters[0].shape}
        probe.learn(seq,0);np.testing.assert_array_equal(probe.parameters[0].detach().numpy()[active],first_rows)
        for key,value in saved_moments.items():np.testing.assert_array_equal(probe.optimizer.state[probe.parameters[0]][key][active].detach().numpy(),value)


def test_isolated_control_same_forward_output_local_head_and_hidden_freeze():
    if importlib.util.find_spec('torch') is None:pytest.skip('Optional isolated control dependency')
    # Never leave Torch libraries/allocator in the process that measures the
    # <=2GB serving runtime. The production RAM guard stays unchanged.
    result=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--isolated'],capture_output=True,text=True,
        timeout=30,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    assert result.returncode==0,result.stdout+result.stderr


if __name__=='__main__':
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
    _exercise_isolated_control()
