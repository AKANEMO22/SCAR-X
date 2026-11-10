import json
import subprocess
import sys
import numpy as np
import pytest
from scarx import Config, SpikingCore
from scarx.evaluation.tasks import episodes, accuracy


def test_local_updates_and_frozen_control():
    m = SpikingCore(Config(inputs=5))
    frozen = m.fingerprint()
    x,y = next(episodes(1,1))
    m.forward(x,track=True)
    before = m.w_in.copy()
    m.learn(y,mode="frozen")
    assert m.fingerprint() == frozen
    m.learn(y,mode="full",reward=1.)
    assert np.any(m.w_in != before)
    changed = set(np.where(np.any(m.w_in != before,axis=1))[0])
    assert changed.issubset(set(m.rows))
    assert m.fingerprint() != frozen


def test_output_error_does_not_backpropagate():
    a,b = SpikingCore(Config(inputs=5)),SpikingCore(Config(inputs=5))
    x,_ = next(episodes(1,1))
    for m in (a,b): m.forward(x,track=True)
    a.learn(0,reward=.5); b.learn(1,reward=.5)
    np.testing.assert_array_equal(a.w_in,b.w_in)
    np.testing.assert_array_equal(a.w_rec,b.w_rec)
    assert not np.array_equal(a.w_out,b.w_out)


def test_checkpoint_exact_resume(tmp_path):
    a = SpikingCore(Config(inputs=5))
    samples = list(episodes(1,12))
    for x,y in samples[:6]: a.forward(x,track=True); a.learn(y)
    p = tmp_path/"state.npz"; a.save(p)
    b = SpikingCore.load(p)
    for n,v in a.arrays().items(): np.testing.assert_array_equal(v,b.arrays()[n])
    for x,y in samples[6:]:
        for m in (a,b): m.forward(x,track=True); m.learn(y)
    assert a.fingerprint() == b.fingerprint()
    assert a.reward_mean == b.reward_mean and a.updates == b.updates


def test_bounded_state_and_config():
    m = SpikingCore(); n = m.memory_bytes()
    for _ in range(1000): m.step(np.zeros(8,np.float32))
    assert m.memory_bytes() == n
    with pytest.raises(MemoryError): SpikingCore(Config(hidden=100000,inputs=100000))
    with pytest.raises(ValueError): SpikingCore(Config(fan_in=100))


def test_core_has_no_autograd_import():
    from pathlib import Path
    import scarx.neural.core as module
    import ast
    tree = ast.parse(Path(module.__file__).read_text())
    imports = [n.module or "" for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
    imports += [a.name for n in ast.walk(tree) if isinstance(n,ast.Import) for a in n.names]
    assert not any(x.startswith(("torch","jax","tensorflow")) for x in imports)

