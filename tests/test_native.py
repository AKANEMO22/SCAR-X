from pathlib import Path
import os
import numpy as np
import pytest
from scarx import Config, SpikingCore
from scarx.runtime.native import NativeCore
from scarx.evaluation.tasks import episodes

LIB = Path(__file__).resolve().parents[1]/"build"/("scarx_cpu.dll" if os.name=="nt" else "libscarx_cpu.so")


@pytest.mark.skipif(not LIB.exists(),reason="Build native kernel with scripts/build_native.py")
@pytest.mark.parametrize("mode",["frozen","readout","stdp","full"])
def test_native_learning_parity(mode):
    config = Config(inputs=5,hidden=48,plastic_rows=12)
    a,b = SpikingCore(config),NativeCore(config,LIB)
    for x,y in episodes(101,64):
        pa,pb = a.forward(x,track=True),b.forward(x,track=True)
        np.testing.assert_allclose(pa,pb,atol=3e-5,rtol=3e-5)
        la,lb = a.learn(y,mode),b.learn(y,mode)
        assert abs(la-lb)<5e-5
    for name in a.weight_names:
        np.testing.assert_allclose(getattr(a,name),getattr(b,name),atol=3e-5,rtol=3e-5)
