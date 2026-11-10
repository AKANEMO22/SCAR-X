import numpy as np
import pytest
from scarx.neural.unified import UnifiedCore
from scarx.evaluation.parameter_interventions import PARTS,from_payloads,transplant_arrays


@pytest.mark.parametrize('parts',[(),('w_out',),('bias',),('w_in','w_rec'),('w_out','bias'),PARTS])
def test_selected_afterarrays_and_every_unselected_beforearray_state_exact(parts):
    a=UnifiedCore(['<eos>','<bos>','hello','.']);b=UnifiedCore.loads(a.dumps())
    for i,name in enumerate(PARTS):getattr(b.core,name)[:]+=.01*(i+1)
    before=a.dumps();after=b.dumps();out=from_payloads(before,after,parts)
    for name,arr in a.core.arrays().items():np.testing.assert_array_equal(getattr(out.core,name),getattr(b.core,name) if name in parts else arr)
    assert a.dumps()==before and b.dumps()==after


def test_transplant_rejects_unknown_shapes_and_topology_before_mutation():
    a=UnifiedCore(['<eos>','<bos>','hello','.']);before=a.dumps()
    with pytest.raises(ValueError):transplant_arrays(a,{'w_out':a.core.w_out+.01,'bias':np.zeros(1,np.float32)},('w_out','bias'))
    assert a.dumps()==before
    with pytest.raises(ValueError):transplant_arrays(a,{},('voltage',))
    b=UnifiedCore.loads(before);b.core.indices[0]=b.core.indices[0][::-1]
    with pytest.raises(ValueError):from_payloads(before,b.dumps(),('w_out',))
