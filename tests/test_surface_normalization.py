import numpy as np
import pytest
from scarx.neural.unified import UnifiedCore


def test_surface_normalization_is_optional_preserves_internal_punctuation_and_checkpoint():
    vocabulary=['<eos>','<bos>','hello','.'];old=UnifiedCore(vocabulary);model=UnifiedCore(vocabulary,encoding={'kind':'legacy','surface_normalization':'terminal-punctuation-v1'})
    assert not np.array_equal(old.context('hi'),old.context('hi?'))
    np.testing.assert_array_equal(model.context('hi'),model.context('Hi...?!'))
    np.testing.assert_array_equal(model.context('3.14'),model.context('3.14?'))
    assert not np.array_equal(model.context('3.14'),model.context('314'))
    assert np.linalg.norm(model.context('???'))>0
    before=model.dumps();clone=UnifiedCore.loads(before);assert clone.dumps()==before
    assert clone.encoding['surface_normalization']=='terminal-punctuation-v1'
    a=UnifiedCore.loads(before);b=UnifiedCore.loads(before)
    assert a.generate('hi?')==b.generate('hi')
    assert model.core.fingerprint()==old.core.fingerprint()
    with pytest.raises(ValueError):UnifiedCore(vocabulary,encoding={'kind':'legacy','surface_normalization':'unknown'})
