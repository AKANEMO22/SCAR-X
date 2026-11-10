import numpy as np
import pytest
from scarx.evaluation.rgb_grounding import different_class_pairing


def test_image_negative_pairing_changes_every_class_and_is_a_permutation():
    for y in (np.repeat(np.arange(10),16),np.array([2,0,1,2,0,1])):
        order=different_class_pairing(y)
        assert np.all(y[order]!=y)
        np.testing.assert_array_equal(np.sort(order),np.arange(len(y)))
    with pytest.raises(ValueError):different_class_pairing([0,0,1])
    with pytest.raises(ValueError):different_class_pairing([0,0])
