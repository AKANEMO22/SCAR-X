"""Pure malformed-checkpoint tests, never CPU training or model allocation."""
import io
import json
from unittest.mock import patch
import numpy as np
import pytest
from scarx.evaluation.semantic_metaplastic_alif import ProtectedALIF, FORMAT


def test_CPU_rejected_before_allocation():
    with patch('scarx.evaluation.semantic_metaplastic_alif.CreditALIF', side_effect=AssertionError('No backend allocation')):
        with pytest.raises(ValueError):
            ProtectedALIF(260113, 'protected', device='cpu')
    with pytest.raises(ValueError):
        ProtectedALIF.loads(b'anything', device='cpu')


@pytest.mark.parametrize('format,rule,dtype', [(FORMAT, 'wrong', np.uint8), ('old', 'protected', np.uint8), (FORMAT, 'protected', np.float32)])
def test_wrong_rule_or_dtype_cannot_load_GPU(format, rule, dtype):
    out = io.BytesIO()
    np.savez(out, inner=np.zeros(2, dtype=dtype),
             metadata=np.frombuffer(json.dumps({'format': format, 'rule': rule}).encode(), dtype=np.uint8))
    with patch('scarx.evaluation.semantic_metaplastic_alif.CreditALIF.loads', side_effect=AssertionError('No restore')):
        with pytest.raises(ValueError):
            ProtectedALIF.loads(out.getvalue())
