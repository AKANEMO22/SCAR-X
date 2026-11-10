import numpy as np
from scarx.evaluation.credit_alignment import permuted


def test_permuting_direction_does_not_guarantee_matching_adam_step_with_synapse_specific_history():
    direction=np.array([[1.,0.,0.,0.]],np.float32);control=permuted(direction,np.random.default_rng(1))
    # Find a nonidentity permutation deterministically, without any modelquality.
    for seed in range(20):
        control=permuted(direction,np.random.default_rng(seed))
        if not np.array_equal(control,direction):break
    variance=np.array([[1.,100.,10000.,1000000.]],np.float32)
    def step(g):return (.1*g)/np.sqrt(.999*variance+.001*g*g)
    np.testing.assert_array_equal(np.sort(direction.ravel()),np.sort(control.ravel()))
    assert not np.isclose(np.linalg.norm(step(direction)),np.linalg.norm(step(control)))
