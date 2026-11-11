import numpy as np
import pytest
from scarx.evaluation.forensic.event_oracle import decode_events,pair_contract,EventOracleError,pulse_tensor,move_known_query_tensor


def overwritten_pair():
    x=np.zeros((2,12,8),np.float32)
    x[:,2,[0,2]]=1.5  # A=0
    x[:,5,[1,3]]=1.5  # B=1
    x[:,8,[0,3]]=1.5  # A becomes1
    x[0,10,4]=1.5;x[1,10,5]=1.5
    return x


def test_latest_write_oracle_does_not_read_the_first_value_or_a_label_metadata():
    x=overwritten_pair();d=decode_events(x,expected_writes=3)
    np.testing.assert_array_equal(d['label'],[1,1])
    np.testing.assert_array_equal(d['first_written_value'],[0,0])
    assert pair_contract(x,d)['same_nonquery_noise_pairs']==1


def test_background_structured_write_cannot_overwrite_a_key():
    x=overwritten_pair();x[:,8,:4]=0;x[:,8,6:8]=1.5
    d=decode_events(x,expected_writes=2)
    np.testing.assert_array_equal(d['label'],[0,1])
    np.testing.assert_array_equal(d['vA'],[0,0])


def test_wrong_write_count_and_extra_query_are_detected():
    x=overwritten_pair()
    with pytest.raises(EventOracleError,match='marker count'):decode_events(x,expected_writes=2)
    x[0,9,4]=1.5
    with pytest.raises(EventOracleError,match='marker count'):decode_events(x,expected_writes=3)


def test_valid_individual_labels_do_not_prove_same_noise_counterfactual():
    x=overwritten_pair();x[1,0,7]=1.5
    d=decode_events(x,expected_writes=3)
    np.testing.assert_array_equal(d['label'],[1,1])
    assert pair_contract(x,d)['different_nonquery_noise_pairs']==1


def test_known_background_removal_is_not_a_general_raw_noisy_oracle():
    x=overwritten_pair();noise=np.zeros_like(x);noise[:,0,0]=.8
    noisy=x+noise
    with pytest.raises(EventOracleError,match='marker count'):decode_events(noisy,expected_writes=3)
    np.testing.assert_array_equal(decode_events(noisy-noise,expected_writes=3)['label'],[1,1])


def test_moved_query_background_sum_is_not_a_second_latent_query():
    x=overwritten_pair();d=decode_events(x,expected_writes=3)
    x[0,10,5]=.32;x[0,11,5]=.32
    moved=move_known_query_tensor(x,d)
    with pytest.raises(EventOracleError,match='marker count'):decode_events(moved,expected_writes=3)
    latent=move_known_query_tensor(pulse_tensor(d,x.shape),d)
    truth=decode_events(latent,expected_writes=3)
    np.testing.assert_array_equal(truth['label'],d['label'])
    np.testing.assert_array_equal(truth['q'],[0,1])
