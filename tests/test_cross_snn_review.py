import pytest
from scarx.evaluation.cross_snn_review import paired_distractor,formula_counterexamples


def test_pairing_uses_same_diagnostic_baseline_not_older_campaign():
    rows={str(seed):{'std_ew_he':{'direct':.6,'probe':.7},'delay_eligibility_suppressed':{'direct':.65,'probe':.75}}
          for seed in (1,2,3)}
    result=paired_distractor(rows)
    assert result['means']['probe_effect_pp']==pytest.approx(5.)
    assert result['paired_probe_interval_pp']['lower95']==pytest.approx(5.)
    assert result['training_runs_opened']==0


def test_one_losing_seed_is_preserved_in_cluster_uncertainty():
    rows={str(seed):{'std_ew_he':{'direct':.6,'probe':.7},'delay_eligibility_suppressed':{'direct':.6,'probe':v}}
          for seed,v in enumerate((.75,.8,.6))}
    result=paired_distractor(rows)
    assert result['seeds_with_negative_probe_effect']==1
    assert result['paired_probe_interval_pp']['lower95']<0


def test_invalid_accuracy_and_seed_count_fail_closed():
    with pytest.raises(ValueError):paired_distractor({})
    rows={str(i):{'std_ew_he':{'direct':.6,'probe':float('nan')},'delay_eligibility_suppressed':{'direct':.6,'probe':.8}} for i in range(3)}
    with pytest.raises(ValueError):paired_distractor(rows)


def test_no_gate_does_not_equal_frozen_eligibility():
    x=formula_counterexamples()['zero_gate']
    assert x['result']==.25 and not x['freezes_old_trace']


def test_noise_can_open_amplitude_gate_without_event_oracle():
    x=formula_counterexamples()['amplitude_gate_fixture']
    assert x['noise_opens_gate'] and x['distractor_opens_gate']


def test_time_suppression_also_overlaps_a_valid_binding_event():
    assert formula_counterexamples()['time_gate_overlap']['suppressed_valid_event_ticks']==[4,5]


def test_diffusion_assignments_use_different_old_and_new_gaps():
    x=formula_counterexamples()['diffusion_assignment']
    assert not x['same_update']
    assert x['simultaneous_weighted_mass']==pytest.approx(1)
    assert x['sequential_weighted_mass']<1


def test_array_budget_scales_with_real_input_width():
    x=formula_counterexamples()['array_bytes_only']
    assert x['gamma_48_float32']==192
    assert x['one_hidden_slow_state_H48_D8_K8']==3072
    assert x['one_hidden_slow_state_public_G1_D222']==44160
    assert x['one_hidden_slow_state_codec_V55_D253']==50112
