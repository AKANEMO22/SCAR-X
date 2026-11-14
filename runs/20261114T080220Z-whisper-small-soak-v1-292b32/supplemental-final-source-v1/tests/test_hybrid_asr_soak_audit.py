import pytest
from scarx.evaluation.asr_soak_audit import measured_gate


def call():return dict(audio_seconds=20.,seconds=2.,RTF=.1,native_CUDA_FP32_verified=True)


def test_component_profile_requires_elapsed_coverage_and_active_resource_samples():
    calls=[call() for _ in range(80)]
    samples=[dict(free_MiB=1024,ASR_active=True)]
    assert measured_gate(1800,1800,calls,samples,[])['gate']
    assert not measured_gate(1800,30,calls,samples,[])['gate']
    assert not measured_gate(1800,1800,calls,[dict(free_MiB=1024,ASR_active=False)],[])['gate']
    assert not measured_gate(1800,1800,calls,[dict(free_MiB=511,ASR_active=True)],[])['gate']
    assert not measured_gate(1800,1800,calls,samples,['actualerror'])['gate']


@pytest.mark.parametrize('value',(float('nan'),float('inf'),-1,True))
def test_nonfinite_or_forged_timing_cannot_pass(value):
    with pytest.raises(ValueError):measured_gate(30,value,[call()],[dict(free_MiB=1024,ASR_active=True)],[])


def test_repeat_count_is_not_new_quality_samples_and_bad_rtf_is_detected():
    result=measured_gate(30,30,[call()]*3,[dict(free_MiB=1024,ASR_active=True)],[])
    assert result['gate'] and result['new_independent_quality_samples']==0
    with pytest.raises(ValueError):measured_gate(30,30,[dict(call(),RTF=.2)],[dict(free_MiB=1024,ASR_active=True)],[])
