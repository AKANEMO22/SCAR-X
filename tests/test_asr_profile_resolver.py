import pytest
from scarx.runtime.asr_profiles import PROFILES,selected_profile,model_path,validate_receipt,verify_asr_storage,verify_execution_log


def receipt(profile):
    entry={'bytes':profile.native_bytes,'sha256':profile.native_sha256}
    if profile.name.endswith('base'):
        return {'status':'NATIVE_STORAGE_INTEGRITY_PASS','quantized':False,
            'source':{'models/whisper-base/model.safetensors':{'sha256':profile.source_sha256}},'exports':{
            'whisper-base-f32.bin':dict(entry,dtype='F32',direct_exact_tensor_readback_count=245)}}
    return {'status':'NATIVE_FP32_STORAGE_READY_INFERENCE_PENDING','source_revision':profile.revision,
        'source_sha256':profile.source_sha256,'tensor_count':479,'quality_qualified':False,
        'export':dict(entry,precision='originalFP32',every_tensor_exact_readback=True)}


def test_named_revision_selects_only_qualified_device_and_preserves_distinct_files(tmp_path):
    base,small=PROFILES['openai/whisper-base'],PROFILES['openai/whisper-small']
    assert selected_profile(base.name,base.revision,'cpu') is base
    assert selected_profile(small.name,small.revision,'gpu') is small
    assert model_path(tmp_path,base)!=model_path(tmp_path,small)
    with pytest.raises(ValueError):selected_profile(small.name,small.revision,'cpu')
    with pytest.raises(ValueError):selected_profile(base.name,small.revision,'gpu')
    with pytest.raises(ValueError):selected_profile('../unregistered','unknown','gpu')
    with pytest.raises(ValueError):selected_profile(True,small.revision,'gpu')


@pytest.mark.parametrize('name',tuple(PROFILES))
def test_correct_receipt_is_storage_evidence_and_never_an_implied_quality_pass(name):
    profile=PROFILES[name];value=receipt(profile)
    assert validate_receipt(profile,value)['quality_qualification_implied'] is False
    value['quality_qualified']=True
    assert validate_receipt(profile,value)['quality_qualification_implied'] is False


@pytest.mark.parametrize('field,value',(('sha256','0'*64),('bytes',True),('precision','FP16'),
    ('every_tensor_exact_readback',False)))
def test_tampered_small_export_metadata_is_rejected(field,value):
    profile=PROFILES['openai/whisper-small'];bad=receipt(profile);bad['export'][field]=value
    with pytest.raises(ValueError):validate_receipt(profile,bad)


def test_missing_or_short_selected_model_cannot_be_loaded_or_silently_replaced_by_base(tmp_path):
    profile=PROFILES['openai/whisper-small']
    with pytest.raises(ValueError):verify_asr_storage(tmp_path,profile.name,profile.revision,'gpu')
    import json
    target=model_path(tmp_path,profile);target.parent.mkdir(parents=True)
    (target.parent/'native-storage-receipt.json').write_text(json.dumps(receipt(profile)))
    target.write_bytes(b'invalid tiny fixture, not model weights')
    with pytest.raises(ValueError,match='byte-size'):verify_asr_storage(tmp_path,profile.name,profile.revision,'gpu')


@pytest.mark.parametrize('bad',([],None,{'export':[]},{'export':True}))
def test_malformed_nested_receipt_does_not_become_a_valid_profile(bad):
    profile=PROFILES['openai/whisper-small']
    value=bad if not isinstance(bad,dict) else dict(receipt(profile),**bad)
    with pytest.raises(ValueError):validate_receipt(profile,value)


def test_whitelisted_path_cannot_escape_owned_root_through_a_directory_link(tmp_path):
    root=tmp_path/'owned';root.mkdir();(root/'candidates').mkdir()
    outside=tmp_path/'outside';outside.mkdir()
    try:(root/'candidates/whisper-small').symlink_to(outside,target_is_directory=True)
    except OSError as exc:pytest.skip('Owned directory symlink is unavailable: '+str(exc))
    with pytest.raises(ValueError,match='escapes'):model_path(root,PROFILES['openai/whisper-small'])


def test_execution_log_requires_actual_cuda_not_just_a_gpu_requested_flag():
    log='whisper_model_load: ftype         = 0\nwhisper_init_with_params_no_state: use gpu    = 1\n'
    with pytest.raises(ValueError,match='CUDA'):verify_execution_log(log,'gpu',0)
    assert verify_execution_log(log+'whisper_backend_init_gpu: using CUDA0 backend\n','gpu',0)['actual_neural_device']=='CUDA0'
    with pytest.raises(ValueError):verify_execution_log(log+'whisper_backend_init_gpu: using CUDA1 backend\n','gpu',0)
    with pytest.raises(ValueError):verify_execution_log(log.replace('= 0','= 1')+'whisper_backend_init_gpu: using CUDA0 backend\n','gpu',0)
    assert verify_execution_log(log.replace('use gpu    = 1','use gpu    = 0'),'cpu',0)['actual_neural_device']=='CPU'
