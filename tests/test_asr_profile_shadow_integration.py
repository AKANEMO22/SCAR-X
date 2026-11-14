"""Shadow modules only: configuration/command contracts, no real model workers."""

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
from scarx.runtime.asr_profiles import PROFILES

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def shadow():
    # This explicitly tests the archived preparation candidate, not whatever
    # serving code might later be installed. All fixture modules are tracked.
    folder=ROOT/'runs/20261114T084052Z-ASR-profile-resolver-p0-v1-791db8/shadow-preparation-v1/generated-shadow'
    loaded={}
    for filename in ('config','backends','qualification','gateway'):
        path=folder/(filename+'.py')
        if not path.is_file():pytest.fail('Tracked ASR preparation module is missing')
        name='scarx.hybrid._ASR_shadow_'+filename
        spec=importlib.util.spec_from_file_location(name,path)
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
        loaded[filename]=module
    return SimpleNamespace(**loaded)


def config(shadow,name='openai/whisper-small',mode='gpu'):
    return shadow.config.HybridConfig(execution_mode=mode,asr_model_name=name,asr_revision=PROFILES[name].revision)


def test_shadow_config_loads_real_selected_identity_and_rejects_mismatch(shadow,tmp_path):
    default=shadow.config.load_runtime_config(tmp_path)
    assert default.asr_model_name=='openai/whisper-base' and default.execution_mode=='cpu'
    value={'execution_mode':'gpu','asr_model_name':'openai/whisper-small',
           'asr_revision':PROFILES['openai/whisper-small'].revision}
    (tmp_path/'config.json').write_text(json.dumps(value))
    selected=shadow.config.load_runtime_config(tmp_path)
    assert selected.asr_model_name=='openai/whisper-small'
    value['asr_revision']=PROFILES['openai/whisper-base'].revision
    (tmp_path/'config.json').write_text(json.dumps(value))
    with pytest.raises(ValueError):shadow.config.load_runtime_config(tmp_path)
    with pytest.raises(ValueError):config(shadow,mode='cpu')


def test_small_gpu_pressure_fails_before_unloading_any_vlm(shadow,tmp_path,monkeypatch):
    backend=shadow.backends.NativeBackends(tmp_path,SimpleNamespace(),SimpleNamespace(),config(shadow))
    terminated=[]
    backend.vlm=SimpleNamespace(poll=lambda:None,terminate=lambda:terminated.append(True))
    monkeypatch.setattr(shadow.backends,'gpu_headroom',lambda _:dict(free_MiB=1799))
    with pytest.raises(MemoryError):backend.pause_vlm_for_audio()
    assert not terminated and backend.vlm is not None
    monkeypatch.setattr(shadow.backends,'gpu_headroom',lambda _:dict(free_MiB=1800))
    backend.pause_vlm_for_audio()
    assert not terminated


def test_selected_ASR_command_and_result_identity_use_small_file_not_base(shadow,tmp_path,monkeypatch):
    cfg=config(shadow);events=[];commands=[]
    backend=shadow.backends.NativeBackends(tmp_path,SimpleNamespace(event=lambda *x:events.append(x)),
        SimpleNamespace(check=lambda:None),cfg)
    wav=tmp_path/'owned.wav';wav.write_bytes(b'owned transport fixture, not realaudio')
    small=tmp_path/'candidates/whisper-small/whisper-small-f32.bin'
    verified=[]
    def integrity():
        verified.append(True);backend.asr_storage_receipt={'status':'fixture_transport_only'};return small
    backend.ensure_asr_storage=integrity
    monkeypatch.setattr(shadow.backends,'gpu_headroom',lambda _:dict(free_MiB=2000))
    emit_actual_gpu=[True]
    def spawn(command,logname):
        commands.append(command)
        output=Path(command[command.index('-of')+1]).with_suffix('.json')
        output.write_text(json.dumps({'transcription':[{'text':'Owned transportfixture text'}]}))
        with (tmp_path/logname).open('a') as log:
            log.write('whisper_model_load: ftype         = 0\nwhisper_init_with_params_no_state: use gpu    = '+('1' if emit_actual_gpu[0] else '0')+'\n')
            if emit_actual_gpu[0]:log.write('whisper_backend_init_gpu: using CUDA0 backend\n')
        return SimpleNamespace(wait=lambda timeout:0,poll=lambda:0)
    backend._spawn=spawn
    assert backend.transcribe(wav)=='Owned transportfixture text'
    command=commands[0]
    assert Path(command[command.index('-m')+1])==small
    assert command[command.index('-dev')+1]=='0' and '-ng' not in command
    assert verified and events[-1][0]=='asr_result'
    assert events[-1][1]['ASR_identity']['name']=='openai/whisper-small'
    assert events[-1][1]['model_path']==str(small)
    assert events[-1][1]['storage_verified'] is True
    assert events[-1][1]['execution_device_log_verified'] is True
    emit_actual_gpu[0]=False
    with pytest.raises(ValueError,match='CUDA'):backend.transcribe(wav)
    assert len(commands)==2 and sum(e[0]=='asr_result' for e in events)==1


def test_waiting_ASR_yields_on_foreground_and_sampled_GPU_pressure(shadow,tmp_path,monkeypatch):
    backend=shadow.backends.NativeBackends(tmp_path,SimpleNamespace(),SimpleNamespace(check=lambda:None),config(shadow))
    backend.asr=SimpleNamespace(poll=lambda:None)
    monkeypatch.setattr(shadow.backends,'gpu_headroom',lambda _:dict(free_MiB=511))
    with pytest.raises(MemoryError,match='512'):backend.wait_asr()
    monkeypatch.setattr(shadow.backends,'gpu_headroom',lambda _:dict(free_MiB=2000))
    backend.cancel_idle.set()
    with pytest.raises(InterruptedError,match='Foreground'):backend.wait_asr()


def test_storage_tamper_rechecks_and_clears_old_verified_receipt(shadow,tmp_path,monkeypatch):
    cfg=config(shadow);backend=shadow.backends.NativeBackends(tmp_path,SimpleNamespace(),SimpleNamespace(),cfg)
    path=backend.asr_model_path();path.parent.mkdir(parents=True);path.write_bytes(b'fake storagefixture')
    verified=[]
    def check(directory,name,revision,mode):
        verified.append(name);return {'status':'fixture only, no native integrity asserted'}
    monkeypatch.setattr(shadow.backends,'verify_asr_storage',check)
    assert backend.ensure_asr_storage()==path
    assert backend.ensure_asr_storage()==path and len(verified)==1
    path.write_bytes(b'changed-size fake storagefixture')
    def reject(*args):raise ValueError('Fixture checksum failure')
    monkeypatch.setattr(shadow.backends,'verify_asr_storage',reject)
    with pytest.raises(ValueError):backend.ensure_asr_storage()
    assert backend.asr_storage_receipt is None


def test_gateway_reports_selected_identity_without_claiming_unverified_storage(shadow):
    cfg=config(shadow)
    gateway=SimpleNamespace(engine=SimpleNamespace(status=lambda:{},model=SimpleNamespace(device='gpu')),ready=False,failure=None,
        store=SimpleNamespace(get=lambda *x:None,events=lambda *x:[]),port=8775,config=cfg,
        backend=SimpleNamespace(asr_storage_receipt=None))
    status=shadow.gateway.Gateway.status(gateway)
    assert status['ASR_identity']==dict(name='openai/whisper-small',revision=cfg.asr_revision,
        precision='originalFP32',storage_verified=False)
    assert status['execution_placement']['ASR']=='gpu'
