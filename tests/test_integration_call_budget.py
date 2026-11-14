import json
import pytest
from scarx.evaluation.integration_call_budget import charge


def test_budget_keeps_ASR_and_startup_VLM_reservations_separate_and_bounded(tmp_path):
    path=tmp_path/'calls.json'
    assert charge(path,'ASR')==1
    assert charge(path,'VLM',3)==3
    assert charge(path,'VLM')==4
    with pytest.raises(RuntimeError):charge(path,'VLM')
    assert charge(path,'ASR',3)==4
    with pytest.raises(RuntimeError):charge(path,'ASR')
    assert json.loads(path.read_text())['attempted']=={'ASR':4,'VLM':4}
    assert not path.with_suffix('.lock').exists()


def test_corrupt_or_locked_budget_is_never_reset(tmp_path):
    path=tmp_path/'calls.json';path.write_text(json.dumps({'limits':{'ASR':4,'VLM':4},'attempted':{'ASR':True,'VLM':0}}))
    with pytest.raises(ValueError):charge(path,'ASR')
    assert json.loads(path.read_text())['attempted']['ASR'] is True
    path.with_suffix('.lock').write_text('owned other active fixture')
    with pytest.raises(FileExistsError):charge(path,'ASR')
    assert path.with_suffix('.lock').read_text()=='owned other active fixture'
