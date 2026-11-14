import json
from types import SimpleNamespace
import pytest
from scarx.evaluation.hermes_resource_roots import verified_gateway_pid


def process(home,pid,created=100.25):
    return SimpleNamespace(pid=pid,create_time=lambda:created,is_running=lambda:True,
        exe=lambda:str(home/'tools/python.exe'))


def descriptor(home,pid=7,stamp=10025,kind='hermes-gateway'):
    (home/'gateway.pid').write_text(json.dumps({'pid':pid,'start_time':stamp,'kind':kind}),encoding='utf-8')


def test_descriptor_refresh_follows_new_gateway_without_adopting_stale_reused_pid(tmp_path):
    descriptor(tmp_path)
    assert verified_gateway_pid(tmp_path,lambda pid:process(tmp_path,pid))==7
    assert verified_gateway_pid(tmp_path,lambda pid:process(tmp_path,pid,created=101.25)) is None
    descriptor(tmp_path,pid=9,stamp=10125)
    assert verified_gateway_pid(tmp_path,lambda pid:process(tmp_path,pid,created=101.25))==9


@pytest.mark.parametrize('field,value',(('pid',True),('pid',-1),('start_time',float('nan')),('kind','other')))
def test_invalid_descriptor_fields_cannot_select_a_root(tmp_path,field,value):
    descriptor(tmp_path);path=tmp_path/'gateway.pid';row=json.loads(path.read_text());row[field]=value
    path.write_text(json.dumps(row))
    assert verified_gateway_pid(tmp_path,lambda pid:process(tmp_path,pid)) is None


def test_outside_home_process_is_not_claimed_as_the_gateway(tmp_path):
    descriptor(tmp_path)
    bad=process(tmp_path,7);bad.exe=lambda:str(tmp_path.parent/'outside/python.exe')
    assert verified_gateway_pid(tmp_path,lambda pid:bad) is None
