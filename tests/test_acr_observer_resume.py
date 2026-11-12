"""Regression for the stale-PID failure, executing the real safety function AST."""
import ast,json
from pathlib import Path
from types import SimpleNamespace
import psutil
import pytest

ROOT=Path(__file__).resolve().parents[1]

def function(filename,tmp_path,process_factory):
    tree=ast.parse((ROOT/'scripts'/filename).read_text(encoding='utf-8'))
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    safety=next(n for n in main.body if isinstance(n,ast.FunctionDef) and n.name=='safety')
    calls=[];monitor=SimpleNamespace(observer_identity=None,check=lambda:calls.append('checked'))
    scope={'ROOT':tmp_path,'monitor':monitor,'json':json,'Path':Path,'deadline':100,
       'time':SimpleNamespace(perf_counter=lambda:0),
       'psutil':SimpleNamespace(Process=process_factory,NoSuchProcess=psutil.NoSuchProcess)}
    exec(compile(ast.Module(body=[safety],type_ignores=[]),filename,'exec'),scope)
    return scope['safety'],monitor,calls

def owner(tmp_path):
    p=tmp_path/'local/acr-observer-owner.json';p.parent.mkdir();p.write_text(json.dumps({'pid':42}),encoding='utf-8')

def test_missing_optional_observer_no_longer_aborts_preflight(tmp_path):
    owner(tmp_path)
    def missing(pid):raise psutil.NoSuchProcess(pid)
    old,_,_=function('run_acr_resume.py',tmp_path,missing)
    with pytest.raises(psutil.NoSuchProcess):old()
    new,m,calls=function('run_acr_resume_v2.py',tmp_path,missing);new()
    assert m.observer_identity is None and calls==['checked']

@pytest.mark.parametrize('owned',[True,False])
def test_adopt_only_matching_owned_process(tmp_path,owned):
    owner(tmp_path)
    process=SimpleNamespace(pid=42,cwd=lambda:str(tmp_path),
       cmdline=lambda:['python','serve_acr_observer.py' if owned else 'unrelated.py'],create_time=lambda:123.)
    new,m,calls=function('run_acr_resume_v2.py',tmp_path,lambda pid:process);new()
    assert m.observer_identity==((42,123.) if owned else None)
    assert calls==['checked']
