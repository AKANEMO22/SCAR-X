"""Source-status compatibility before any model call; no tool/model execution."""
import ast,copy,importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest

ROOT=Path(__file__).resolve().parents[1]


def source_effect():
    tree=ast.parse((ROOT/'scripts/qualify_hybrid_p1b_multi_api_v2.py').read_text(encoding='utf-8'))
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='effect')
    scope={'copy':copy}
    exec(compile(ast.Module(body=[node],type_ignores=[]),'actual-checker-function','exec'),scope)
    return scope['effect']


def test_exact_complete_state_matches_actual_pinned_API_source():
    path=ROOT/'local/upstream/ToolTalk-e05f4ce6132c80ed33392b81535b077d56ab28fd/src/tooltalk/apis/reminder.py'
    tree=ast.parse(path.read_text(encoding='utf-8'))
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='CompleteReminder')
    call=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='call')
    values=[n.value.value for n in ast.walk(call) if isinstance(n,ast.Assign) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str)]
    assert 'complete' in values and 'completed' not in values
    before={'Reminder':{'owner':{'x':{'status':'pending'}}},'Alarm':{}}
    after=copy.deepcopy(before);after['Reminder']['owner']['x']['status']='complete'
    row={'api':'CompleteReminder','correct_request':{'parameters':{'reminder_id':'x'}}}
    assert source_effect()(before,after,row,'owner')
    after['Reminder']['owner']['x']['status']='completed'
    assert not source_effect()(before,after,row,'owner')


def test_other_record_changes_cannot_pass_exact_effect_contract():
    before={'Reminder':{'owner':{'x':{'status':'pending'},'y':{'status':'pending'}}},'Alarm':{}}
    after=copy.deepcopy(before);after['Reminder']['owner']['x']['status']='complete';after['Reminder']['owner']['y']['status']='complete'
    row={'api':'CompleteReminder','correct_request':{'parameters':{'reminder_id':'x'}}}
    assert not source_effect()(before,after,row,'owner')
