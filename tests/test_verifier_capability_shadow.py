"""Actual unbound controller mask, no neural/assistant fake output."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest

ROOT=Path(__file__).resolve().parents[1]


def shadow(patched=True):
    spec=importlib.util.spec_from_file_location('verifier_guard_recipe_test',ROOT/'integrations/hermes/verifier_capability_guard_patch.py')
    recipe=importlib.util.module_from_spec(spec);spec.loader.exec_module(recipe)
    module_name='scarx.hybrid._verifier_capability_shadow_test' if patched else 'scarx.hybrid._verifier_baseline_shadow_test'
    scope={'__name__':module_name,'__package__':'scarx.hybrid'}
    original=(ROOT/'runs/20261114T105211Z-verifier-capability-p0-v1-8e205a/source/src/scarx/hybrid/engine.py').read_bytes()
    exec(compile(recipe.patch_bytes(original) if patched else original,'owned-shadow-engine.py','exec'),scope)
    return scope['HybridEngine']


def candidate():return shadow(True)


def baseline():return shadow(False)


def controller():
    return SimpleNamespace(state={'search_times':[],'question':None},config=SimpleNamespace(search_per_hour=12))


def goal(text,**extra):
    return dict(question=text,public_topic_id=None,origin='user',reviewed=True,attempts=1,
                verified_attempted=False,searched=False,**extra)


@pytest.mark.parametrize('text', ['12+3','9 minus 11','có 9 bánh, thêm 2','１８＋３'])
def test_supported_checker_choices_keep_original_mask(text):
    assert candidate().allowed(controller(),goal(text))==baseline().allowed(controller(),goal(text))


@pytest.mark.parametrize('text',['12*3','123+2','12+3+4','not 12+3','màu chiếc xe là gì?'])
def test_unsupported_checker_is_removed_without_removing_other_authorizations(text):
    before=baseline().allowed(controller(),goal(text))
    after=candidate().allowed(controller(),goal(text))
    assert 'verify' in before and after==[action for action in before if action!='verify']


def test_initial_review_and_pending_question_constraints_still_apply():
    g=goal('non arithmetic request');g['reviewed']=False;g['attempts']=0
    assert candidate().allowed(controller(),g)==['review']
    g['reviewed']=True
    c=controller();c.state['question']={'id':'owned-pending'}
    assert candidate().allowed(c,g)==['defer']
