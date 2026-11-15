"""Extend the locked qualified bank to TRAIN/TEST with identical native executor."""
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import uuid

ROOT=Path(__file__).resolve().parents[1]


def main():
    qualified=Path((ROOT/'local/current-p1b-action-usefulness.txt').read_text(encoding='utf-8').strip())
    proof=json.loads((qualified/'result.json').read_text(encoding='utf-8'))
    assert proof['dataset_qualified_for_training'] and proof['oracle_minus_strongest_control_pp']>=5
    spec=importlib.util.spec_from_file_location('_locked_qualifier',ROOT/'scripts/qualify_hybrid_p1b_action_usefulness.py')
    q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
    # Reuse the exact executed macro/native loop by substituting the data source
    # in an archived in-memory module. No prompt/hyperparameter tuning.
    source=(ROOT/'scripts/qualify_hybrid_p1b_action_usefulness.py').read_text(encoding='utf-8')
    source=source.replace("'-p1b-action-usefulness-v1-'","'-p1b-train-test-outcomes-v1-'")
    source=source.replace("local/p1b-action-usefulness-v1-execution.json","local/p1b-train-test-outcomes-v1-execution.json")
    source=source.replace("dev=[r for r in allrows if r['split']=='DEV']","dev=[r for r in allrows if r['split'] in ('TRAIN','TEST')]")
    source=source.replace('assert len(dev)==8 and len(train)==32','assert len(dev)==48 and len(train)==32')
    source=source.replace("'split':'DEV only'","'split':'TRAIN and TEST outcome generation; no TEST used for choosing methods'")
    source=source.replace("'max_native_calls':56,'max_seconds':900","'max_native_calls':336,'max_seconds':900")
    source=source.replace('calls>=56','calls>=336')
    source=source.replace("local/current-p1b-action-usefulness.txt","local/current-p1b-training-outcomes.txt")
    # No gate is chosen on these data; inherited summary fields are diagnostic.
    source=source.replace("dataset_qualified_for_training=oracle-max(rule,rag)>=.05 and max(rule,rag)<1,",
        "dataset_qualified_for_training=False,qualification_is_fixed_by_prior_DEV=True,")
    source=source.replace('/8\n','/len(dev)\n')
    source=source.replace('TEST_used=False','TEST_used=True,TEST_is_outcome_only_not_training=True')
    scope={'__name__':'_p1b_extended_counterfactual_runner','__file__':str(Path(__file__))}
    exec(compile(source,'p1b-outcome-expansion-locked-loop.py','exec'),scope)
    scope['main']()


if __name__=='__main__':main()
