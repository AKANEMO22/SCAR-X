"""Verify saved CUDA probabilities, reset weights and recorded masks; no forwards."""

import argparse
import ast
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from scarx.hybrid.store import Store,encode


def main(run):
    run=run.resolve();assert run.is_relative_to(ROOT/"runs")
    read=lambda name:json.loads((run/name).read_text(encoding="utf-8"))
    manifest,result,cases=map(read,("manifest.json","result.json","locked-cases.json"))
    assert not result['failures'] and result['attempted_GPU_forwards']==80 and result['learning_steps']==0
    assert len(cases)==20 and len(result['cells'])==4 and result['seconds']<=180
    assert hashlib.sha256((run/'locked-cases.json').read_bytes()).hexdigest()==manifest['locked_cases_sha256']
    for rel,digest in manifest['source_sha256'].items():
        assert hashlib.sha256((run/'source'/rel).read_bytes()).hexdigest()==digest
    program=ast.parse((run/'source/scripts/diagnose_hybrid_action_policy.py').read_text(encoding='utf-8'))
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ('learn','consolidate','backward') for n in ast.walk(program))
    parent=ROOT/'local'/manifest['parent']/'hybrid.sqlite'
    with sqlite3.connect(parent.resolve().as_uri()+'?mode=ro',uri=True) as db:
        digest,payload=db.execute('SELECT digest,payload FROM checkpoints WHERE digest=?',(manifest['parent_checkpoint_sha256'],)).fetchone()
    assert hashlib.sha256(payload).hexdigest()==digest
    rows=[]
    with np.load(run/'actual-CUDA-forward-arrays.npz',allow_pickle=False) as data,np.load(io.BytesIO(payload),allow_pickle=False) as saved:
        meta=json.loads(saved['metadata'].tobytes())
        decisions=meta['controller']['decisions']
        for case in cases:
            assert np.array_equal(np.array(case['sequence'],np.float32),np.array(decisions[case['decision_id']]['sequence'],np.float32))
        # Independent source-initialization arithmetic, no neural forward.
        rng=np.random.default_rng(260113)
        for _ in range(128):rng.choice(128,8,replace=False)
        expected={'w_in':rng.normal(.15,.45,(128,64)).astype(np.float32),
                  'w_rec':rng.normal(0,.12/np.sqrt(8),(128,8)).astype(np.float32),
                  'w_out':rng.normal(0,.03,(8,128)).astype(np.float32),
                  'bias':np.zeros(8,np.float32)}
        for key,value in expected.items():
            np.testing.assert_array_equal(data['initial_'+key],value)
            np.testing.assert_array_equal(data['final_'+key],saved['core.'+key])
        for arm in ('hidden_reset','readout_reset'):
            reset=('w_in','w_rec') if arm=='hidden_reset' else ('w_out','bias')
            for key in expected:
                np.testing.assert_array_equal(data[arm+'_'+key],data[('initial' if key in reset else 'final')+'_'+key])
        names=('review','search','verify','ask','defer','consolidate')
        for cell in result['cells']:
            arm=cell['arm'];f=data[arm+'_standardized'];p=data[arm+'_probabilities']
            assert cell['actual_device']=='CUDA0' and f.shape==(20,128) and p.shape==(20,6)
            assert f.dtype==p.dtype==np.float32
            for key in expected:
                assert hashlib.sha256(data[arm+'_'+key].tobytes()).hexdigest()==cell['parameter_fingerprints_unchanged']['core.'+key]
            # Numerical audit of saved arrays, not executing an ALIF model.
            logits=f.astype(np.float64)@data[arm+'_w_out'][2:].astype(np.float64).T+data[arm+'_bias'][2:]
            logits-=logits.max(axis=1,keepdims=True)
            reference=np.exp(logits);reference/=reference.sum(axis=1,keepdims=True)
            np.testing.assert_allclose(p,reference,atol=2e-6,rtol=2e-6)
            chosen=[names[max((names.index(a) for a in case['allowed']),key=lambda i:float(prob[i]))]
                    for case,prob in zip(cases,p)]
            assert chosen==cell['selected_actions'] and chosen.count('verify')==cell['verify_choices']
            rows.append({'arm':arm,'verify_choices_on_recorded_seen_states':chosen.count('verify'),
                         'cases':20,'probability_max_abs_error':float(np.max(np.abs(p-reference)))})
    assert min(sample['free_MiB'] for sample in result['GPU_samples'])>=512
    report={'run_id':run.name,'verdict':'RECORDED_GPU_ACTION_RESETS_DATA_VERIFIED_NOT_HIDDEN_GENERALIZATION_PASS',
            'cells':rows,'initialization_final_parent_and_exact_reset_arrays_match':True,
            'probabilities_rederived_from_saved_features_and_readout':True,
            'new_neural_forwards_or_learning_in_audit':0,'source_diagnostic_forwards':80,
            'posthoc_seen_states_one_seed':True,'statistics_retained_in_reset_arms':True,
            'no_readout_only_matched_training_or_unseen_confirmation':True,
            'formal_quality_retention_or_autonomous_tool_repair_PASS':False}
    with (run/'independent-data-audit-v1.json').open('x',encoding='utf-8') as output:output.write(encode(report))
    store=Store(ROOT/'local/hybrid-v1')
    try:store.event('action_policy_counterfactual.data_verified',report,'orchestrator')
    finally:store.close()
    print(encode(report))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('run',type=Path);main(parser.parse_args().run)
