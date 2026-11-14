"""Data-only credit/history decomposition; no neural inference, updates or tools."""

import hashlib
import json
from pathlib import Path
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/20261114T064319Z-credit-direction-diagnostic-v1-fc84eb'
PARENT=ROOT/'runs/20261114T035022Z-predicate-retention-dev-v1-032f30'


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def metric(value, reference):
    value,reference=value.astype(np.float64).ravel(),reference.astype(np.float64).ravel()
    vn,rn=np.linalg.norm(value),np.linalg.norm(reference)
    mask=np.abs(reference)>1e-8
    return {'cosine':float(value@reference/(vn*rn)) if vn*rn else None,
        'candidate_norm':float(vn),'surrogate_reference_norm':float(rn),
        'sign_agreement_on_reference_abs_gt_1e_8':float(np.mean(np.sign(value[mask])==np.sign(reference[mask]))) if mask.any() else None,
        'reference_nonzero_coordinates':int(mask.sum()),'norm_ratio':float(vn/rn) if rn else None}


def main():
    started=time.monotonic()
    manifest=json.loads((RUN/'manifest.json').read_text())
    for rel,expected in manifest['source_sha256'].items():assert digest(RUN/'source'/rel)==expected
    assert digest(PARENT/'locked-owned-data.json')==manifest['parent_data_sha256']
    original=json.loads((RUN/'result.json').read_text());assert not original['failures']
    control=json.loads((RUN/'control-result.json').read_text());assert control['backward_passes']==48 and control['optimizer_steps']==0
    rows=[]
    for row in control['records']:
        with np.load(RUN/row['state_file'],allow_pickle=False) as z:
            state={key:z[key].copy() for key in z.files if key!='config'}
        if row['state']=='final640':
            parent=PARENT/f"{row['seed']}-current_hidden/final-full-state.bin"
            assert digest(parent)==row['source_checkpoint_sha256']
            with np.load(parent,allow_pickle=False) as z:
                for key in ('w_in','w_rec','w_out','bias','indices'):
                    np.testing.assert_array_equal(state[key],z['core.'+key])
                for key in ('f_mean','f_var'):np.testing.assert_array_equal(state[key],z[key])
                previous_trace=z['syn_trace'].copy();previous_energy=z['syn_energy'].copy()
                assert json.loads(z['metadata'].tobytes())['updates']==640
        else:
            previous_trace=np.zeros((128,72),np.float32)
            previous_energy=np.full((128,72),1e-4,np.float32)
        with np.load(RUN/row['case_file'],allow_pickle=False) as z:
            current=z['local_direction'].copy();recorded=z['prospective_delta'].copy()
        with np.load(RUN/row['case_file'].replace('.npz','-surrogate.npz'),allow_pickle=False) as z:
            reference=z['hidden_direction'].copy();input_credit=z['input_direction'].copy()
        trace=.9*previous_trace+.1*current
        energy=.99*previous_energy+.01*current**2
        denominator=np.sqrt(energy)+1e-4
        preconditioned=.008*trace/denominator
        clipped=np.clip(preconditioned,-.01,.01)
        weights=np.concatenate((state['w_in'],state['w_rec']),axis=1)
        actual=np.clip(weights+clipped,-3,3)-weights
        np.testing.assert_allclose(actual,recorded,atol=2e-7,rtol=0)
        for kind,array in (('local_vs_surrogate',current),('prospective_delta_vs_surrogate',recorded)):
            derived=metric(array,reference)
            for key,value in derived.items():
                expected=row[kind][key]
                if value is None:assert expected is None
                else:assert abs(value-expected)<1e-10
        current_only_same_energy=np.clip(.008*(.1*current)/denominator,-.01,.01)
        current_only_same_energy=np.clip(weights+current_only_same_energy,-3,3)-weights
        q=float(np.linalg.norm(input_credit[:8].astype(np.float64)))
        r=float(np.linalg.norm(input_credit[8:].astype(np.float64)))
        assert abs(q-row['query_input_credit_norm'])<1e-10 and abs(r-row['record_input_credit_norm'])<1e-10
        rows.append({'id':row['id'],'api':row['api'],'seed':row['seed'],'state':row['state'],
            'local':metric(current,reference),'accumulated_trace':metric(trace,reference),
            'current_only_same_energy_counterfactual':metric(current_only_same_energy,reference),
            'recorded_actual_delta':metric(recorded,reference),
            'preconditioned_coordinate_clip_fraction':float(np.mean(np.abs(preconditioned)>.01)),
            'weight_bound_changes_fraction':float(np.mean(np.abs(actual-clipped)>2e-7)),
            'query_to_record_input_credit_norm_ratio':q/r if r else None,
            'reconstructed_delta_max_absolute_error':float(np.max(np.abs(actual-recorded))),
            'forward_state_parity_max_error':max(row['state_max_abs_errors'])})
    summaries=[]
    for state in ('initial','final640'):
        group=[r for r in rows if r['state']==state];summary={'state':state,'cases':len(group)}
        for name in ('local','accumulated_trace','current_only_same_energy_counterfactual','recorded_actual_delta'):
            cos=[r[name]['cosine'] for r in group if r[name]['cosine'] is not None]
            summary[name]={'cosine_median':float(np.median(cos)),'cosine_min':min(cos),'cosine_max':max(cos),
                           'negative_cosine_cases':sum(value<0 for value in cos)}
        summary['median_clipped_fraction']=float(np.median([r['preconditioned_coordinate_clip_fraction'] for r in group]))
        summary['max_weight_bound_changes_fraction']=max(r['weight_bound_changes_fraction'] for r in group)
        summary['median_query_to_record_credit_ratio']=float(np.median([r['query_to_record_input_credit_norm_ratio'] for r in group]))
        summaries.append(summary)
    assert time.monotonic()-started<60
    supplement=RUN/'history-decomposition-v1';supplement.mkdir()
    files=[Path(__file__),ROOT/'experiments/hybrid-credit-history-decomposition-v1.md']
    hashes={}
    for path in files:
        rel=path.relative_to(ROOT);dest=supplement/'source'/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(path.read_bytes());hashes[rel.as_posix()]=digest(path)
    report={'run_id':RUN.name,'source_sha256':hashes,'recorded_metrics_rederived':True,
        'recorded_prospective_delta_reconstructed':True,'records':rows,'summary':summaries,
        'source_checkpoint_arrays_and_normalization_match':True,
        'max_reference_forward_parity_error':max(r['forward_state_parity_max_error'] for r in rows),
        'new_forward_or_backward_or_learning':False,'counterfactual_is_not_a_new_trained_learner':True,
        'initial_zero_trace_energy_is_documented_not_regenerated_here':True,
        'scope':'Posthoc8oldtrainingcases×3seeds×2states, surrogate gradient not true hard-spike gradient; not causal forgetting/quality proof',
        'verdict':'RECORDED_CREDIT_AND_UPDATE_HISTORY_DIAGNOSTIC_VERIFIED_NO_PROMOTION',
        'seconds':time.monotonic()-started}
    with (supplement/'independent-data-audit.json').open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2))


if __name__=='__main__':main()
