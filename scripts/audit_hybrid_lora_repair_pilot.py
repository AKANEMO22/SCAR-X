"""Independent actual-effect replay and adapter artifact audit. No model calls."""
import copy,hashlib,json
from pathlib import Path
import sys
from datetime import datetime,timezone
import uuid
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from scarx.evaluation.tooltalk_sandbox import Sandbox,load_executor
from scarx.hybrid.store import encode


def main():
    original=Path((ROOT/'local/current-native-lora-repair-run.txt').read_text(encoding='utf-8').strip())
    result=json.loads((original/'result.json').read_text(encoding='utf-8'));assert result['completed']
    manifest=json.loads((original/'manifest.json').read_text(encoding='utf-8'))
    data_dir=ROOT/'runs'/manifest['data_run'];data={r['id']:r for r in json.loads((data_dir/'verified-data.json').read_text(encoding='utf-8'))}
    assert hashlib.sha256((data_dir/'verified-data.json').read_bytes()).hexdigest()==manifest['data_sha256']
    intake=json.loads((ROOT/'runs/20261113T233305Z-tooltalk-intake-v1-08df3e/manifest.json').read_text(encoding='utf-8'))
    upstream=Path(intake['local_snapshot']);cls=load_executor(upstream,intake);sandbox=Sandbox(cls,upstream/'data/databases')
    reference=json.loads((upstream/'data/easy/AddAlarm-easy.json').read_text(encoding='utf-8'));owner=reference['user']['username']
    run=ROOT/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-lora-repair-audit-v1-'+uuid.uuid4().hex[:6]);run.mkdir()
    records=[];calls=0
    for prediction in result['predictions']:
        row=data[prediction['id']];sandbox.initialize(reference['metadata'],reference['user'])
        before=copy.deepcopy(sandbox.executor.databases);effect=False;actual=None
        try:
            request=json.loads(prediction['actual_answer'])
            if set(request)=={'api_name','parameters'} and request['api_name']=='AddAlarm':
                actual=sandbox.dispatch(request,origin='independent_saved_LoRA_prediction_replay');calls+=1
                after=copy.deepcopy(sandbox.executor.databases);old=before.get('Alarm',{}).get(owner,{});new=after.get('Alarm',{}).get(owner,{})
                added=set(new)-set(old)
                if len(added)==1 and actual['exception'] is None:
                    identifier=next(iter(added));reconstructed=copy.deepcopy(after);del reconstructed['Alarm'][owner][identifier]
                    if owner not in before.get('Alarm',{}):del reconstructed['Alarm'][owner]
                    effect=(new[identifier].get('time')==row['requested_time'] and reconstructed==before)
        except (ValueError,KeyError,TypeError):pass
        assert bool(effect)==prediction['exact_JSON_repair']
        records.append({'id':row['id'],'phase':prediction['phase'],'split':row['split'],'error_kind':row['error_kind'],
                        'verified_task_effect':bool(effect),'actual_replay_result':actual})
    assert len(records)==40
    test={phase:{r['id']:r for r in records if r['phase']==phase and r['split']=='TEST'} for phase in ('baseline','final')}
    diff=np.array([int(test['final'][i]['verified_task_effect'])-int(test['baseline'][i]['verified_task_effect']) for i in sorted(test['baseline'])])
    family={f:np.array([int(test['final'][i]['verified_task_effect'])-int(test['baseline'][i]['verified_task_effect'])
        for i in test['baseline'] if test['baseline'][i]['error_kind']==f]).mean()
        for f in sorted({r['error_kind'] for r in test['baseline'].values()})}
    rng=np.random.default_rng(260115);values=np.array(list(family.values()))
    boot=values[rng.integers(0,len(values),size=(10000,len(values)))].mean(axis=1)
    adapter=ROOT/result['adapter_path'];configuration=json.loads((adapter/'adapter_config.json').read_text(encoding='utf-8'))
    raw=(adapter/'adapter_model.safetensors').read_bytes();import struct
    n=struct.unpack('<Q',raw[:8])[0];header=json.loads(raw[8:8+n]);tensors={k:v for k,v in header.items() if k!='__metadata__'}
    assert tensors and all('lora_A' in name or 'lora_B' in name for name in tensors)
    assert all(item['dtype']=='F32' for item in tensors.values())
    b_norms={}
    for name,item in tensors.items():
        start,end=item['data_offsets'];values=np.frombuffer(raw,offset=8+n+start,count=(end-start)//4,dtype='<f4')
        assert np.isfinite(values).all()
        if 'lora_B' in name:b_norms[name]=float(np.linalg.norm(values))
    assert all(n>0 for n in b_norms.values())
    assert configuration['r']==4 and configuration['lora_alpha']==8
    attempts=[json.loads(line) for line in (original/'attempts.jsonl').read_text(encoding='utf-8').splitlines()]
    assert sum(a['kind']=='update' for a in attempts)==32 and result['optimizer_steps']==32
    update_ids=[a['id'] for a in attempts if a['kind']=='update']
    assert len(set(update_ids))==32 and all(data[i]['split']=='TRAIN' for i in update_ids)
    assert {p['id'] for p in result['predictions'] if p['phase']=='baseline'}=={p['id'] for p in result['predictions'] if p['phase']=='final'}
    assert all(u['id']==i for u,i in zip(result['updates'],update_ids))
    assert len(result['updates'])==32 and all(np.isfinite(u['loss']) for u in result['updates'])
    assert all(u['resources']['dedicated_GPU_free_bytes']>=512*1024**2 for u in result['updates'])
    assert result['save_reload_logits_exact'] and result['base_weights_exactly_unchanged']
    output={'verdict':'LORA_PILOT_ACTUAL_REPAIR_IMPROVEMENT_REPLAY_VERIFIED_NOT_PRODUCTION',
        'original_run':original.name,'actual_owned_tool_replays':calls,'new_model_or_optimizer_calls':0,
        'predictions_replayed':40,'baseline_TEST_correct':int(sum(r['verified_task_effect'] for r in test['baseline'].values())),
        'final_TEST_correct':int(sum(r['verified_task_effect'] for r in test['final'].values())),
        'TEST_count':16,'paired_gain_pp':100*float(diff.mean()),
        'exploratory_error_family_cluster_CI95_pp':(100*np.quantile(boot,[.025,.975])).tolist(),
        'cluster_count':4,'single_seed_small_bilingual_template_pilot':True,
        'not_same_backend_as_native_llama_cpp_RAG_scores':True,'not_a_SNN_advantage':True,
        'long_term_or_general_retention_measured':False,'production_or_retention_qualified':False,
        'optimizer_saw_only32_unique_TRAIN_rows_no_DEV_TEST_updates':True,
        'loss_first_last':[result['updates'][0]['loss'],result['updates'][-1]['loss']],
        'sampled_dedicated_GPU_reserve_during_updates_held':True,
        'adapter_tensor_count':len(tensors),'all_adapter_dtypes':'FP32','all_B_matrices_nonzero':True,
        'adapter_safetensors_sha256':hashlib.sha256(raw).hexdigest(),'adapter_bytes':len(raw),
        'save_reload_logits_exact_observed_in_original_run':True,'base_original_dtypes_retained':True,
        'raw_predictions_sha256':hashlib.sha256((original/'result.json').read_bytes()).hexdigest(),'rows':records}
    (run/'result.json').write_text(encode(output),encoding='utf-8');(run/'source.py').write_bytes(Path(__file__).read_bytes())
    (ROOT/'local/current-lora-repair-audit.txt').write_text(str(run),encoding='utf-8')
    print(encode({'run':str(run),**{k:v for k,v in output.items() if k!='rows'}}),flush=True)


if __name__=='__main__':main()
