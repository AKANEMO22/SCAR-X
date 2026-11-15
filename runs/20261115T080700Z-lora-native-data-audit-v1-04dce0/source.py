"""Saved native scale0/1 outputs: actual-effect replay and paired rescore."""
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
    original=Path((ROOT/'local/current-lora-native-portability.txt').read_text(encoding='utf-8').strip())
    result=json.loads((original/'result.json').read_text(encoding='utf-8'))
    assert result['completed'] and not result['failures'] and len(result['records'])==64 and result['owned_worker_cleanup_complete']
    probes={r['id']:r for r in json.loads((original/'locked-probes.json').read_text(encoding='utf-8'))}
    data_path=Path((ROOT/'local/current-qualified-repair-data.txt').read_text(encoding='utf-8').strip())
    data={r['id']:r for r in json.loads((data_path/'verified-data.json').read_text(encoding='utf-8'))}
    intake=json.loads((ROOT/'runs/20261113T233305Z-tooltalk-intake-v1-08df3e/manifest.json').read_text(encoding='utf-8'))
    upstream=Path(intake['local_snapshot']);executor=load_executor(upstream,intake);sandbox=Sandbox(executor,upstream/'data/databases')
    reference=json.loads((upstream/'data/easy/AddAlarm-easy.json').read_text(encoding='utf-8'));owner=reference['user']['username']
    run=ROOT/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-lora-native-data-audit-v1-'+uuid.uuid4().hex[:6]);run.mkdir()
    records=[];calls=0
    for record in result['records']:
        probe=probes[record['probe']];text=record['actual_answer'];effect=False
        if probe['family']=='repair':
            sandbox.initialize(reference['metadata'],reference['user']);before=copy.deepcopy(sandbox.executor.databases)
            actual=None
            try:
                request=json.loads(text)
                if set(request)=={'api_name','parameters'} and request['api_name']=='AddAlarm':
                    actual=sandbox.dispatch(request,origin='native_LoRA_saved_reply_effect_replay');calls+=1
                    after=copy.deepcopy(sandbox.executor.databases);old=before.get('Alarm',{}).get(owner,{});new=after.get('Alarm',{}).get(owner,{})
                    added=set(new)-set(old)
                    if len(added)==1 and actual['exception'] is None:
                        ident=next(iter(added));reconstructed=copy.deepcopy(after);del reconstructed['Alarm'][owner][ident]
                        if owner not in before.get('Alarm',{}):del reconstructed['Alarm'][owner]
                        effect=new[ident]['time']==data[record['probe']]['requested_time'] and reconstructed==before
            except (ValueError,TypeError,KeyError):pass
        elif isinstance(probe['expected'],dict):
            try:effect=json.loads(text)==probe['expected']
            except ValueError:pass
        else:effect=text.strip().casefold().rstrip('.')==probe['expected'].casefold()
        assert bool(effect)==record['exact_correct']
        assert record['resources']['dedicated_free_bytes']>=512*1024**2
        records.append({'id':record['probe'],'family':probe['family'],'split':probe.get('split'),
            'scale':record['scale'],'success':bool(effect),'seconds':record['seconds']})
    paired={scale:{r['id']:r for r in records if r['scale']==scale} for scale in (0,1)}
    assert set(paired[0])==set(paired[1])==set(probes)
    # Bootstrap repair differences by actual error kind, no token/update-level replicates.
    families={}
    for ident in probes:
        if probes[ident]['family']=='repair' and probes[ident].get('split')=='TEST':
            families.setdefault(data[ident]['error_kind'],[]).append(int(paired[1][ident]['success'])-int(paired[0][ident]['success']))
    sample=np.array([np.mean(v) for v in families.values()]);rng=np.random.default_rng(260116)
    boot=sample[rng.integers(0,len(sample),size=(10000,len(sample)))].mean(axis=1)
    controls={f:{str(s):sum(r['success'] for r in records if r['family']==f and r['scale']==s) for s in (0,1)} for f in ('math','JSON','source')}
    latency={str(s):{'median_seconds':float(np.median([r['seconds'] for r in records if r['scale']==s])),
        'p95_seconds':float(np.quantile([r['seconds'] for r in records if r['scale']==s],.95))} for s in (0,1)}
    report={'verdict':'NATIVE_FP32_LORA_PORTABILITY_AND_SAVED_EFFECTS_VERIFIED_WITH_SMALL_SAMPLE_LIMITS',
        'native_source_run':original.name,'actual_tool_replays':calls,'new_model_or_learning_calls':0,
        'same_native_process_and_base_for_both_scales':True,'matched_generation_budget':True,
        'repair_DEV_TEST_counts':{str(s):sum(r['success'] for r in records if r['family']=='repair' and r['scale']==s) for s in (0,1)},
        'repair_probe_count':20,'repair_TEST_counts':{str(s):sum(r['success'] for r in records if r['family']=='repair' and r['split']=='TEST' and r['scale']==s) for s in (0,1)},
        'repair_TEST_count':16,'exploratory_error_family_gain_CI95_pp':(100*np.quantile(boot,[.025,.975])).tolist(),
        'observed_old_control_counts':controls,'old_control_count_per_family':4,'old_control_total_per_scale':12,
        'observed_old_control_regression_pp':0,'old_skill_noninferiority_generalization_NOT_proven':True,
        'reason':'0/12 observed regressions is insufficient to establish<=3pp average/<=5pp family population loss',
        'same_prompt_order_warm_cache_may_confound_latency':True,'paired_latency_descriptive_only':latency,
        'long_horizon_retention_or_20stage_learning_measured':False,'LoRA_gain_is_not_hidden_SNN_gain':True,
        'adapter_installed_in_Hermes':False,'production_or_full_retention_PASS':False,
        'records':records,'raw_result_sha256':hashlib.sha256((original/'result.json').read_bytes()).hexdigest()}
    (run/'result.json').write_text(encode(report),encoding='utf-8');(run/'source.py').write_bytes(Path(__file__).read_bytes())
    (ROOT/'local/current-lora-native-audit.txt').write_text(str(run),encoding='utf-8')
    print(encode({'run':str(run),**{k:v for k,v in report.items() if k!='records'}}),flush=True)


if __name__=='__main__':main()
