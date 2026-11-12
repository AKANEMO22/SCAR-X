"""Regenerate already-scored raw inputs and independently validate event labels.

No neural forward, fitting, model loading, new seed or historical-result write.
"""
from pathlib import Path
import hashlib,importlib.util,json,os,subprocess,sys,time
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[1];SOURCE=ROOT.parent/'forensic-source-gemini-8f24a5a'
sys.path.insert(0,str(SOURCE/'src'))
import numpy as np
from scarx.evaluation import r6_associative as R
from scarx.runtime.resources import ResourceMonitor
from scarx.runtime.limits import apply_memory_limit


def load_reference():
    path=ROOT/'src/scarx/evaluation/forensic/event_oracle.py'
    spec=importlib.util.spec_from_file_location('independent_event_oracle',path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module


def main():
    if subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip()!='8f24a5abde0598201e558eee559564ccde889145':
        raise ValueError('Pinned source moved')
    protocol=ROOT/'docs/research/forensic-r6-20261111/RAW_EVENT_ORACLE_FOLLOWUP_PROTOCOL.md'
    # The scorer deliberately does not import model/evaluation code from audit
    # source; load its new reference by explicit file path alongside source G/R.
    O=load_reference();cap=apply_memory_limit()
    # runs.py has relative runtime imports. Import the ordinary source package's
    # identical plumbing, with ROOT explicitly passed as the destination.
    from scarx.evaluation.runs import new_run,write_json
    run=new_run(ROOT,'forensic-gemini-events-v1',{'display_title':'Gemini actual raw event/label oracle · no learning',
        'run_class':'DATA_ONLY_HISTORICAL_AUDIT','source_sha':'8f24a5abde0598201e558eee559564ccde889145',
        'protocol_sha256':hashlib.sha256(protocol.read_bytes()).hexdigest(),'neural_forward_calls':0,'learning_updates':0},
        data_description='Already-scored historical synthetic raw inputs and seeded noise; no private/live data.',
        data_files=['docs/research/forensic-r6-20261111/RAW_EVENT_ORACLE_FOLLOWUP_PROTOCOL.md'])
    records=[];started=time.perf_counter();error=None
    with ResourceMonitor(interval=.02) as monitor:
        try:
            for seed in (121011,121017,121023,121029,121031,121037,121041,121043):
                train=R.G.independent_query_set_v1(seed+200000,1024)
                held=R.G.independent_query_set_v1(seed+300000,512)
                paired=R.G.independent_paired_set_v1(seed+400000,256)
                base=[('train',train[0],train[1]),('held',held[0],held[1]),('paired',paired[0],paired[1])]
                conditions=[]
                for split,x,y in base:
                    conditions.append(('clean',split,x,y,2,None))
                    conditions.append(('shift_plus_one',split,R.G.shifted(x,1,seed=seed),y,2,None))
                    conditions.append(('reversed_writes',split,R.G.reversed_writes(x),y,2,None))
                    base_events=O.decode_events(x)
                    moved=R.move_query(x,1);independently_moved=O.move_known_query_tensor(x,base_events)
                    if not np.array_equal(moved,independently_moved):raise ValueError('Source query movement differs from registered independent linear transform')
                    moved_events=O.move_known_query_tensor(O.pulse_tensor(base_events,x.shape),base_events)
                    conditions.append(('query_tick_variation',split,moved,y,2,{'event_tensor':moved_events,'source_transform_exact':True}))
                    offset={'train':11,'held':12,'paired':13}[split]
                    conditions.append(('distractors',split,R.G.with_distractors(x,seed+offset),y,2,None))
                    noise_offset={'train':21,'held':22,'paired':23}[split]
                    noisy=R.G.noisier(x,seed+noise_offset,sigma=.16)
                    background=np.random.default_rng(seed+noise_offset).normal(0,.16,x.shape).astype(np.float32)
                    conditions.append(('noise_change',split,noisy,y,2,background))
                conditions.append(('paired_same_sample_distractors','paired',R.paired_with_same_sample_distractors(paired[0],seed+13),paired[1],2,None))
                for split,count,offset in [('train',1024,500000),('held',512,510000)]:
                    x,y,_=R.irrelevant_third_write_windowed_set_v1(count,seed=seed+offset)
                    conditions.append(('aux_structured_third_write_windowed',split,x,y,2,None))
                x,y,_=R.irrelevant_third_write_windowed_paired_set_v1(256,seed=seed+520000)
                conditions.append(('aux_structured_third_write_windowed','paired',x,y,2,None))
                for split,count,offset in [('train',1024,530000),('held',512,540000)]:
                    x,y,_=R.key_overwrite_query_set_v1(count,seed=seed+offset)
                    conditions.append(('aux_key_overwrite',split,x,y,3,None))
                x,y,_=R.key_overwrite_paired_set_v1(256,seed=seed+550000)
                conditions.append(('aux_key_overwrite','paired',x,y,3,None))
                for condition,split,x,y,writes,background in conditions:
                    if time.perf_counter()-started>120:raise TimeoutError('Data-only120s limit')
                    unresolved_raw=None
                    if background is not None:
                        try:raw_decode=O.decode_events(x,expected_writes=writes)
                        except O.EventOracleError as exc:unresolved_raw=str(exc)
                        event_input=background['event_tensor'] if isinstance(background,dict) else x.astype(np.float64)-background.astype(np.float64)
                    else:event_input=x
                    decoded=O.decode_events(event_input,expected_writes=writes)
                    wrong=np.flatnonzero(decoded['label']!=y)
                    item={'seed':seed,'condition':condition,'split':split,'samples':len(x),
                        'input_sha256':hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest(),
                        'target_sha256':hashlib.sha256(np.ascontiguousarray(y).tobytes()).hexdigest(),
                        'oracle_sha256':hashlib.sha256(decoded['label'].tobytes()).hexdigest(),
                        'wrong_label_rows':wrong.tolist(),'query_ticks':np.unique(decoded['query_tick']).tolist(),
                        'write_count':writes,'evaluation_only_latent_event_reconstruction':background is not None,
                        'oracle_background_method':'independent_query_tick_linear_transform' if isinstance(background,dict) else 'registered_seeded_gaussian_subtraction' if background is not None else 'ordinary_marker_parse',
                        'raw_noisy_threshold_oracle_ambiguous':unresolved_raw}
                    source_cue=R.G.query_cue(x,tick=int(decoded['query_tick'][0]))[:,1].astype(np.int32)
                    cue_mismatch=np.flatnonzero(source_cue!=decoded['q'])
                    item['source_cue_disagrees_with_latent_query_rows']=cue_mismatch.tolist()
                    item['source_cue_disagrees_with_latent_query_count']=len(cue_mismatch)
                    if split=='paired':item['pair_contract']=O.pair_contract(x,decoded)
                    if condition=='clean':
                        old=R.G.baseline_first_key(x)
                        item['source_first_key_matches_key_A_value']=bool(np.array_equal(old,decoded['vA']))
                        item['source_first_key_differs_from_chronological_first_rows']=int(np.sum(old!=decoded['first_written_value']))
                    records.append(item);monitor.check()
                print(json.dumps({'historical_raw_seed_done':seed,'conditions_splits':len(conditions)}),flush=True)
        except Exception as exc:error=type(exc).__name__+': '+str(exc)
    result={'decision':'RAW_EVENT_LABELS_SOURCE_VERIFIED' if error is None and not any(r['wrong_label_rows'] for r in records) else 'RAW_EVENT_AUDIT_FAILED',
        'records':records,'rows_checked':sum(r['samples'] for r in records),'wrong_labels':sum(len(r['wrong_label_rows']) for r in records),
        'neural_forward_calls':0,'learning_updates':0,'fresh_seeds_used':False,'resources':monitor.report(),'cap':cap,'error':error,
        'scope':'Latent generated-event/label consistency. Seeded noise removed only by evaluator, never fed to model; not a guarantee every noisy observation has an unambiguous threshold decoding.',
        'delivered_24453bd_package_overwritten':False}
    write_json(run/'result.json',result)
    (run/'report.md').write_text('# Gemini raw event follow-up\n\n'+json.dumps({k:v for k,v in result.items() if k!='records'},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    output=ROOT/'artifacts/forensic-r6/gemini-events-receipt.json'
    write_json(output,{'run':run.relative_to(ROOT).as_posix(),**{k:v for k,v in result.items() if k!='records'}})
    print(json.dumps({'run':run.name,'decision':result['decision'],'rows_checked':result['rows_checked'],'wrong_labels':result['wrong_labels'],'error':error}),flush=True)
    return int(error is not None or result['wrong_labels']>0)


if __name__=='__main__':raise SystemExit(main())
