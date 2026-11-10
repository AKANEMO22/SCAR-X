"""Independent40review: copiedmodels/causaltransplants, no livelearning/restore."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import sys,json,time,hashlib,io,zipfile,sqlite3,subprocess,copy
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
import numpy as np
from scarx.neural.unified import UnifiedCore
from scarx.app.generation import Generation,PRODUCT_GUARD_PROMPTS
from scarx.evaluation.parameter_interventions import PARTS,from_payloads
from scarx.evaluation.product_acceptance import CASES,_judge,_judge_v1,canonical_reply,weights_sha256
from scarx.evaluation.runs import new_run,write_json
from scarx.runtime.resources import ResourceMonitor
from scarx.runtime.limits import apply_memory_limit
from scarx.runtime.work_trace import activity,append
from scarx.runtime.experiment_status import progress

LESSONS=(('hi','Hello.'),('cảm ơn','Rất vui được giúp bạn.'),('chào','Chào bạn.'),('bạn khỏe không','Tôi đang hoạt động.'))
PROBES={**PRODUCT_GUARD_PROMPTS,'bạn khỏe không':'Tôi đang hoạt động.','tôi muốn dạy bạn':'Bạn có thể dạy tôi.'}
GROUPS={'none':(),'w_out':('w_out',),'bias':('bias',),'hidden':('w_in','w_rec'),'head':('w_out','bias'),'w_in':('w_in',),'w_rec':('w_rec',),'all':PARTS}


def model_from_session(raw):
    if len(raw)>3_000_000:raise ValueError('Boundedlocalsession')
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        if set(z.namelist())!={'model.zip','session.json'} or sum(i.file_size for i in z.infolist())>3_000_000:raise ValueError('Localsessionformat')
        payload=z.read('model.zip');meta=json.loads(z.read('session.json'))
    if hashlib.sha256(payload).hexdigest()!=meta['sha256']:raise ValueError('Localsessionsha')
    return UnifiedCore.loads(payload)


def snapshot_sources(private):
    # Consistent SQLiteonlinebackup. Never copy a changing DB/WAL pair or write
    # the private source. Sourcepublicmetadata/weights are not added to runzip.
    dbpath=root/'local/generation1/generation1.sqlite';backup=private/'source.sqlite'
    with sqlite3.connect(dbpath.as_uri()+'?mode=ro',uri=True,timeout=5) as db,sqlite3.connect(backup) as target:db.backup(target)
    with sqlite3.connect(backup) as db:
        digest,raw=db.execute('SELECT sha,payload FROM runtime WHERE singleton=1').fetchone();assert hashlib.sha256(raw).hexdigest()==digest
        current=model_from_session(raw);sources=[('current_snapshot',current)]
        for identifier,digest,payload in db.execute('SELECT id,sha,payload FROM checkpoints ORDER BY created,id'):
            if hashlib.sha256(payload).hexdigest()!=digest:raise ValueError('Historicalcheckpointsha')
            model=model_from_session(payload)
            if model.vocabulary==current.vocabulary and model.encoding==current.encoding and model.core.config==current.core.config and np.array_equal(model.core.indices,current.core.indices) and weights_sha256(model)!=weights_sha256(current):
                sources.append(('oldest_compatible_checkpoint',model));(private/'historical-source.json').write_text(json.dumps({'checkpoint_id':identifier,'weights_sha256':weights_sha256(model)}),encoding='utf-8');break
    release=json.loads((root/'artifacts/generation1-release.json').read_text(encoding='utf-8'));raw=(root/release['checkpoint']).read_bytes();assert hashlib.sha256(raw).hexdigest()==release['sha256']
    sources.append(('public_release',UnifiedCore.loads(raw)))
    for name,model in sources:(private/(name+'.model.zip')).write_bytes(model.dumps())
    return sources


def evaluate(model,changing=None):
    result={}
    for prompt,target in PROBES.items():
        if prompt==changing:continue
        reply=canonical_reply(model,prompt);v=Generation._pair_verdict(reply['text'],target)
        result[prompt]={'generated':reply['text'],'target':target,**v,'stop_reason':reply.get('stop_reason')}
    return result


def main():
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    cap=apply_memory_limit();began=time.perf_counter();head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    run=new_run(root,'deepseek-product-review-v1',{'protocol':'experiments/deepseek-product-review-v1.md','display_title':'Review40 · kiểm độc lập guard/metric/readout interference',
        'model_scope':'COPY ONLY, unchangedALIF/localrule, never livelearning/restore. Checkpoints are startingstates, NOT independentseeds. Publicfixedfixtures only.',
        'incoming_git_head':head,'lessons':4,'steps':6,'arms':['full','readout'],'transplants':list(GROUPS)},
        data_description='Publicfixed10probe/4lesson fixtures; consistentprivateSQLite snapshot onlylocal, never included in source/publicdatafiles.',data_files=['artifacts/generation1-release.json'])
    private=root/'local/deepseek-review'/run.name;private.mkdir(parents=True);sources=snapshot_sources(private)
    (run/'report.md').write_text('# Independentreview40\n\nCopiedmodels only. No liveweights/rule/restore changes or broaderqualityclaim.\n',encoding='utf-8')
    activity(run,'Bắt đầu kiểm độc lập cập nhật DeepSeek','Đã sửa/test phépghépnhómweights và clonefailure/SQLitebackup/metricversion/acquisition. Canthiệp học chỉ trên copiedmodels; không đổi learningrule hoặc restore live.',status='started',evidence={'incoming_head':head,'protocol':'experiments/deepseek-product-review-v1.md','source_count':len(sources),'main_training':False})
    rows=[];details=[];metric_rows=[]
    with ResourceMonitor(interval=.1) as monitor:
        for source,model in sources:
            before_model=model.dumps();before_fp=weights_sha256(model)
            for prompt,expect,required in CASES:
                output=canonical_reply(model,prompt)['text'];v1=_judge_v1(output,expect,prompt,required);v2=_judge(output,expect,prompt,required)
                metric_rows.append({'source':source,'prompt':prompt,'generated':output,'legacy_v1_correct':v1['correct'],'canonical_v2_correct':v2['correct'],'scope':'Sameactualoutput, metricsdiffer; not modelimprovement or generalsemanticgrade.'})
            for prompt,target in LESSONS:
                if any(t not in model.ids for t in __import__('scarx.neural.language',fromlist=['tokenize']).tokenize(target)):
                    details.append({'source':source,'lesson':prompt,'skipped':'Targetrequiresvocabgrowth, excluded to avoid changingrepresentationdimensions.'});continue
                base=UnifiedCore.loads(before_model);base.core.reset_state();base.last_active[:]=base.ticks;base.select_capacity(min(64,base.core.config.hidden))
                initial=base.dumps();baseline=evaluate(base,prompt);baseline_pass={p for p,v in baseline.items() if v['content_ok']}
                for arm in ('full','readout'):
                    trained=UnifiedCore.loads(initial);steps=[];losses=[]
                    for step in range(1,7):
                        old={n:getattr(trained.core,n).copy() for n in PARTS};loss=float(trained.train_pair(prompt,target,mode=arm));losses.append(loss)
                        observed=evaluate(trained,prompt);lost=sorted(p for p in baseline_pass if not observed[p]['content_ok'])
                        steps.append({'step':step,'loss':loss,'newly_lost':lost,'deltas':{n:float(np.linalg.norm(getattr(trained.core,n)-old[n])) for n in PARTS}})
                    after=trained.dumps();observed=evaluate(trained,prompt);full_lost={p for p in baseline_pass if not observed[p]['content_ok']}
                    gained={p for p,v in baseline.items() if not v['content_ok'] and observed[p]['content_ok']}
                    style_lost={p for p,v in baseline.items() if v['complete_ok'] and observed[p]['content_ok'] and not observed[p]['complete_ok']}
                    results={}
                    for name,parts in GROUPS.items():
                        clone=from_payloads(initial,after,parts)
                        for parameter in PARTS:np.testing.assert_array_equal(getattr(clone.core,parameter),getattr(trained.core,parameter) if parameter in parts else getattr(base.core,parameter))
                        result=evaluate(clone,prompt);newly_lost=sorted(p for p in baseline_pass if not result[p]['content_ok']);results[name]={'newly_lost':newly_lost,'probe_outputs':result}
                        if name=='none':assert result==baseline
                        if name=='all':assert result==observed
                    assert base.dumps()==initial and model.dumps()==before_model and weights_sha256(model)==before_fp
                    steps_updated=trained.core.updates-base.core.updates
                    append(run,'learning',{'scope':f'COPY ONLY Review40 · {source}/{arm} · NOT MAIN','prompt':prompt,'target':target,
                        'passes':6,'sample_count':1,'losses':losses,'loss_mean':float(np.mean(losses)),'update_steps':steps_updated,'update_unit':'tokenupdates on disposablemodel',
                        'hidden_update_steps':steps_updated if arm=='full' else 0,
                        'weight_delta':{n:float(np.linalg.norm(getattr(trained.core,n)-getattr(base.core,n))) for n in ('w_in','w_rec','w_out')},
                        'attempted_weight_delta':{n:float(np.linalg.norm(getattr(trained.core,n)-getattr(base.core,n))) for n in ('w_in','w_rec','w_out')},
                        'changed_synapses':{n:int(np.count_nonzero(getattr(trained.core,n)!=getattr(base.core,n))) for n in ('w_in','w_rec','w_out')},
                        'accepted':True,'rolled_back':False,'guard_checked':False,'checkpoint':'Disposablecopy only, no new release/session checkpoint',
                        'reason':'DIAGNOSTIC on copiedweights; guard intentionallybypassed to observeinterference, not acceptedproductionlesson. Copy discarded afterreview; usermodel untouched.'})
                    already=sorted(p for p,v in baseline.items() if not v['content_ok']);head_lost=set(results['head']['newly_lost']);hidden_lost=set(results['hidden']['newly_lost'])
                    row={'source':source,'lesson':prompt,'arm':arm,'baseline_pass_count':len(baseline_pass),'already_failing_count':len(already),'newly_lost_count':len(full_lost),'newly_gained_count':len(gained),'style_only_loss_count':len(style_lost),
                        **{name+'_loss_count':len(results[name]['newly_lost']) for name in GROUPS},
                        'head_sufficient_for_this_loss_set':full_lost<=head_lost if full_lost else None,'hidden_sufficient_for_this_loss_set':full_lost<=hidden_lost if full_lost else None,
                        'first_loss_step':next((x['step'] for x in steps if x['newly_lost']),None),'loss_mean':float(np.mean(losses))}
                    rows.append(row);details.append({'summary':row,'lesson_target':target,'baseline':baseline,'already_failing':already,'after':observed,'steps':steps,'transplants':results})
                    append(run,'answer',{'scope':f'COPY ONLY Review40 · {source}/{arm} · not main/newlearner','case_id':source+'/'+arm+'/'+prompt,'prompt':prompt,'target':target,
                        'generated':canonical_reply(trained,prompt)['text'],'input_scope':'Actualneuraloutput on discardedcopiedweights after6unchanged train_pair steps. Guard intentionallybypassed for interferencecausaldiagnostic; no livelearning/restore.',
                        'active':min(64,trained.core.config.hidden),'allocated':trained.core.config.hidden,'checkpoint':'Disposable copiedmodel; not deployed/savedasnewrelease'})
                    progress(run,f'Review40 {source}/{prompt}/{arm} checked',len(rows),len(sources)*8,metrics={'newly_lost':len(full_lost),'baseline_pass':len(baseline_pass),'main_updates':0});monitor.check()
                    if time.perf_counter()-began>300:raise TimeoutError('Review40registered300s')
        write_json(run/'causal-details.json',details);write_json(run/'metric-comparison.json',metric_rows)
        result={'run':run.relative_to(root).as_posix(),'decision':'DEEPSEEK_PRODUCT_REVIEW_DIAGNOSTIC','deepseek_review_rows':rows,'metric_comparison_rows':metric_rows,
            'incoming_git_head':head,'source_count':len(sources),'independent_network_seed_claim':False,'all_parameter_transplants_exact':True,'all_none_and_all_controls_reproduce_expected':True,
            'all_source_model_weights_states_unchanged':True,'main_training':False,'main_restored':False,'learning_rule_changed':False,'normal_chat_or_vision_proven':False,
            'resources':monitor.report(),'cap':cap,'seconds':time.perf_counter()-began,
            'limits':['Finite10publicprobes/4lessons/6steps; checkpointsnotindependentseeds, no generalforgetting/hiddenimportance conclusions.',
                'Readoutonlysufficient in somecases doesnotprove hiddenrepresentationunimportant. Norm/LRratios not causalcontribution.',
                'v1/v2productmetrics labelclosedcanonicalstyle, not broadsemanticquality; historicalP1/P2gates/results unchanged.',
                'Cloned diagnosticupdates bypassguard intentionally; not productionacceptedlessons, no currentmodelknowledgegain/deployment.']}
        (run/'report.md').write_text('# Independentreview40 of DeepSeek productchanges\n\nTechnical/copy-onlydiagnostic. Samecodelearningrule, no real-sessionlearning/restore.\n\n'+json.dumps(result,indent=2)+'\n',encoding='utf-8')
        write_json(run/'result.json',result);write_json(root/'artifacts/deepseek-product-review.json',result)
        activity(run,'Hoàn tất canthiệp kiểmđộc lập DeepSeek','Transplants selectedAFTER/unselectedBEFORE khớp, all/none controls khớp. Không lấy normhoặcmộtbài đểkếtluận hidden=0, khôngdạy/restoremain.',evidence={'run':result['run'],'source_count':len(sources),'causal_rows':len(rows),'all_transplants_exact':True})
        print(json.dumps({'run':result['run'],'source_count':len(sources),'rows':rows,'resources':result['resources']},ensure_ascii=False))


if __name__=='__main__':main()
