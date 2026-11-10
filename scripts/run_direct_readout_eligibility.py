"""Preregistered proposal30 screen; no new serving model or globalbackward candidate."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import sys,json,time,itertools,subprocess,hashlib,argparse,zipfile,ast
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
import numpy as np
from direct_eligibility_common import TASKS,METHODS,DEV,CONFIRM
from direct_eligibility_worker import trial
from visual_word_control_common import atomic
from scarx.evaluation.runs import new_run,write_json
from scarx.evaluation.phase21 import interval,signflip,holm
from scarx.runtime.work_trace import activity
from scarx.runtime.experiment_status import progress
from scarx.runtime.resources import ResourceMonitor
from scarx.runtime.limits import apply_memory_limit
from scarx.runtime.atomic_file import read_text_with_retry
from scarx.runtime.experiment_status import RUN_ID,_process


def child(job,run,monitor,deadline):
    phase,task,seed,method,_=job['trial'];mode=job['mode'];identifier=f'{phase}-{task}-{seed}-{method}';path=run/f'job-{identifier}-{mode}.json';atomic(path,job)
    out=(run/f'{identifier}-{mode}.stdout.log').open('wb');err=(run/f'{identifier}-{mode}.stderr.log').open('wb');started=time.perf_counter()
    p=subprocess.Popen([sys.executable,str(root/'scripts/direct_eligibility_worker.py'),'--job',str(path)],cwd=root,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    signature=None
    try:
        while p.poll() is None:
            monitor.check()
            if time.perf_counter()>deadline:raise TimeoutError('Registered eligibility childdeadline')
            beat=run/f'heartbeat-{identifier}.json'
            if mode=='train' and beat.is_file() and beat.stat().st_size<65536:
                r=json.loads(read_text_with_retry(beat))
                if r.get('examples')!=signature:
                    signature=r['examples'];progress(run,f'{identifier} · {signature}/1024',metrics={k:r[k] for k in ('task','seed','method','examples','loss_mean','hidden_updates','elapsed_seconds')},neural=r['neural'])
            time.sleep(.25)
        if p.returncode:raise RuntimeError(f'{identifier}/{mode} failed; actual logs retained')
    finally:
        if p.poll() is None:
            p.terminate()
            try:p.wait(timeout=5)
            except subprocess.TimeoutExpired:p.kill();p.wait()
        out.close();err.close()
    return time.perf_counter()-started


def cold_check(row,run,monitor,deadline,prior_cost=0.):
    job=(row['phase'],row['task'],row['seed'],row['method'],str(run));checkpoint=root/row['checkpoint']
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==row['checkpoint_sha256']
    seconds=child({'mode':'cold','trial':job,'checkpoint':row['checkpoint'],'sha256':row['checkpoint_sha256']},run,monitor,min(deadline,time.perf_counter()+max(1.,180-row['seconds']-prior_cost)))
    path=run/f"cold-{row['phase']}-{row['task']}-{row['seed']}-{row['method']}.json"
    assert json.loads(path.read_text(encoding='utf-8'))==row.pop('cold_expected') and hashlib.sha256(checkpoint.read_bytes()).hexdigest()==row['checkpoint_sha256']
    row['cold_exact']=True;row['active_seconds_including_cold']=row['seconds']+seconds+prior_cost;row['prior_attempt_cost_charged']=prior_cost
    if row['active_seconds_including_cold']>180:raise TimeoutError('Eligibilitytrial incl cold exceeded180s')


def run_phase(phase,seeds,run,monitor,deadline,reused=(),prior_charges=None):
    rows=[dict(r) for r in reused if r['phase']==phase];keys={(r['task'],r['seed'],r['method']) for r in rows};prior_charges=prior_charges or {}
    local=[(phase,t,s,c,str(run)) for t,s,c in itertools.product(TASKS,seeds,METHODS[:-1]) if (t,s,c) not in keys];pending={};todo=iter(local);signature=None
    progress(run,phase+' · localeligibility trials',0,len(seeds)*15)
    with ProcessPoolExecutor(max_workers=2) as pool:
        for _ in range(2):
            try:job=next(todo);pending[pool.submit(trial,job)]=job
            except StopIteration:break
        while pending:
            ready,_=wait(pending,timeout=.25,return_when=FIRST_COMPLETED);monitor.check()
            if time.perf_counter()>deadline:raise TimeoutError('Registered eligibilityphase deadline')
            beats=[run/f"heartbeat-{'-'.join(map(str,job[:4]))}.json" for job in pending.values()]
            beats=[p for p in beats if p.is_file()]
            if beats:
                newest=max(beats,key=lambda p:p.stat().st_mtime)
                if newest.stat().st_size<65536:
                    value=json.loads(read_text_with_retry(newest));key=(newest.name,value['examples'])
                    if key!=signature:
                        signature=key;progress(run,f"{phase}/{value['task']}/seed{value['seed']}/{value['method']} · {value['examples']}/1024",len(rows),len(seeds)*15,
                            metrics={k:value[k] for k in ('task','seed','method','examples','loss_mean','hidden_updates','elapsed_seconds')},neural=value['neural'])
            for future in ready:
                row,neural=future.result();pending.pop(future);cold_check(row,run,monitor,deadline);rows.append(row);atomic(run/(phase+'-progress.json'),rows)
                progress(run,f"{phase}: {row['task']}/{row['seed']}/{row['method']} done/coldexact",len(rows),len(seeds)*15,metrics={k:row[k] for k in ('task','seed','method','direct','probe_gain','train_seconds')},neural=neural)
                print(json.dumps({k:row[k] for k in ('phase','task','seed','method','direct','probe_gain')}),flush=True)
                try:job=next(todo);pending[pool.submit(trial,job)]=job
                except StopIteration:pass
    # NumPyworkers have exited before any Torchchild, keeping totalmemory bounded.
    for task,seed in itertools.product(TASKS,seeds):
        if (task,seed,'bptt_control') in keys:continue
        job=(phase,task,seed,'bptt_control',str(run));progress(run,f'{phase}/{task}/{seed}/BPTT CONTROL initializing',len(rows),len(seeds)*15)
        prior_cost=prior_charges.get(f'{phase}-{task}-{seed}-bptt_control',0.)
        child({'mode':'train','trial':job},run,monitor,min(deadline,time.perf_counter()+180-prior_cost))
        row=json.loads(read_text_with_retry(run/f'{phase}-{task}-{seed}-bptt_control.json'))['record'];cold_check(row,run,monitor,deadline,prior_cost);rows.append(row);atomic(run/(phase+'-progress.json'),rows)
        progress(run,f'{phase}/{task}/{seed}/BPTT CONTROL done/coldexact',len(rows),len(seeds)*15,metrics={k:row[k] for k in ('task','seed','method','direct','probe_gain')})
        print(json.dumps({k:row[k] for k in ('phase','task','seed','method','direct','probe_gain')}),flush=True)
    for task,seed in itertools.product(TASKS,seeds):
        zero=next(r for r in rows if r['task']==task and r['seed']==seed and r['method']=='zero');frozen=next(r for r in rows if r['task']==task and r['seed']==seed and r['method']=='frozen')
        assert zero['core_weight_fingerprint']==frozen['core_weight_fingerprint'] and zero['head_sha256']==frozen['head_sha256'] and all(zero[k]==frozen[k] for k in ('direct','probe','probe_gain'))
    return rows


def summary(rows):
    return {t:{method:{k:float(np.mean([r[k] for r in rows if r['task']==t and r['method']==method])) for k in ('direct','probe','probe_gain','bag_only','train_seconds','probe_time_permuted','probe_state_reset')} for method in METHODS} for t in TASKS}


def recovery_source(identifier):
    if not RUN_ID.fullmatch(identifier):raise ValueError('Invalid original run')
    old=root/'runs'/identifier;journal=json.loads(read_text_with_retry(root/'local/experiment-status'/(identifier+'.json')))
    if journal['status']!='stopped_without_result' or _process(journal)[0] or (old/'result.json').exists():raise ValueError('Original run not verified terminal without finalresult')
    import psutil
    for p in psutil.process_iter(['cmdline','name']):
        command=p.info['cmdline'] or []
        if (p.info['name'] or '').lower() in ('python.exe','pythonw.exe','python') and any(Path(a).name=='direct_eligibility_worker.py' for a in command) and any(identifier in a for a in command):raise ValueError('Original owned worker remains live')
    manifest=json.loads(read_text_with_retry(old/'manifest.json'))
    if manifest['config'].get('technical_recovery'):raise ValueError('Only one IO recovery, not a recoverychain')
    for name in ('src/scarx/evaluation/readout_eligibility.py','src/scarx/evaluation/controls.py','src/scarx/evaluation/readout_conditioning.py','src/scarx/neural/core.py','src/scarx/evaluation/tasks21.py','src/scarx/evaluation/binding_audit21.py','scripts/direct_eligibility_common.py','scripts/direct_eligibility_worker.py','experiments/direct-readout-eligibility-v1.md'):
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=manifest['source_sha256'][name]:raise ValueError('IO recovery changed neural/evaluation math or protocol: '+name)
    with zipfile.ZipFile(old/'source.zip') as z:before=z.read('scripts/run_direct_readout_eligibility.py').decode('utf-8')
    after=Path(__file__).read_text(encoding='utf-8')
    def gateblock(source):
        node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='main');body=ast.get_source_segment(source,node)
        return body.split('        valid={t:',1)[1].split('        actual=confirmation',1)[0]
    if gateblock(before)!=gateblock(after):raise ValueError('Recovery changed selection/confirmation/gate math')
    rows=json.loads(read_text_with_retry(old/'development-progress.json'))
    if len(rows)!=38:raise ValueError('Expected38 completed trials from original IO failure')
    for r in rows:
        name=f"{r['phase']}-{r['task']}-{r['seed']}-{r['method']}";record=json.loads(read_text_with_retry(old/(name+'.json')))['record']
        if not r.get('cold_exact') or r['experiences']!=1024 or hashlib.sha256((root/r['checkpoint']).read_bytes()).hexdigest()!=r['checkpoint_sha256']:raise ValueError('Recovered trial/checkpoint not complete')
        if json.loads(read_text_with_retry(old/('cold-'+name+'.json')))!=record['cold_expected']:raise ValueError('Original coldproof mismatch')
        r['reused_without_retraining']=True;r['source_run']='runs/'+identifier;r['source_measurement_file']='runs/'+identifier+'/'+name+'.json'
    missing='development-order-35141-bptt_control'
    if (root/'artifacts/direct-readout-eligibility'/identifier/(missing+'.eligibility.zip')).exists():raise ValueError('Unexpected completed checkpoint for missingtrial; do not retrain')
    heartbeat=json.loads(read_text_with_retry(old/('heartbeat-'+missing+'.json')));active=journal['updated']-journal['started'];calendar=time.time()-journal['started']
    if not 0<active<180 or not active<=calendar<1800:raise TimeoutError('Original trial/round budget unavailable')
    info={'source_run':'runs/'+identifier,'failure_kind':'Windows_PermissionError_reading_atomic_heartbeat','recovery_count':1,'completed_trials_reused':38,
        'original_active_seconds':active,'original_and_repair_calendar_seconds_charged_to_round':calendar,'repair_calendar_seconds':time.time()-journal['updated'],
        'partial_trial_restarted':missing,'lost_partial_observed_examples_lower_bound':heartbeat['examples'],'lost_partial_exact_example_count':None,
        'entire_original_active_seconds_conservatively_charged_to_first_restarted_trial':active,'original_parent_tree_peak_memory_unknown':True,
        'repair_scope':'Bounded readPermissionError retry and reuse of38completed/coldverified trials ONLY; unchanged algorithm/data/hyperparameters/gates.'}
    if not (old/'failure.json').exists():atomic(old/'failure.json',info)
    atomic(root/'artifacts/direct-eligibility-read-io-failure.json',info)
    return rows,info,{missing:active}


def main():
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser();parser.add_argument('--recover-read-io');args=parser.parse_args();reused=[];recovery=None;prior_charges={};prior_calendar=0.
    if args.recover_read_io:
        reused,recovery,prior_charges=recovery_source(args.recover_read_io);prior_calendar=recovery['original_and_repair_calendar_seconds_charged_to_round']
    cap=apply_memory_limit();began=time.perf_counter()-prior_calendar;recovery_started=time.perf_counter();engineering=json.loads((root/'artifacts/direct-eligibility-engineering.json').read_text(encoding='utf-8'))
    assert engineering['formal_screen_permitted'] and engineering['proposal30_approved']
    manifest=json.loads((root/engineering['run']/'manifest.json').read_text(encoding='utf-8'))
    for name in ('src/scarx/evaluation/readout_eligibility.py','src/scarx/evaluation/controls.py','src/scarx/evaluation/readout_conditioning.py','src/scarx/neural/core.py','src/scarx/evaluation/tasks21.py','src/scarx/evaluation/binding_audit21.py','scripts/direct_eligibility_common.py','experiments/direct-readout-eligibility-v1.md'):
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==manifest['source_sha256'][name],name
    run=new_run(root,'direct-readout-eligibility-v1',{'protocol':'experiments/direct-readout-eligibility-v1.md','display_title':'PROPOSAL30 · one-stepreadouterror/localeligibility',
        'model_scope':'APPROVED RESEARCH, no deployment. SameALIF, forwardown-neuron traces; BPTT onlyisolatedcontrol. Numerictemporalclassifiers, NOT normalchat/vision.',
        'proposal30_approved':True,'development_seeds':DEV,'confirmation_seeds':CONFIRM,'tasks':TASKS,'methods':METHODS,'examples_per_trial':1024,
        'engineering_run':engineering['run'],'num_workers':2,'torch_children_at_once':1,'technical_recovery':recovery})
    (run/'report.md').write_text('# Proposal30 localeligibility screen\n\nActual training in progress, no mainquality/deployment claim. All criteria fixed before scores.\n',encoding='utf-8')
    activity(run,'Bắt đầu45thử nghiệm cơ chế30 đãduyệt','P0math64/partialcheckpoint/zero-frozen qua.3freshseed×3task×5methods,1024experiences. '+(f'IO recovery: {len(reused)}complete trials giữnguyên, khônghọc lại. ' if recovery else '')+'MainALIF/Hermesgiữ nguyên; khôngTransformer/networkbackwardcandidate.',status='started',evidence={'engineering':engineering['run'],'registered_protocol':'experiments/direct-readout-eligibility-v1.md','formal_trials':45,'completed_reused':len(reused),'technical_recovery':recovery})
    confirmation=[];stats={}
    with ResourceMonitor(interval=.1) as monitor:
        development=run_phase('development',DEV,run,monitor,began+1800,reused,prior_charges);atomic(run/'development.json',development);screen_summary=summary(development)
        valid={t:all(r['bag_only']<=.6 for r in development if r['task']==t) for t in TASKS}
        screen_tasks={}
        for task in TASKS:
            correct=screen_summary[task]['correct'];frozen=screen_summary[task]['frozen']
            screen_tasks[task]={'valid_bag_baseline':valid[task],'mean_probe_gain':correct['probe_gain'],'mean_direct':correct['direct'],
                'mean_probe_vs_frozen':correct['probe']-frozen['probe'],'pass':valid[task] and correct['probe_gain']>=.05 and correct['direct']>=.65 and correct['probe']>frozen['probe']}
        screen=sum(r['pass'] for r in screen_tasks.values())>=2 and any(screen_tasks[t]['pass'] for t in ('xor','strict_binding'))
        selection={'fixed_rule_no_parameter_selection':True,'screen_tasks':screen_tasks,'screen_pass':screen,'confirmation_opened':False}
        atomic(run/'screen-locked.json',selection)
        forecast_confirm=96*engineering['timing']['correct']['forecast_trial_with25pct_headroom']+24*engineering['timing']['bptt_control']['forecast_trial_with25pct_headroom']
        if screen and forecast_confirm<3600:
            selection['confirmation_opened']=True;confirmation=run_phase('confirmation',CONFIRM,run,monitor,time.perf_counter()+3600);atomic(run/'confirmation.json',confirmation)
            for task in TASKS:
                full=[r for r in confirmation if r['task']==task and r['method']=='correct'];gain=[r['probe_gain'] for r in full]
                def contrast(method):return [r['probe']-next(b['probe'] for b in confirmation if b['task']==task and b['seed']==r['seed'] and b['method']==method) for r in full]
                diff=contrast('frozen');stats[task]={'probe_gain':interval(gain),'direct':interval([r['direct'] for r in full]),'vs_frozen':interval(diff),'vs_shuffled':interval(contrast('shuffled')),
                    'gain_p':signflip(gain),'vs_frozen_p':signflip(diff),'valid_bag_baseline':all(r['bag_only']<=.6 for r in confirmation if r['task']==task)}
            gp=holm([stats[t]['gain_p'] for t in TASKS]);fp=holm([stats[t]['vs_frozen_p'] for t in TASKS])
            for i,t in enumerate(TASKS):
                v=stats[t];v['holm_gain_p']=gp[i];v['holm_vs_frozen_p']=fp[i]
                v['pass']=v['valid_bag_baseline'] and v['probe_gain']['lower95']>=.05 and v['direct']['mean']>=.65 and v['vs_frozen']['mean']>0 and gp[i]<=.05 and fp[i]<=.05
        gate=sum(r.get('pass',False) for r in stats.values())>=2 and any(stats.get(t,{}).get('pass',False) for t in ('xor','strict_binding'))
        decision='GO_DIRECT_ELIGIBILITY_P2_RESEARCH_ONLY' if gate else 'CONFIRM_DIRECT_READOUT_ELIGIBILITY_REQUIRED' if screen and not confirmation else 'STOP_DIRECT_READOUT_ELIGIBILITY_V1'
        actual=confirmation or development;result={'run':run.relative_to(root).as_posix(),'decision':decision,'proposal30_approved':True,'selection':selection,'confirmation_statistics':stats,
            'control_summary':summary(actual),'primary_representation_gate':gate,'screen_trials':len(development),'confirmation_trials':len(confirmation),
            'all_cold_exact':True,'zero_frozen_equivalence':True,'deployed':False,'main_weights_changed':False,'normal_chat_or_vision_proven':False,'retention_proven':False,
            'candidate_network_backward':False,'candidate_signal_locality':'One learned readout-error transport, own-neuron forwardtrace; not fullybiological local/exactgradient.',
            'engineering_run':engineering['run'],'resources':monitor.report(),'cap':cap,'seconds':time.perf_counter()-began,
            'limits':['Fresh3developmentseeds; confirmation8onlyifscreenpassed. Order isweakrecency/delayed task, not complexreasoning.',
                'Readout-only localdelta and correct/shuffled/zero hiddenablations; BPTTcontrol separate extra graphcost, not main.',
                'No word/vision/retention/modelscale or deployment authorization follows from toyrepresentationgate.']}
        result.update(technical_recovery=recovery,completed_trials_reused_without_retraining=len(reused),recovery_active_seconds=time.perf_counter()-recovery_started,
            original_and_repair_calendar_seconds_charged=prior_calendar,resource_peak_scope='Recovery process-tree only; original failed parent/losttrial peak unknown' if recovery else 'Full registered run')
        (run/'report.md').write_text('# Approved proposal30 research\n\n'+decision+'\n\nNo productioncore/weightschange; no normalchat/vision/P2claim.\n\n'+json.dumps(result,indent=2)+'\n',encoding='utf-8')
        write_json(run/'result.json',result);write_json(root/'artifacts/direct-eligibility-latest.json',result)
        activity(run,'Hoàn tất cơ chế30 boundedresearch',decision+'; actualtrials/coldresumes/ablation giữnguồn. Không đổiqualitygate hoặc đưa candidate/control vàoHermes.',evidence={'decision':decision,'screen_trials':len(development),'confirmation_trials':len(confirmation),'primary_representation_gate':gate})
        print(json.dumps({'run':result['run'],'decision':decision,'selection':selection,'confirmation_statistics':stats,'control_summary':result['control_summary'],'resources':result['resources']},ensure_ascii=False))


if __name__=='__main__':main()
