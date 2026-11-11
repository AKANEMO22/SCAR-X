"""Pinned R7 source/raw review and one historical positive-task verification."""
from pathlib import Path
import hashlib,json,os,subprocess,sys
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from scarx.evaluation.forensic.reference_evaluator import seed_bootstrap
from scarx.evaluation.runs import new_run,write_json
from scarx.runtime.limits import apply_memory_limit
from forensic_run_r6 import run_child

GEMINI=ROOT.parent/'review-gemini-r7-64441fb'
DEEPSEEK=ROOT.parent/'review-deepseek-r7-aacde3f'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def stats():
    d=read(GEMINI/'artifacts/gemini-r7/dev-screen.json');tasks={}
    for task in sorted({r['task'] for r in d['dev_p1_trials']}):
        rows=[r for r in d['dev_p1_trials'] if r['task']==task]
        means={c:{m:float(np.mean([r['conditions'][c][m] for r in rows])) for m in ('id_direct_acc','id_probe_acc','ood_probe_acc')}
               for c in ('frozen_hidden','r35_baseline','learned_ew_he','learned_cg_mhe')}
        contrast=lambda metric,base:seed_bootstrap({r['seed']:r['conditions']['learned_ew_he'][metric]-r['conditions'][base][metric] for r in rows})
        tasks[task]={'means':means,'hidden_probe_gain':contrast('id_probe_acc','frozen_hidden'),
            'ood_hidden_probe_gain':contrast('ood_probe_acc','frozen_hidden'),
            'gain_over_r35_wrapper':contrast('id_probe_acc','r35_baseline')}
    paired=read(GEMINI/'artifacts/gemini-r7/paired-orders-18cell.json');cells=paired['cells']
    def gate(r):return r['acquisition_accuracy']>=.65 and r['final_accuracy']>=.65 and r['mean_forgetting']<=.03 and r['worst_forgetting']<=.05
    retention={arm:{'cells_passing_joint_gate':sum(gate(c[arm]) for c in cells),'cells':len(cells),
        'means':{m:float(np.mean([c[arm][m] for c in cells])) for m in ('acquisition_accuracy','final_accuracy','mean_forgetting','worst_forgetting')}}
        for arm in ('unconsolidated_no_replay','ew_he_ca_msc')}
    clustered={}
    for metric in ('mean_forgetting','final_accuracy','acquisition_accuracy'):
        effects={seed:float(np.mean([c['ew_he_ca_msc'][metric]-c['unconsolidated_no_replay'][metric] for c in cells if c['seed']==seed])) for seed in paired['dev_seeds']}
        clustered[metric]=seed_bootstrap(effects)
    deep={}
    for name,file in [('three_tasks','nonlinear-90cells.json'),('six_tasks_corrected','t6-corrected.json')]:
        data=read(DEEPSEEK/'artifacts/deepseek-r7'/file);by_order=[]
        for order in data['orders']:
            rs=[r['arms']['replay_tail'] for r in data['cells'] if r['order']==order]
            ok=[r['acq']>=.65 and r['final']>=.65 and r['meanF']<=3 and r['worstF']<=5 for r in rs]
            by_order.append({'order':order,'seeds_passing_all_gates':sum(ok),'seeds':len(rs),
                'meanF_pp':float(np.mean([r['meanF'] for r in rs])),'mean_worstF_pp':float(np.mean([r['worstF'] for r in rs])),
                'mean_acquisition':float(np.mean([r['acq'] for r in rs]))})
        deep[name]={'by_order':by_order,'total_cells':len(data['cells']),
            'total_passing_cells':sum(row['seeds_passing_all_gates'] for row in by_order),
            'hidden_alif_frozen':True,'readout':'32tanh-unit two-layer MLP, manual gradient through readout'}
    return {'gemini_p1':tasks,'gemini_retention':retention,'gemini_retention_seed_cluster_contrasts':clustered,
        'gemini_raw_claimed_hypotheses':d['hypothesis_decisions'],'deepseek':deep,
        'all_R7_past_training_reproduced':False}


def main():
    pins={GEMINI:'64441fb9be162864cf2e1252638d101a2031f2e4',DEEPSEEK:'aacde3fa0983b9cebc585e19980022a52a45ccad'}
    for path,sha in pins.items():
        if subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()!=sha:raise ValueError('Source moved')
    cap=apply_memory_limit()
    run=new_run(ROOT,'r7-progress-review-v1',{'display_title':'R7 latest branches · independent raw review and bounded historical verification',
        'scope':'No new candidate/seed/confirmation/P2; one selected positive historical P1 cell only',
        'sources':{str(p):s for p,s in pins.items()}},data_description='Existing public synthetic raw trial/cell records and already-scored source configurations.',
        data_files=['docs/research/forensic-r6-20261111/R7_PROGRESS_REVIEW_SCOPE.md'])
    result=stats()
    result['p0_command']=run_child(run,'gemini-r7-p0',[sys.executable,'-m','pytest','-q','-p','no:cacheprovider','tests/test_r7_hidden_plasticity.py','--junitxml='+str(run/'gemini-r7-p0.xml')],GEMINI,120)
    if not result['p0_command']['error']:
        code="""from pathlib import Path
import hashlib,json,sys
from scarx.runtime.limits import apply_memory_limit
from scarx.evaluation import r7_hidden_plasticity as R
cap=apply_memory_limit()
value=R.run_p1_task_seed('strict_binding',140011)
old=json.loads(Path('artifacts/gemini-r7/dev-screen.json').read_text())
expected=[r for r in old['dev_p1_trials'] if r['task']=='strict_binding' and r['seed']==140011][0]['conditions']
metrics=('id_direct_acc','ood_direct_acc','id_probe_acc','ood_probe_acc')
comparison={c:{k:value[c].get(k)==expected[c].get(k) for k in metrics} for c in expected}
result={'run_class':'HISTORICAL_SELECTED_POSITIVE_TASK_REPRODUCTION','seed':140011,'task':'strict_binding','all_primary_scores_exact':all(all(v.values()) for v in comparison.values()),'comparison':comparison,'conditions':value,'cap':cap,'fresh_confirmation':False,'whole_R7_campaign_reproduced':False}
Path(sys.argv[1]).write_text(json.dumps(result,indent=2)+'\\n')
print(json.dumps({'all_primary_scores_exact':result['all_primary_scores_exact'],'learned_id_probe':value['learned_ew_he']['id_probe_acc'],'frozen_id_probe':value['frozen_hidden']['id_probe_acc']}))
"""
        result['historical_check_command']=run_child(run,'gemini-r7-positive-history',[sys.executable,'-c',code,str(run/'positive-history.json')],GEMINI,180)
        if (run/'positive-history.json').exists():result['historical_positive_task']=read(run/'positive-history.json')
    result['run']=run.relative_to(ROOT).as_posix();result['cap']=cap
    result['main_or_live_changed']=False;result['new_confirmation_opened']=False
    write_json(run/'result.json',result)
    doc=ROOT/'docs/research/forensic-r6-20261111/R7_PROGRESS_RECALCULATION.json';write_json(doc,result)
    (run/'report.md').write_text('# R7 review evidence\n\nRaw metrics and bounded engineering/reproduction, not production approval.\n\n'+json.dumps({k:v for k,v in result.items() if k.endswith('command')},indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'run':run.name,'p0_error':result['p0_command']['error'],'positive_history_scores_exact':result.get('historical_positive_task',{}).get('all_primary_scores_exact')}),flush=True)


if __name__=='__main__':main()
