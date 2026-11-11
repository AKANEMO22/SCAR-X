"""Inspect historical raw synthetic splits; no neural model or fitting."""
from pathlib import Path
import hashlib,importlib.util,json,os,subprocess,sys
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT.parent/'forensic-source-r6d-4bacd7b'
sys.path.insert(0,str(SOURCE/'src'))
import numpy as np
from scarx.evaluation import r5_independent as G
from scarx.runtime.resources import ResourceMonitor
from scarx.runtime.limits import apply_memory_limit


def main():
    spec=importlib.util.spec_from_file_location('independent_ref',ROOT/'src/scarx/evaluation/forensic/reference_evaluator.py')
    ref=importlib.util.module_from_spec(spec);sys.modules[spec.name]=ref;spec.loader.exec_module(ref)
    seeds=(130001,130007,130013,130019,130023,130029,130031,130037)
    cap=apply_memory_limit();rows=[]
    with ResourceMonitor(interval=.02) as monitor:
        for seed in seeds:
            sets={};logical={};digests={}
            for name,offset,n in [('train_1024',200000,1024),('train_16384',200000,16384),('calibration',700000,256),('held',300000,1024)]:
                x,y,a,b,q,_=G.independent_query_set_v1(seed+offset,n)
                obs=ref.oracle_events(x)
                if not np.array_equal(obs['labels'],y):raise ValueError('Actual source oracle/labels mismatch')
                sets[name]=ref.sample_fingerprints(x)
                logical[name]=sorted(set(zip(a.tolist(),b.tolist(),q.tolist())))
                digests[name]=G.dataset_sha256(x,y)
                monitor.check()
            xp,yp,*_=G.independent_paired_set_v1(seed+400000,256);sets['paired']=ref.sample_fingerprints(xp)
            disjoint={}
            for left,right in [('train_16384','held'),('train_16384','calibration'),('train_16384','paired'),('calibration','held'),('held','paired')]:
                count=len(sets[left]&sets[right]);disjoint[left+'__'+right]=count
                if count:raise ValueError('Raw split overlap in '+left+' '+right)
            rows.append({'seed':seed,'dataset_digests':digests,'exact_episode_cross_split_overlap':disjoint,
                'train_1024_vs_16384_common_episodes':len(sets['train_1024']&sets['train_16384']),
                'small_training_is_subset_of_large':sets['train_1024']<=sets['train_16384'],
                'logical_states_per_split':{k:len(v) for k,v in logical.items()},'logical_state_definition':['vA','vB','query'],
                'raw_training_rows_oracle_checked':17408})
    # Search all pinned result records, not only DEV/FRESH constants in one file.
    inv=json.loads((ROOT/'docs/research/forensic-r6-20261111/BASELINE_INVENTORY.json').read_text())
    def collect(d,output,key=''):
        if isinstance(d,dict):
            for k,v in d.items():collect(v,output,k)
        elif isinstance(d,list):
            for v in d:collect(v,output,key)
        elif type(d) is int and key in ('seed','seeds','model_seed','model_seeds','fresh_seeds','confirmation_seeds'):output.add(d)
    mentions=[];seen_blobs=set();project=ROOT.parents[2]
    for branch in inv['branches']:
        for file in branch['result_files']:
            if file['git_blob'] in seen_blobs:continue
            seen_blobs.add(file['git_blob'])
            data=json.loads(subprocess.check_output(['git','-C',str(project),'show',branch['sha']+':'+file['path']]))
            used=set();collect(data,used);overlap=sorted(used&set(seeds))
            if overlap:mentions.append({'branch':branch['branch'],'sha':branch['sha'],'file':file['path'],'seeds':overlap,
                'is_r6d_historical_run':branch['branch']=='research/deepseek/r6d-audit'})
    result={'classification':'HISTORICAL_DATA_LINEAGE_AUDIT','seeds':list(seeds),'per_seed':rows,
        'seed_mentions_in_pinned_result_files':mentions,'unique_result_blobs_scanned':len(seen_blobs),
        'seed_freshness_scope':'No use of these seeds found in other pinned result blobs; not a proof about unpublished/private probes or all ancestor work before registration.',
        'source_sha':'4bacd7b2c8a1df7ab47e55f266f2b6bf0e193893','resources':monitor.report(),'cap':cap,
        'neural_forward_calls':0,'learning_updates':0,'logical_state_coverage_is_not_knowledge_count':True}
    target=ROOT/'docs/research/forensic-r6-20261111/DATA_LINEAGE.json'
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'output':str(target),'split_overlaps':sum(sum(r['exact_episode_cross_split_overlap'].values()) for r in rows),
        'train_ladder_nested':[r['small_training_is_subset_of_large'] for r in rows],'source_runs_with_seed_mentions':len(mentions)}))


if __name__=='__main__':main()
