"""Pin/read primary repository sources and licenses. Never install/run them."""
from pathlib import Path
import concurrent.futures,hashlib,json,os,sys,time,urllib.request,urllib.error
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
from scarx.evaluation.runs import new_run,write_json
from scarx.runtime.experiment_status import progress
from public_research_mirror import publish

REPOS={
 'e-prop':('IGITUGraz/eligibility_propagation',('spiking_models','tutorial','eprop','alif','eligibility')),
 'DECOLLE':('nmi-lab/decolle-public',('base_model','decolle.py','utils.py')),
 'BindsNET':('BindsNET/bindsnet',('learning/learning.py','network/nodes.py')),
 'ContinualAI':('ContinualAI/continual-learning-baselines',('ewc','synaptic','gem','requirements','config')),
 'SuperSpike':('fzenke/pub2018superspike',('superspike','supervised','spike')),
 'SpikingJelly':('fangwei123456/spikingjelly',('activation_based/learning.py','activation_based/neuron.py','activation_based/surrogate.py'))}


def fetch(url,limit):
    request=urllib.request.Request(url,headers={'User-Agent':'SCAR-X-research-source-audit','Accept':'application/vnd.github+json' if 'api.github.com' in url else 'text/plain'})
    with urllib.request.urlopen(request,timeout=20) as response:
        raw=response.read(limit+1)
        if len(raw)>limit:raise ValueError('Bounded source response exceeded')
        return raw


def audit_one(name,slug,patterns,cache):
    record={'method':name,'repository':'https://github.com/'+slug,'code_executed':False,'code_reused':False,'license_review_required_before_reuse':True,'files':[]}
    try:
        repository=json.loads(fetch('https://api.github.com/repos/'+slug,300_000))
        if repository.get('private'):raise ValueError('Private source not allowed')
        branch=repository['default_branch'];commit=json.loads(fetch(f'https://api.github.com/repos/{slug}/commits/{branch}',800_000));sha=commit['sha']
        tree=json.loads(fetch(f'https://api.github.com/repos/{slug}/git/trees/{sha}?recursive=1',3_000_000))
        if tree.get('truncated'):raise ValueError('Truncated repository tree, no completeness claim')
        files={r['path']:r for r in tree['tree'] if r['type']=='blob'}
        record.update(revision=sha,default_branch=branch,license_spdx=(repository.get('license') or {}).get('spdx_id'),commit_date=commit['commit']['committer']['date'],tree_entries=len(tree['tree']))
        directory=cache/name/sha;directory.mkdir(parents=True)
        licenses=[p for p in files if '/' not in p and (Path(p).name.lower().startswith(('license','copying')))]
        readmes=[p for p in files if '/' not in p and p.lower().startswith('readme')]
        candidates=[p for p,v in files.items() if p.endswith(('.py','.cpp','.h','.cc','.toml','.txt','.yaml','.yml')) and any(word in p.lower() for word in patterns) and v.get('size',0)<700_000 and not any(s in p.lower() for s in ('test','example','benchmark'))]
        candidates.sort(key=lambda p:(next((i for i,word in enumerate(patterns) if word in p.lower()),99),len(p),p))
        selected=list(dict.fromkeys(licenses+readmes[:1]+candidates[:4]));total=0
        # License text is downloaded first; all source stays an inert local cache.
        for relative in selected:
            if files[relative].get('size',0)>700_000:record.setdefault('skipped_size',[]).append(relative);continue
            raw=fetch(f'https://raw.githubusercontent.com/{slug}/{sha}/{relative}',700_000);total+=len(raw)
            if total>1_500_000:raise ValueError('Per-repository1.5MB source cap')
            destination=(directory/relative).resolve()
            if not destination.is_relative_to(directory.resolve()):raise ValueError('Source cache path escape')
            destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(raw)
            record['files'].append({'path':relative,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'url':f'https://github.com/{slug}/blob/{sha}/{relative}','local_cache':destination.relative_to(root).as_posix(),'role':'license' if relative in licenses else 'readme' if relative in readmes else 'implementation'})
        record.update(status='PRIMARY_SOURCE_CACHED_REVIEW_PENDING',bytes=total,license_files_found=len(licenses),error=None)
    except (OSError,ValueError,KeyError,urllib.error.HTTPError) as exc:record.update(status='SOURCE_AUDIT_INCOMPLETE',error=type(exc).__name__+': '+str(exc)[:220])
    return record


def main():
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    started=time.perf_counter();run=new_run(root,'external-learning-sources-v2',{'display_title':'6primary SNN/CL repos · pin revision/license/source · no training','observability_kind':'research_audit','model_scope':'Read-only public primary sources, inert cache. No externalframework execution/import, no newlearningrule or pilot.','stage':1,'repositories':list(REPOS),'max_workers':3,'source_limit_per_repo_bytes':1_500_000},data_description='GitHubpublic metadata/license/selectedalgorithmfiles only; no userhistory/weights uploaded.')
    cache=root/'local/external-learning-sources';cache.mkdir(parents=True,exist_ok=True)
    (run/'report.md').write_text('# External learning source audit\n\nMetadata/cache in progress; mathematical/license review not yet claimed.\n',encoding='utf-8');publish(run)
    records=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures={executor.submit(audit_one,name,slug,patterns,cache):name for name,(slug,patterns) in REPOS.items()}
        for future in concurrent.futures.as_completed(futures):
            row=future.result();records.append(row);progress(run,'Đã pin nguồn '+row['method'],len(records),6,metrics={'sources_cached':sum(r['status']=='PRIMARY_SOURCE_CACHED_REVIEW_PENDING' for r in records),'training_runs':0});publish(run)
            print(json.dumps({k:row.get(k) for k in ('method','revision','license_spdx','status','bytes','error')}),flush=True)
    result={'run':run.relative_to(root).as_posix(),'decision':'EXTERNAL_PRIMARY_SOURCES_REVIEW_PENDING','stage':1,'sources':sorted(records,key=lambda r:r['method']),'all_sources_cached':all(r['status']=='PRIMARY_SOURCE_CACHED_REVIEW_PENDING' for r in records),'mathematical_comparison_complete':False,'code_reused':False,'code_executed':False,'main_training':False,'pilot_run':False,'seconds':time.perf_counter()-started}
    write_json(run/'result.json',result);write_json(root/'artifacts/external-learning-sources-v2.json',result)
    (run/'report.md').write_text('# External primary sources pinned\n\nSources are inert cache, not executed/reused. License and mathematical comparison remain to review.\n\n'+json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');publish(run)
    return int(not result['all_sources_cached'])


if __name__=='__main__':raise SystemExit(main())
