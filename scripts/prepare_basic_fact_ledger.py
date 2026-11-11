"""Extract fixed public facts from source AST; never generate answer labels."""
from pathlib import Path
import os,sys,json,hashlib,subprocess,urllib.request,time
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
from scarx.evaluation.knowledge_factory import PUBLIC_COMMIT,CORE,UNIFIED,SERVER,APP,extract_facts
from scarx.evaluation.runs import new_run,write_json
from public_research_mirror import publish


def main():
    started=time.perf_counter();run=new_run(root,'basic-fact-ledger-v1',{'protocol':'experiments/basic-knowledge-factory-v1.md','display_title':'40public facts · AST/source verification · no neurallearning','observability_kind':'data_generation','stage':2,'source_commit':PUBLIC_COMMIT,'known_facts':32,'outside_domain_facts':8},data_description='User-owned publicGitHub source at fixed495dab commit; no private data.')
    directory=root/'data/candidates'/run.name;(directory/'sources').mkdir(parents=True);sources={};records=[]
    (run/'report.md').write_text('# Public factledger\n\nPublicsource fetch/AST checks in progress; not traininglabels.\n',encoding='utf-8');publish(run)
    for filename in (CORE,UNIFIED,SERVER,APP):
        url=f'https://raw.githubusercontent.com/AKANEMO22/SCAR-X/{PUBLIC_COMMIT}/{filename}'
        request=urllib.request.Request(url,headers={'User-Agent':'SCAR-X-fact-source-validation'})
        with urllib.request.urlopen(request,timeout=20) as response:raw=response.read(300001)
        if len(raw)>300000:raise ValueError('Public source bound')
        expected=subprocess.check_output(['git','show',PUBLIC_COMMIT+':'+filename],cwd=root)
        if raw!=expected:raise ValueError('Public source and local git object disagree')
        target=directory/'sources'/filename;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
        sources[filename]=raw.decode('utf-8');records.append({'file':filename,'url':url,'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'local_git_object_exact':True,'cached_file':target.relative_to(root).as_posix()})
    facts=extract_facts(sources);output=directory/'facts.json';write_json(output,{'format':'scarx-basic-fact-ledger-v1','source_commit':PUBLIC_COMMIT,'sources':records,'facts':facts,'oracle':'Publicsource AST literals; no code execution or modelanswer agreement','not_training_examples':True})
    result={'run':run.relative_to(root).as_posix(),'decision':'PUBLIC_FACT_LEDGER_VERIFIED_NOT_TRAINING_DATA','data':output.relative_to(root).as_posix(),'sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'source_commit':PUBLIC_COMMIT,'facts':len(facts),'known_facts':sum(f['known_domain'] for f in facts),'outside_domain_facts':sum(not f['known_domain'] for f in facts),'public_sources':records,'main_training':False,'pilot_run':False,'seconds':time.perf_counter()-started}
    write_json(run/'result.json',result);write_json(root/'artifacts/basic-fact-ledger-v1.json',result)
    (run/'report.md').write_text('# Fixed public factledger\n\nAST oracle values, not Codex/teacher traininganswers. Generator must propose candidates, then independent verification.\n\n'+json.dumps(result,indent=2)+'\n',encoding='utf-8');publish(run)
    print(json.dumps({k:result[k] for k in ('run','decision','facts','known_facts','outside_domain_facts','seconds')}))


if __name__=='__main__':main()
