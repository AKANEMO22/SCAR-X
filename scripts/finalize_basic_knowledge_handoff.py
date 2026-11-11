"""Freeze a verified-source/data handoff, not a learned-neural release."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from scarx.evaluation.runs import new_run,write_json
from scarx.runtime.resources import ResourceMonitor
from scarx.runtime.limits import apply_memory_limit
from basic_knowledge_factory import assert_factory_not_started
from public_research_mirror import publish


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def main():
    source=read(ROOT/'artifacts/basic-knowledge-provenance-review-v1.json')
    public=read(ROOT/'artifacts/basic-knowledge-public-provenance-review-v1.json')
    pool=ROOT/source['source_pool']
    assert not source['issues'] and not public['issues'] and hashlib.sha256(pool.read_bytes()).hexdigest()==source['source_pool_sha256']
    compatibility=read(ROOT/'artifacts/basic-knowledge-compatibility-v1.json');acceptance=read(ROOT/'configs/basic_knowledge_acceptance_v1.json')
    assert acceptance['factual_learning']['minimum_exact_normalized_accuracy']==.8
    assert acceptance['generalization']['minimum_heldout_paraphrase_accuracy']==.75
    assert acceptance['unknown_detection']['minimum_unknown_rejection']==.95
    assert acceptance['retention']['maximum_mean_forgetting_pp']==3 and acceptance['retention']['maximum_worst_forgetting_pp']==5
    guards={}
    for protocol in ('basic-knowledge-factory-v1','basic-knowledge-repair-v1'):
        try:assert_factory_not_started(ROOT,protocol);guards[protocol]=False
        except ValueError:guards[protocol]=True
    assert all(guards.values())
    protected=['src/scarx/neural','src/scarx/app','src/scarx/learning','artifacts/generation1-release.json']
    diff=subprocess.check_output(['git','diff','main','--',*protected],cwd=ROOT,text=True)
    if diff:raise ValueError('Protected G1source changed')
    cap=apply_memory_limit()
    files=['configs/basic_knowledge_acceptance_v1.json',source['source_pool'],
        'docs/research/BASIC_KNOWLEDGE_FACTORY_DELIVERY_V1.md','docs/research/BASIC_KNOWLEDGE_CODEC_PROPOSAL_V1.md']
    run=new_run(ROOT,'basic-knowledge-handoff-v1',{'display_title':'Verified dataset factory · reproducible handoff · codec/research approval pending',
        'stage':'DATA_AND_SOURCE_HANDOFF','model_scope':'No neural training, no new teacher/reference calls, no encoder/codec/vocab migration or live access.'},
        data_description='Frozen320publicAGYsource-verified examples and proposal-only codec design.',data_files=files)
    command=[sys.executable,'-m','pytest','-q','-p','no:cacheprovider','tests/test_knowledge_factory.py','tests/test_basic_knowledge_eval.py',
        'tests/test_adaptive_curriculum_data.py','tests/test_public_research_mirror.py','--junitxml='+str(run/'data-tests.xml')]
    started=time.perf_counter()
    with ResourceMonitor(interval=.02) as monitor:
        result=subprocess.run(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',timeout=45)
        monitor.check()
    (run/'tests.log').write_text(result.stdout,encoding='utf-8')
    checks={'command':command,'exit_code':result.returncode,'seconds':time.perf_counter()-started,'resources':monitor.report(),'cap':cap}
    write_json(run/'test-command.json',checks)
    if result.returncode:raise RuntimeError('Data handoff tests failed; preserve log')
    result={'decision':'NEEDS_RESEARCH_APPROVAL','product_status':'NOT_PRODUCTION_READY','data_status':'SOURCE_AND_ORIGIN_VERIFIED_CODEC_BLOCKED',
        'verified_rows':320,'source_pool':source['source_pool'],'source_pool_sha256':source['source_pool_sha256'],
        'source_origin_review':source['run'],'public_recheck':public['run'],'codec_compatibility':compatibility['decision'],
        'legacy_current_vocabulary_tokens':24,'complete_training_and_declared_unknown_vocabulary_union':55,
        'supported_vocabulary_cap':256,'current_surface_roundtrip_failure_rows':len(compatibility['audit']['surface_roundtrip_mismatches']),
        'registered_generation_budget_guards_blocked':guards,'new_generation_calls':0,'neural_training':False,'final_quality_scores_opened':False,
        'codec_or_encoder_implemented':False,'live_state_accessed':False,'protected_source_diff_empty':not diff,
        'acceptance_thresholds_preserved':True,'tests':checks,
        'required_next_approval':'Research-only output codec/migration P0 proposal; separate learning/P1/P2 and unknown curriculum approvals, no production migration.',
        'factory_document':'docs/research/BASIC_KNOWLEDGE_FACTORY_DELIVERY_V1.md','codec_proposal':'docs/research/BASIC_KNOWLEDGE_CODEC_PROPOSAL_V1.md'}
    write_json(run/'result.json',result);write_json(ROOT/'artifacts/basic-knowledge-handoff-v1.json',{'run':run.relative_to(ROOT).as_posix(),**result})
    (run/'report.md').write_text('# Verified basicdata sourcehandoff\n\n'+json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    mirror=publish(run)
    print(json.dumps({'run':run.name,'decision':result['decision'],'data_status':result['data_status'],'tests_exit':checks['exit_code'],'mirror':mirror.get('mirror'),'mirror_skipped':mirror.get('skipped')}))


if __name__=='__main__':main()
