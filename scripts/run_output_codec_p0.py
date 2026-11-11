"""Run approved isolated output-codec P0 engineering gates only."""
from pathlib import Path
import copy,hashlib,json,os,subprocess,sys,time,types
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from scarx.evaluation.output_codec_p0 import AnswerCodec,CodecCoreP0,CodecFixtureSession
from scarx.evaluation.output_codec_p0_validation import continuation_fingerprint
from scarx.evaluation.knowledge_factory import UNKNOWN
from scarx.evaluation.runs import new_run,write_json
from scarx.neural.unified import UnifiedCore
from scarx.neural.core import SpikingCore
from scarx.runtime.limits import apply_memory_limit
from scarx.runtime.resources import ResourceMonitor
from scarx.runtime.work_trace import activity
from public_research_mirror import publish


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def hashfile(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    receipt=read(ROOT/'artifacts/basic-knowledge-verified-v1.json');pool_path=ROOT/receipt['data']
    if hashfile(pool_path)!=receipt['sha256']:raise ValueError('Frozen dataset changed')
    rows=read(pool_path)['items'];train=[r for r in rows if r['split']=='train']
    release=read(ROOT/'artifacts/generation1-release.json');release_path=ROOT/release['checkpoint']
    if hashfile(release_path)!=release['sha256']:raise ValueError('Public release identity changed')
    source_bytes=release_path.read_bytes()
    source=lambda:UnifiedCore.loads(source_bytes)
    inventories={v:{t for r in train for t in AnswerCodec(v).encode(r['canonical_answer'])}|
        {t for text in UNKNOWN.values() for t in AnswerCodec(v).encode(text)} for v in ('legacy-word-v1','numeric-atom-v2')}
    cap=apply_memory_limit()
    run=new_run(ROOT,'output-codec-p0-v1',{'display_title':'Approved output codec P0 · engineering only, no dataset training',
        'protocol':'experiments/output-codec-p0-v1.md','approval_scope':'User approved isolated codec P0 only',
        'neural_core':'Copied public H48 ALIF, same equations/local learning; numeric compatibility grouping',
        'main_training':False,'new_teacher_calls':False,'p1_p2':False,'final_quality_inference':False},
        data_description='Frozen canonical target strings for codec validation, train/policy vocabulary only, bounded synthetic state fixtures; no live/private data.',
        data_files=[receipt['data'],'artifacts/basic-knowledge-verified-v1.json','artifacts/generation1-release.json',release['checkpoint'],
                    'tests/test_output_codec_p0.py','experiments/output-codec-p0-v1.md','experiments/output-codec-p0-numerical-note.md'])
    activity(run,'Kiểm tra codec P0 cô lập','Đo roundtrip, parity và cold restart trên bản sao public H48; không train dataset.',status='started',evidence='experiments/output-codec-p0-v1.md')
    result={'scope':'P0 engineering only; not a learned knowledge or retention result','gates':{},'details':{},'error':None}
    with ResourceMonitor(interval=.02) as monitor:
        try:
            # Keep the failed naive full-width-GEMV method as a diagnostic control.
            a=source();naive=CodecCoreP0.migrate(a,inventories['numeric-atom-v2'])
            naive.core.step=types.MethodType(SpikingCore.step,naive.core)
            context=a.context('p0 hello 0.25');a.word_step(1,context,0);naive.word_step(1,context,0)
            result['details']['naive_width_expansion']={n:float(np.max(np.abs(getattr(a.core,n)-getattr(naive.core,n)))) for n in ('voltage','adaptation','spikes','features')}
            result['details']['naive_width_expansion']['status']='Diagnostic control, not used to relax exact compatibility gate'

            roundtrips={};sizes={}
            for version in inventories:
                codec=AnswerCodec(version);m=CodecCoreP0.migrate(source(),inventories[version],version)
                sizes[version]=len(m.vocabulary)
                cases=[{'id':r['id'],'codec_rendered_reference':codec.render(codec.encode(r['canonical_answer'])),
                        'normalized_roundtrip_matches':codec.roundtrip_matches(r['canonical_answer']),
                        'all_target_tokens_covered':all(t in m.ids for t in codec.encode(r['canonical_answer'])),
                        'kind':'CODEC_REFERENCE_ROUNDTRIP_NOT_NEURAL_RESPONSE'} for r in rows]
                write_json(run/(version+'-roundtrip.json'),cases)
                roundtrips[version]={'rows':len(cases),'roundtrip_matches':sum(c['normalized_roundtrip_matches'] for c in cases),
                    'covered_rows':sum(c['all_target_tokens_covered'] for c in cases)}
            result['details']['codec_roundtrips']=roundtrips;result['details']['vocabulary_sizes']=sizes
            result['gates']['v2_all_target_roundtrips_and_coverage']=roundtrips['numeric-atom-v2']['roundtrip_matches']==320 and roundtrips['numeric-atom-v2']['covered_rows']==320
            result['gates']['fixed_55_inventory_within_256']=set(sizes.values())=={55}

            a=source();b=CodecCoreP0.migrate(a,inventories['numeric-atom-v2'])
            v=len(a.vocabulary);nv=len(b.vocabulary);errors={n:0. for n in ('context','word_probabilities','voltage','adaptation','spikes','features','elig_rec','digit_probabilities')}
            for prompt in ['P0 hello','Tôi hỏi 0.25 và 20.0','P0 Unicode α / digits 2000000000']:
                errors['context']=max(errors['context'],float(np.max(np.abs(a.context(prompt)-b.context(prompt)))))
                for position,previous in enumerate([1,2,3]):
                    pa=a.word_step(previous,a.context(prompt),position,True);pb=b.word_step(previous,b.context(prompt),position,True)
                    errors['word_probabilities']=max(errors['word_probabilities'],float(np.max(np.abs(pa[:v]-pb[:v]))))
                    if pb[v:nv].any():raise AssertionError('New word output not masked')
                    for name in ('voltage','adaptation','spikes','features','elig_rec'):
                        errors[name]=max(errors[name],float(np.max(np.abs(getattr(a.core,name)-getattr(b.core,name)))))
            for pixels in [np.zeros(64,np.float32),np.arange(64,dtype=np.float32)%17]:
                va=a.image(pixels);vb=b.image(pixels)
                errors['digit_probabilities']=max(errors['digit_probabilities'],float(np.max(np.abs(np.asarray(va['probabilities'])-np.asarray(vb['probabilities'])))))
                for name in ('voltage','adaptation','spikes','features'):
                    errors[name]=max(errors[name],float(np.max(np.abs(getattr(a.core,name)-getattr(b.core,name)))))
            result['details']['legacy_parity_errors']=errors;result['gates']['old_context_word_digit_neuron_parity_exact']=all(value==0 for value in errors.values())

            cold_records=[];fixture_updates=0;actual_runner_update_calls=0
            for version in ('legacy-word-v1','numeric-atom-v2'):
                m=CodecCoreP0.migrate(source(),inventories[version],version);codec=AnswerCodec(version)
                m.enable_tokens(codec.encode('0.25.'));s=CodecFixtureSession(m,seed=913)
                s.add_replay_fixture('P0 replay decimal','0.25.');s.add_replay_fixture('P0 replay Unicode','Tôi chưa biết.')
                s.begin_fixture('P0 partial-state fixture','0.25.',mode='full');s.advance_fixture()
                checkpoint=run/(version+'-partial.zip');s.save(checkpoint);saved_hash=hashfile(checkpoint)
                expected=continuation_fingerprint(CodecFixtureSession.loads(checkpoint.read_bytes()))
                output=run/(version+'-cold-output.json')
                command=[sys.executable,str(ROOT/'scripts/output_codec_p0_cold.py'),str(checkpoint),str(output)]
                started=time.perf_counter();child=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,encoding='utf-8',timeout=30)
                (run/(version+'-cold.log')).write_text(child.stdout+child.stderr,encoding='utf-8')
                actual=read(output) if output.exists() else None
                cold_records.append({'codec':version,'command':command,'exit_code':child.returncode,'seconds':time.perf_counter()-started,
                    'full_continuation_json_exact':actual==expected,'saved_checkpoint_preserved':hashfile(checkpoint)==saved_hash,
                    'updates_before_snapshot':1,'parent_continuation_updates':expected['fixture_updates']-1,
                    'cold_continuation_updates':actual['fixture_updates']-1 if actual else None,
                    'expected':expected,'actual':actual})
                fixture_updates+=expected['fixture_updates'];monitor.check()
                actual_runner_update_calls+=1+(expected['fixture_updates']-1)+(actual['fixture_updates']-1 if actual else 0)
            result['details']['cold_process_checks']=cold_records
            result['gates']['cold_state_rng_replay_cursor_exact']=all(r['exit_code']==0 and r['full_continuation_json_exact'] and r['saved_checkpoint_preserved'] for r in cold_records)

            m=CodecCoreP0.migrate(source(),inventories['numeric-atom-v2']);m.enable_tokens(['0.25']);s=CodecFixtureSession(m)
            s.add_replay_fixture('P0 rollback fixture','0.25.');snapshot=s.dumps()
            def reject(session):session.next_replay_fixture();session.model.capacity_rng.integers(10000);return False
            accepted=s.guarded_fixture('P0 rejected state fixture','0.25.',reject,mode='full')
            result['gates']['full_snapshot_rollback_exact']=accepted is False and s.dumps()==snapshot
            result['details']['logical_fixture_synaptic_updates_including_rejected']=fixture_updates+3
            result['details']['actual_runner_fixture_update_calls_including_cold_verification']=actual_runner_update_calls+3
            result['details']['unit_test_fixture_update_calls_included']=False
            result['gates']['source_release_and_dataset_bytes_unchanged']=hashfile(pool_path)==receipt['sha256'] and hashfile(release_path)==release['sha256']

            command=[sys.executable,'-m','pytest','-q','-p','no:cacheprovider','tests/test_output_codec_p0.py','--junitxml='+str(run/'p0-unit-tests.xml')]
            tests=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,encoding='utf-8',timeout=45)
            (run/'p0-unit-tests.log').write_text(tests.stdout+tests.stderr,encoding='utf-8')
            result['details']['unit_test_command']={'command':command,'exit_code':tests.returncode}
            result['gates']['unit_bounds_corruption_foreign_loader_checks']=tests.returncode==0
            monitor.check()
        except Exception as exc:result['error']=type(exc).__name__+': '+str(exc)
    result.update(resources=monitor.report(),cap=cap,main_training=False,dataset_training=False,new_teacher_calls=0,
        codec_used_for_final_quality_inference=False,main_or_live_migrated=False,production_ready=False,
        limits=['Public H48 only; capacity growth and foreign PlasticCore/importance/optimizer migration rejected, not certified.',
                'Fixture controller/replay only; not production Generation-session/SQLite migration.',
                'Numeric compatibility block arithmetic; not a speed/quality improvement or a new learning rule.'])
    result['gates']['observed_process_tree_below_2gb']=not result['resources']['observed_over_limit']
    result['decision']='P0_TECHNICAL_PASS_NOT_TRAINING_OR_RELEASE_APPROVAL' if not result['error'] and all(result['gates'].values()) else 'P0_FAILED_KEEP_SOURCE_AND_LIVE'
    write_json(run/'result.json',result);write_json(ROOT/'artifacts/output-codec-p0-v1.json',{'run':run.relative_to(ROOT).as_posix(),**result})
    (run/'report.md').write_text('# Approved isolated output-codec P0\n\nReference roundtrip strings are not neural answers. Fixture updates are state tests, not dataset training.\n\n'+json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    mirror=publish(run);write_json(run/'publication.json',mirror)
    activity(run,'Hoàn tất kiểm tra codec P0',result['decision'],status='completed' if result['decision'].startswith('P0_TECHNICAL_PASS') else 'failed',evidence='result.json')
    print(json.dumps({'run':run.name,'decision':result['decision'],'gates':result['gates'],'error':result['error']}))
    return int(result['error'] is not None or not all(result['gates'].values()))


if __name__=='__main__':raise SystemExit(main())
