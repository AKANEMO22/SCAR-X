"""Actual coldworkingstate continuation on all9readoutonly visualmodels."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import sys,json,time,hashlib,subprocess
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
import numpy as np
from scarx.evaluation.visual_words import VisualWords
from scarx.evaluation.runs import new_run,write_json
from scarx.runtime.experiment_status import progress
from scarx.runtime.work_trace import activity
from scarx.runtime.resources import ResourceMonitor
from scarx.runtime.limits import apply_memory_limit


def continuation(model,case,image):
    model.prepare(case['prompt'],image);model.word_feature(1,0);model=VisualWords.loads(model.dumps())
    f=model.word_feature(2,1).tolist();reply=model.generate(case['prompt'],image);lesson=model.learn(case['prompt'],image,case['target']);rng=int(model.sampler_rng.integers(2**31))
    return {'next_word_feature':f,'reply':reply,'lesson':lesson,'sampler_next':rng,'sha256':hashlib.sha256(model.dumps()).hexdigest()}


def main():
    cap=apply_memory_limit();began=time.perf_counter();source=json.loads((root/'artifacts/visual-word-readout-latest.json').read_text(encoding='utf-8'));origin=root/source['run']
    rows=json.loads((origin/'trials.json').read_text(encoding='utf-8'));manifest=json.loads((origin/'manifest.json').read_text(encoding='utf-8'));assert len(rows)==9
    dataset=root/source['dataset'];assert hashlib.sha256(dataset.read_bytes()).hexdigest()==source['dataset_sha256'];data=json.loads(dataset.read_text(encoding='utf-8'));case=next(r for r in data['cases'] if r['split']=='train');scene=next(s for s in data['scenes'] if s['id']==case['scene_id']);image_path=dataset.parent/scene['file'];image=image_path.read_bytes()
    for name in ['src/scarx/evaluation/visual_words.py','src/scarx/evaluation/readout_conditioning.py','src/scarx/neural/core.py','src/scarx/neural/input_events.py','scripts/train_visual_word_readout.py','experiments/visual-word-readout-v1.md',source['dataset']]+[(dataset.parent/s['file']).relative_to(root).as_posix() for s in data['scenes']]:
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==manifest['source_sha256'][name]
    boundary=json.loads((root/'artifacts/neural-core-boundary.json').read_text(encoding='utf-8'));assert all(hashlib.sha256((root/n).read_bytes()).hexdigest()==sha for n,sha in boundary['main_core_sha256'].items())
    run=new_run(root,'visual-word-audit-v1',{'protocol':'experiments/visual-word-audit-v1.md',
        'display_title':'Audit9visualwordcheckpoints · heldstate/resume, qualitySTOP',
        'model_scope':'Technicaldiscardedcopy checks only, no retraining/main/proposal30change.','source_run':source['run'],'checkpoints':9},
        data_description='Ownedgeometricpubliccheckpointworkingstate, onepredeterminedexample on discardedcopies only.')
    (run/'report.md').write_text('# Visualword technical audit\n\nColdworkingstate checks in progress; qualitydecision unchanged.\n',encoding='utf-8')
    child='''import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from scarx.evaluation.visual_words import VisualWords
m=VisualWords.loads(Path(sys.argv[2]).read_bytes());image=Path(sys.argv[3]).read_bytes();q=sys.argv[4];t=sys.argv[5]
m.prepare(q,image);m.word_feature(1,0);m=VisualWords.loads(m.dumps());f=m.word_feature(2,1).tolist();reply=m.generate(q,image);lesson=m.learn(q,image,t);rng=int(m.sampler_rng.integers(2**31))
print(json.dumps({'next_word_feature':f,'reply':reply,'lesson':lesson,'sampler_next':rng,'sha256':hashlib.sha256(m.dumps()).hexdigest()}))
'''
    checks=[];models={};sizes=set()
    with ResourceMonitor(interval=.1) as monitor:
        for row in rows:
            path=(root/row['checkpoint']).resolve()
            if not path.is_relative_to(root/'artifacts/visual-word-readout'/origin.name):raise ValueError('Checkpointpath')
            raw=path.read_bytes();assert hashlib.sha256(raw).hexdigest()==row['sha256'];m=VisualWords.loads(raw);models[(row['seed'],row['condition'])]=m
            assert (m.core.config.seed,m.condition,m.experiences,m.head.count)==(row['seed'],row['condition'],1536,128) and m.core.updates==m.head.updates==row['word_updates']
            initial=VisualWords(m.vocabulary,row['seed'],row['condition']);np.testing.assert_array_equal(initial.core.w_in,m.core.w_in);np.testing.assert_array_equal(initial.core.w_rec,m.core.w_rec)
            assert m.head.weights is m.core.w_out and m.head.bias is m.core.bias;sizes.add(m.memory_bytes())
            if row['recovered_without_retraining']:
                old=root/row['source_recovery']['checkpoint'];assert raw==old.read_bytes() and hashlib.sha256(old.read_bytes()).hexdigest()==row['source_recovery']['sha256']
            expected=continuation(VisualWords.loads(raw),case,image)
            completed=subprocess.run([sys.executable,'-c',child,str(root/'src'),str(path),str(image_path),case['prompt'],case['target']],cwd=root,capture_output=True,check=True,timeout=20,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            assert json.loads(completed.stdout)==expected and hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
            checks.append({'seed':row['seed'],'condition':row['condition'],'cold_exact':True,'source_sha256':row['sha256'],'hidden_exactinitial':True})
            progress(run,'Coldworkingstate '+str(len(checks))+'/9',len(checks),9,metrics={'all_exact_so_far':True});monitor.check()
            if time.perf_counter()-began>120:raise TimeoutError('Registered120s auditcap')
        for seed in (33113,33119,33133):
            group=[r for r in rows if r['seed']==seed]
            assert len({r['word_updates'] for r in group})==len({r['training_ticks'] for r in group})==1
            assert len({json.dumps(models[(seed,c)].sampler_rng.bit_generator.state) for c in ('full','no_image','no_question')})==1
        assert len(sizes)==1
        proof={'run':run.relative_to(root).as_posix(),'source_run':source['run'],'decision':'VISUAL_WORD_TECHNICAL_PASS_QUALITY_UNCHANGED','source_quality_decision':source['decision'],
            'all_cold_exact':True,'all_hidden_exactinitial':True,'all_source_SHA_unchanged':True,'two_recovered_models_byteidentical_without_retraining':True,
            'same_sampler_wordupdate_trainingtick_budget_by_seed':True,'array_bytes_each':list(sizes)[0],'checks':checks,'new_training':False,'deployed':False,
            'normal_photo_qa_proven':False,'hidden_representation_learning':False,'proposal30_approved':False,'resources':monitor.report(),'cap':cap}
        (run/'report.md').write_text('# Visualword technical audit\n\nAll9coldworkingstate/learning/RNGcontinuations exact, hiddenfixed and first2modelsreusedunchanged. SourcequalityremainsSTOP; technicalpass is not VQAor mainlocallearning success.\n\n'+json.dumps(proof,indent=2)+'\n',encoding='utf-8')
        write_json(run/'result.json',proof);write_json(root/'artifacts/visual-word-readout-audit.json',proof)
        activity(run,'Đã audit9model visualword','Coldworkingstate+learning+RNGexact, hiddenfixed,2recoveredmodels không học lại. QualitySTOPgiữ nguyên, main/proposal30không đổi.',evidence={'all_cold_exact':True,'source_quality_decision':source['decision'],'array_bytes_each':list(sizes)[0]})
        print(json.dumps({k:proof[k] for k in ('run','source_quality_decision','all_cold_exact','array_bytes_each')}))


if __name__=='__main__':main()
