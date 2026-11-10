"""Twelve actual cold resumes, input-block locality and source integrity."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import sys,json,hashlib,subprocess,time
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
import numpy as np
from train_g1_word_score_inputs import source_data
from generate_g1_state_paraphrases import INTENTS
from scarx.evaluation.observed_association import ObservedAssociation
from scarx.evaluation.runs import new_run,write_json
from scarx.runtime.experiment_status import progress
from scarx.runtime.work_trace import activity
from scarx.runtime.resources import ResourceMonitor
from scarx.runtime.limits import apply_memory_limit


def continuation(model,row,observed):
    loss=model.learn(row['prompt'],row['target'],observed);reply=model.generate(row['prompt']);rng=int(model.sampler_rng.integers(2**31))
    return {'learn':loss,'reply':reply,'next_sampler_value':rng,'sha256':hashlib.sha256(model.dumps()).hexdigest()}


def main():
    cap=apply_memory_limit();began=time.perf_counter();source=json.loads((root/'artifacts/g1-observed-association-latest.json').read_text(encoding='utf-8'))
    origin=(root/source['run']).resolve()
    if origin.parent!=root/'runs':raise ValueError('Expected registeredsource')
    records=json.loads((origin/'trials.json').read_text(encoding='utf-8'));manifest=json.loads((origin/'manifest.json').read_text(encoding='utf-8'))
    assert len(records)==12 and len({(r['seed'],r['condition']) for r in records})==12
    train,_,_,vocab,datafiles=source_data()
    for name in datafiles+['src/scarx/evaluation/observed_association.py','scripts/train_g1_observed_association.py','experiments/g1-observed-association-v1.md','src/scarx/neural/core.py','src/scarx/neural/unified.py']:
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==manifest['source_sha256'][name]
    boundary=json.loads((root/'artifacts/neural-core-boundary.json').read_text(encoding='utf-8'))
    assert all(hashlib.sha256((root/n).read_bytes()).hexdigest()==v for n,v in boundary['main_core_sha256'].items())
    run=new_run(root,'g1-observed-association-audit-v1',{'protocol':'experiments/g1-observed-association-audit-v1.md',
        'display_title':'Audit12checkpoint · covarianceassociation / gold-freeprediction',
        'model_scope':'Technical discarded-copy checks only. Qualitydecision remains sourceSTOP; no mainchanges.',
        'source_run':source['run'],'checkpoints':12},data_description='Publicexperimentalcheckpoints, oneknownexample only on discardcopies; no private data.')
    (run/'report.md').write_text('# Observed-association technical audit\n\nColdresume and locality in progress; not a quality result.\n',encoding='utf-8')
    child='''import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from scarx.evaluation.observed_association import ObservedAssociation
m=ObservedAssociation.loads(Path(sys.argv[2]).read_bytes());loss=m.learn(sys.argv[3],sys.argv[4],sys.argv[5]);reply=m.generate(sys.argv[3]);rng=int(m.sampler_rng.integers(2**31))
print(json.dumps({'learn':loss,'reply':reply,'next_sampler_value':rng,'sha256':hashlib.sha256(m.dumps()).hexdigest()}))
'''
    checks=[];models={};arraybytes=set()
    with ResourceMonitor(interval=.1) as monitor:
        for row in records:
            path=(root/row['checkpoint']).resolve()
            if not path.is_relative_to(root/'artifacts/observed-association'/origin.name):raise ValueError('Sourcecheckpointpath')
            payload=path.read_bytes();digest=hashlib.sha256(payload).hexdigest();assert digest==row['sha256']
            model=ObservedAssociation.loads(payload);assert model.core.config.seed==row['seed'] and model.condition==row['condition'] and model.experiences==model.observations==1024 and model.slot_clock==7168
            assert model.vocabulary==vocab and model.memory_bytes()==row['array_bytes'];arraybytes.add(model.memory_bytes())
            initial=ObservedAssociation(vocab,row['seed'],row['condition'],model.encoding)
            noncontext=np.r_[np.arange(model.context_start),np.arange(model.context_end,model.core.config.inputs)]
            np.testing.assert_array_equal(initial.core.w_in[:,noncontext],model.core.w_in[:,noncontext]);np.testing.assert_array_equal(initial.core.w_rec,model.core.w_rec)
            for n in ('stable_in','stable_rec','stable_out','stable_bias'):np.testing.assert_array_equal(getattr(initial.core,n),getattr(model.core,n))
            if row['condition'] in ('frozen','erase_trace'):np.testing.assert_array_equal(initial.core.w_in,model.core.w_in)
            assert all(np.isfinite(a).all() for a in model.core.arrays().values())
            models[(row['seed'],row['condition'])]=model
            first=train[0];observed=next(r['target'] for r in train if r['intent']==INTENTS[1]) if row['condition']=='cross_family' else first['target']
            expected=continuation(ObservedAssociation.loads(payload),first,observed)
            completed=subprocess.run([sys.executable,'-c',child,str(root/'src'),str(path),first['prompt'],first['target'],observed],cwd=root,capture_output=True,check=True,timeout=20,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            assert json.loads(completed.stdout)==expected and hashlib.sha256(path.read_bytes()).hexdigest()==digest
            checks.append({'seed':row['seed'],'condition':row['condition'],'source_sha256':digest,'cold_exact':True,'input_block_locality':True})
            progress(run,'Coldresume '+str(len(checks))+'/12',len(checks),12,metrics={'all_exact_so_far':True});monitor.check()
            if time.perf_counter()-began>120:raise TimeoutError('Registered120s associationauditcap')
        for seed in (31013,31019,31033):
            frozen=models[(seed,'frozen')];erased=models[(seed,'erase_trace')]
            for name,value in frozen.core.arrays().items():np.testing.assert_array_equal(value,erased.core.arrays()[name])
            assert len({models[(seed,c)].ticks for c in ('paired','frozen','cross_family','erase_trace')})==1
            assert len({json.dumps(models[(seed,c)].sampler_rng.bit_generator.state) for c in ('paired','frozen','cross_family','erase_trace')})==1
        assert len(arraybytes)==1
        proof={'run':run.relative_to(root).as_posix(),'source_run':source['run'],'decision':'ASSOCIATION_TECHNICAL_PASS_QUALITY_UNCHANGED',
            'source_quality_decision':source['decision'],'all_cold_exact':True,'all_input_block_locality':True,'all_frozen_erased_exact':True,
            'same_tick_and_sampler_budget_by_seed':True,'all_source_SHA_unchanged':True,'serving_core_unchanged':True,
            'array_bytes_each':list(arraybytes)[0],'checks':checks,'new_training':False,'deployed':False,'normal_chat_or_vision_proven':False,
            'note':'Onecontinuationexample on discardedcopies only; not formaltraining/quality/retention evidence.','resources':monitor.report(),'cap':cap}
        (run/'report.md').write_text('# Observed-association technical audit\n\n12coldprocessresumes exact; only contextsynapses plastic, frozen-erased arrays/ticks/sampler match. Qualitydecision remainsSTOP. No mainweights or sourcecheckpoints changed.\n\n'+json.dumps(proof,indent=2)+'\n',encoding='utf-8')
        write_json(run/'result.json',proof);write_json(root/'artifacts/g1-observed-association-audit.json',proof)
        activity(run,'Đã audit12checkpoint association','Coldresumes và locality/tick/sampler đều khớp; kỹthuậtpass không thay chấtlượngSTOP. Maincore không đổi.',evidence={'all_cold_exact':True,'source_quality_decision':source['decision'],'array_bytes_each':list(arraybytes)[0]})
        print(json.dumps({k:proof[k] for k in ('run','source_quality_decision','all_cold_exact','array_bytes_each')}))


if __name__=='__main__':main()
