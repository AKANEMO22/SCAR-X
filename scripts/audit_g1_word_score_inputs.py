"""Immutable result audit and fresh-process continuation of all27models."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import sys,json,hashlib,subprocess,time
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
import numpy as np
from train_g1_word_score_inputs import source_data
from scarx.evaluation.fixed_word_codes import fit
from scarx.evaluation.score_word import ScoreWord
from scarx.evaluation.runs import new_run,write_json
from scarx.runtime.experiment_status import progress
from scarx.runtime.work_trace import activity
from scarx.runtime.resources import ResourceMonitor
from scarx.runtime.limits import apply_memory_limit


def continuation(payload,prompt,target):
    model=ScoreWord.loads(payload);learn=model.learn(prompt,target);reply=model.generate(prompt)
    return {'learn':learn,'reply':reply,'final_sha256':hashlib.sha256(model.dumps()).hexdigest()}


def main():
    began=time.perf_counter();cap=apply_memory_limit();info=json.loads((root/'artifacts/g1-word-score-inputs-latest.json').read_text(encoding='utf-8'))
    source=(root/info['run']).resolve()
    if source.parent!=root/'runs' or not (source/'result.json').is_file():raise ValueError('Expected completed source run')
    result=json.loads((source/'result.json').read_text(encoding='utf-8'));records=json.loads((source/'trials.json').read_text(encoding='utf-8'))
    assert len(records)==27 and len({(r['seed'],r['condition'],r['input_encoding']) for r in records})==27
    manifest=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    files=['src/scarx/neural/core.py','src/scarx/neural/unified.py','src/scarx/neural/paired_score.py','src/scarx/neural/masked_score.py',
        'src/scarx/neural/language.py','src/scarx/evaluation/score_word.py','src/scarx/evaluation/fixed_word_codes.py',
        'scripts/train_g1_word_score_transfer.py','scripts/train_g1_word_score_inputs.py','experiments/g1-word-score-input-transfer-v2.md']
    train,_,_,vocab,datafiles=source_data();files+=datafiles
    assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==manifest['source_sha256'][name] for name in files)
    boundary=json.loads((root/'artifacts/neural-core-boundary.json').read_text(encoding='utf-8'))['main_core_sha256']
    assert all(hashlib.sha256((root/n).read_bytes()).hexdigest()==v for n,v in boundary.items())
    assert max(r['seconds'] for r in records)<240 and result['resources']['elapsed_seconds']<2400
    assert not result['resources']['observed_over_limit']
    run=new_run(root,'g1-word-score-input-audit-v1',{'protocol':'experiments/g1-word-score-input-audit-v1.md',
        'display_title':'Audit27checkpoint · coldresume và lõi ALIF bất biến',
        'model_scope':'Technical discarded-copy checks only; no new training run or serving weight changes',
        'source_run':info['run'],'checkpoints':27},data_description='Public experimentalcheckpoint continuation only; no private model/data.')
    (run/'report.md').write_text('# Fixed-input technical audit\n\nChecks in progress; source quality decision unchanged.\n',encoding='utf-8')
    child='''import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from scarx.evaluation.score_word import ScoreWord
m=ScoreWord.loads(Path(sys.argv[2]).read_bytes());learn=m.learn(sys.argv[3],sys.argv[4]);reply=m.generate(sys.argv[3])
print(json.dumps({'learn':learn,'reply':reply,'final_sha256':hashlib.sha256(m.dumps()).hexdigest()}))
'''
    checks=[];sizes=set()
    with ResourceMonitor(interval=.1) as monitor:
        for row in records:
            path=(root/row['checkpoint']).resolve()
            if not path.is_relative_to(root/'artifacts/word-score-input-transfer'/source.name):raise ValueError('Checkpoint outside source artifacts')
            payload=path.read_bytes();digest=hashlib.sha256(payload).hexdigest();model=ScoreWord.loads(payload);c=model.core.config
            assert (c.seed,c.hidden,c.fan_in,c.plastic_rows,model.condition,model.responses_seen)==(row['seed'],48,8,12,row['condition'],1024)
            encoding=None if row['input_encoding']=='legacy' else fit([r['prompt'] for r in train],row['input_encoding'])
            assert model.encoding==(encoding or {'kind':'legacy'}) and model.vocabulary==vocab
            initial=ScoreWord(vocab,row['seed'],row['condition'],encoding);changed=any(not np.array_equal(getattr(model.core,k),getattr(initial.core,k)) for k in ('w_in','w_rec'))
            assert changed==(row['condition']!='frozen') and all(np.isfinite(a).all() for a in model.core.arrays().values())
            sizes.add(model.core.memory_bytes());prompt=train[0]['prompt'];target=train[0]['target']
            expected=continuation(payload,prompt,target)
            completed=subprocess.run([sys.executable,'-c',child,str(root/'src'),str(path),prompt,target],cwd=root,capture_output=True,check=True,timeout=30)
            assert json.loads(completed.stdout)==expected and digest==hashlib.sha256(path.read_bytes()).hexdigest()
            checks.append({'seed':row['seed'],'input_encoding':row['input_encoding'],'condition':row['condition'],'sha256':digest,'cold_exact':True,'hidden_changed':changed})
            progress(run,'Cold continuation '+str(len(checks))+'/27',len(checks),27,metrics={'all_exact_so_far':True});monitor.check()
            if time.perf_counter()-began>120:raise TimeoutError('Registered120s checkpointaudit limit')
        assert len(sizes)==1
        proof={'run':run.relative_to(root).as_posix(),'source_run':info['run'],'decision':'TECHNICAL_CHECKS_PASS_QUALITY_DECISION_UNCHANGED',
            'source_quality_decision':result['decision'],'all_cold_exact':True,'all_state_unchanged':True,'serving_core_unchanged':True,
            'train_only_encoding_metadata':True,'all_same_neural_array_bytes':list(sizes)[0],'source_code_protocol_data_hashes_matched':True,
            'checkpoints':checks,'new_training':False,'deployed':False,'normal_chat_or_vision_proven':False,'resources':monitor.report(),'cap':cap}
        (run/'report.md').write_text('# Fixed-input technical audit\n\nAll27coldcontinuations match; original failed quality decision remains failed. No source/live weights changed. One continuation step occurs only on discarded copies.\n\n'+json.dumps(proof,indent=2)+'\n',encoding='utf-8')
        write_json(run/'result.json',proof);write_json(root/'artifacts/g1-word-score-inputs-audit.json',proof)
        activity(run,'Đã audit27checkpoint ở process mới','Fullstate/metadata/RNG continuation khớp; frozenhidden không đổi; core phục vụ giữSHA cũ. Technicalpass không thay quyết định quality.',evidence={'all_cold_exact':True,'source_quality_decision':result['decision'],'array_bytes':list(sizes)[0]})
        print(json.dumps({k:proof[k] for k in ('run','decision','source_quality_decision','all_cold_exact','all_same_neural_array_bytes')}))


if __name__=='__main__':main()
