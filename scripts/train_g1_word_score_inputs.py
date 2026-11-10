"""Final preregistered fixed-input transfer; one spiking core per actual trial."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import sys,json,time,itertools,hashlib
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
import numpy as np
from train_g1_word_score_transfer import trial,CONDITIONS
from generate_g1_state_paraphrases import INTENTS
from scarx.evaluation.score_word import ScoreWord
from scarx.evaluation.fixed_word_codes import fit,coverage
from scarx.neural.language import tokenize
from scarx.evaluation.runs import new_run,write_json
from scarx.evaluation.phase21 import interval
from scarx.runtime.experiment_status import progress
from scarx.runtime.work_trace import activity
from scarx.runtime.resources import ResourceMonitor
from scarx.runtime.limits import apply_memory_limit

SEEDS=(28001,28019,28027)
ENCODERS=('legacy','known-sparse128-v1','known-digest128-v1')
METRICS=('strict_style_exact','normalized_style_exact','teacher_token_accuracy','teacher_nll','first_token_accuracy')


def reviewed(name):
    info=json.loads((root/'artifacts'/name).read_text(encoding='utf-8'));path=(root/info['data']).resolve()
    if not path.is_relative_to(root/'data/candidates') or hashlib.sha256(path.read_bytes()).hexdigest()!=info['sha256']:raise ValueError('Reviewed public source mismatch')
    return info,json.loads(path.read_text(encoding='utf-8'))['items']


def source_data():
    info,data=reviewed('g1-curriculum-reviewed.json');prior_info,prior=reviewed('g1-state-prompts-reviewed.json')
    corpus=json.loads((root/'data/dialogue/generation1.json').read_text(encoding='utf-8'));train=[]
    for intent,family in zip(INTENTS,corpus['families']):train += [{'id':f'known-{intent}-{j}','intent':intent,'prompt':p,'target':family['answer']} for j,p in enumerate(family['train'])]
    train += [r for r in data if r['split']=='train'];dev=[r for r in data if r['split']=='development']
    vocab=['<eos>','<bos>']+sorted({t for r in train for t in tokenize(r['target'])})
    assert (len(train),len(dev),len(prior))==(181,31,48) and all(all(t in vocab for t in tokenize(r['target'])) for r in dev+prior)
    return train,dev,prior,vocab,[info['data'],'data/dialogue/generation1.json',prior_info['data']]


def main():
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    cap=apply_memory_limit();began=time.perf_counter();train,dev,prior,vocab,sources=source_data()
    encodings={name:None if name=='legacy' else fit([r['prompt'] for r in train],name) for name in ENCODERS}
    # Shape/weights equality across adapters before any calibration/labels.
    baseline=ScoreWord(vocab,SEEDS[0],'full')
    for name,encoding in encodings.items():
        model=ScoreWord(vocab,SEEDS[0],'full',encoding)
        for key,value in baseline.core.arrays().items():np.testing.assert_array_equal(value,model.core.arrays()[key])
        assert model.core.memory_bytes()==baseline.core.memory_bytes()
    run=new_run(root,'g1-word-score-input-transfer-v2',{'protocol':'experiments/g1-word-score-input-transfer-v2.md',
        'display_title':'ALIF học từ thật · mã đầu vào128chiều · vòng cuối',
        'model_scope':'Fresh48ALIF experimental; no Transformer/LLMproxy/mainmigration. Same localrule, no scale.',
        'model_seeds':SEEDS,'conditions':CONDITIONS,'input_encodings':ENCODERS,'responses':1024,'workers':2,
        'hidden':48,'context_channels':128,'trainable_array_bytes':baseline.core.memory_bytes(),'final_application_round':True},
        data_description='Same public181train/31dev/48prior reviewed AGY/authored prompts. Train-only numeric codebook, no privateweights or target input.',data_files=sources)
    artifact=root/'artifacts/word-score-input-transfer'/run.name;artifact.mkdir(parents=True)
    for name in ENCODERS:(artifact/name).mkdir()
    write_json(run/'input-metadata.json',encodings)
    cover={name:{split:coverage(encoding,[r['prompt'] for r in rows]) for split,rows in [('train',train),('development',dev),('prior',prior)]} for name,encoding in encodings.items() if encoding}
    write_json(run/'input-coverage.json',cover)
    (run/'report.md').write_text('# Fixed-input neural word transfer v2\n\nActual training in progress, same local-score ALIF core. No serving migration. Protocol fixed before training; outcome pending.\n',encoding='utf-8')
    activity(run,'Bắt đầu vòng thử neural đã đăng ký','27actualtrials, ba freshmodelseeds, ba mã đầu vào cố định và frozen/full/shuffled. Giữ cùng H48, synapse và1024responses; không đổi coreHermes.',status='started',evidence={'trials':27,'codebook_word_types':len(encodings[ENCODERS[-1]]['idf']),'array_bytes':baseline.core.memory_bytes()})
    records=[];pending={};jobs=iter([(s,c,str(run),str(artifact/name),train,dev,prior,vocab,encodings[name],name) for s,name,c in itertools.product(SEEDS,ENCODERS,CONDITIONS)])
    with ResourceMonitor(interval=.1) as monitor:
        with ProcessPoolExecutor(max_workers=2) as pool:
            for _ in range(2):job=next(jobs);pending[pool.submit(trial,job)]=job
            while pending:
                ready,_=wait(pending,timeout=.5,return_when=FIRST_COMPLETED);monitor.check()
                if time.perf_counter()-began>2400:raise TimeoutError('Registered2400s inputtransfer limit')
                for future in ready:
                    pending.pop(future);row,neural=future.result();records.append(row)
                    progress(run,f'{row["input_encoding"]} · seed{row["seed"]} · {row["condition"]}',len(records),27,
                        metrics={'dev_normalized':row['summary']['development']['normalized_style_exact'],'prior_normalized':row['summary']['prior']['normalized_style_exact'],
                            'hidden_delta_in':row['hidden_delta_in'],'train_seconds':row['train_seconds']},neural=neural)
                    write_json(run/'trials.json',records);print(json.dumps({k:row[k] for k in ('seed','input_encoding','condition','summary','train_seconds')}),flush=True)
                    try:job=next(jobs);pending[pool.submit(trial,job)]=job
                    except StopIteration:pass
        summary={name:{c:{split:{m:float(np.mean([r['summary'][split][m] for r in records if r['input_encoding']==name and r['condition']==c])) for m in METRICS} for split in ('train16','development','prior')} for c in CONDITIONS} for name in ENCODERS}
        def score(seed,encoding,condition,split):return next(r['summary'][split]['normalized_style_exact'] for r in records if (r['seed'],r['input_encoding'],r['condition'])==(seed,encoding,condition))
        dense=ENCODERS[-1]
        encoding_gain=[score(s,dense,'full','development')-score(s,'legacy','full','development') for s in SEEDS]
        hidden_gain=[score(s,dense,'full','development')-score(s,dense,'frozen','development') for s in SEEDS]
        prior_gain=[score(s,dense,'full','prior')-score(s,dense,'frozen','prior') for s in SEEDS]
        gates={'absolute_dev40':summary[dense]['full']['development']['normalized_style_exact']>=.4,
            'encoding_dev_gain10_all_positive':bool(np.mean(encoding_gain)>=.1 and all(v>0 for v in encoding_gain)),
            'hidden_dev_gain10_all_positive':bool(np.mean(hidden_gain)>=.1 and all(v>0 for v in hidden_gain)),
            'hidden_prior_gain10':bool(np.mean(prior_gain)>=.1),'all_checkpoint_continuations_exact':all(r['checkpoint_exact'] and r['continuation_exact'] for r in records)}
        passed=all(gates.values());result={'run':run.relative_to(root).as_posix(),
            'decision':'WORD_SCORE_INPUT_FRESH_VALIDATION_REQUIRED' if passed else 'STOP_WORD_SCORE_INPUT_TRANSFER_V2',
            'summary':summary,'gates':gates,'paired_encoding_dev_gain':interval(encoding_gain),'paired_hidden_dev_gain':interval(hidden_gain),
            'paired_hidden_prior_gain':interval(prior_gain),'trials':len(records),'coverage':cover,'resources':monitor.report(),'cap':cap,
            'deployed':False,'normal_chat_or_vision_proven':False,'retention_confirmed':False,'new_hidden_rule_tuning':False,
            'application_route_closed':not passed,'limits':['Exploratory3freshseeds, known8styles and previously inspected dev/prior, not freshfinaltest.',
                'No Transformer/pretrained/LLMproxy/maincorechanges. Fixed numeric frontend is not learnedsemantics.',
                'Finalword-score applicationround; no extraencoding/LR/epochs/modelsize search or retrospectivegatechange.']}
        (run/'report.md').write_text('# Fixed-input neural word transfer v2\n\nActual local training; all structural ablations and seeds retained. No serving migration or general capability claim.\n\n'+json.dumps(result,indent=2)+'\n',encoding='utf-8')
        write_json(run/'result.json',result);write_json(root/'artifacts/g1-word-score-inputs-latest.json',result)
        activity(run,'Hoàn tất vòng thử neural cuối',result['decision']+'; giữ nguyên mọi ngưỡng, không nhập candidate vào modellive.',evidence={'gates':gates,'trials':len(records)})
        print(json.dumps({'run':result['run'],'decision':result['decision'],'gates':gates,'paired_hidden_dev_gain':result['paired_hidden_dev_gain']},ensure_ascii=False))


if __name__=='__main__':main()
