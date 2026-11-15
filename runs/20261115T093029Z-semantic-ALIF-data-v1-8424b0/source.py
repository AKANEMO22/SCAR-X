"""Pinned human-utterance data and deterministic disjoint partitions, no model."""
from datetime import datetime,timezone
import csv,hashlib,io,json,random,re
from pathlib import Path
import sys,uuid
from urllib.request import urlopen,Request

ROOT=Path(__file__).resolve().parents[1]
PIN='57ec275d8078af65b7731c2a98be812d844a6d6b'
LABELS=('card_payment_fee_charged','cash_withdrawal_charge','transfer_fee_charged',
        'card_payment_wrong_exchange_rate','wrong_exchange_rate_for_cash_withdrawal','exchange_rate')


def main():
    run=ROOT/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-semantic-ALIF-data-v1-'+uuid.uuid4().hex[:6]);run.mkdir()
    previous=ROOT/'local/semantic-ALIF-data-v1-execution.json'
    if not previous.exists():raise ValueError('Expected preserved source-split preflight failure')
    prior=Path(json.loads(previous.read_text(encoding='utf-8'))['run'])
    with (ROOT/'local/semantic-ALIF-data-v1-split-compatibility.json').open('x',encoding='utf-8') as f:json.dump({'run':str(run),'prior_source_only_run':prior.name},f)
    local=ROOT/'local'/run.name;local.mkdir();sources={};raws={}
    for name in ('LICENSE','README.md','banking_data/categories.json','banking_data/train.csv','banking_data/test.csv'):
        url='https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/'+PIN+'/'+name
        cached=ROOT/'local'/prior.name/name
        if not cached.is_file():raise ValueError('Original pinned source missing, no silent redownload')
        raw=cached.read_bytes()
        if len(raw)>2_500_000:raise ValueError('Public source size cap')
        sources[name]={'url':url,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'reused_exact_pinned_source_cache':str(cached)}
        path=local/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw);raws[name]=raw
    assert set(LABELS)<=set(json.loads(raws['banking_data/categories.json']))
    assert b'Attribution 4.0 International' in raws['LICENSE']
    chosen=[];seen=set();duplicates=[]
    tables={s:list(csv.DictReader(io.StringIO(raws['banking_data/'+s+'.csv'].decode('utf-8-sig')))) for s in ('train','test')}
    normalize=lambda text:re.sub(r'\W+',' ',text.casefold()).strip()
    reserved_test={normalize(r['text']) for r in tables['test'] if r['category'] in LABELS}
    for index,label in enumerate(LABELS):
        train=[r for r in tables['train'] if r['category']==label];test=[r for r in tables['test'] if r['category']==label]
        random.Random(260216+index).shuffle(train);random.Random(270216+index).shuffle(test)
        train_clean=[];local_seen=set()
        for row in train:
            key=normalize(row['text'])
            if key in reserved_test or key in local_seen:
                duplicates.append({'label':label,'split':'TRAIN_candidate','text_sha256':hashlib.sha256(row['text'].encode()).hexdigest()});continue
            train_clean.append(row);local_seen.add(key)
        for split,count,pool in [('TRAIN',40,train_clean),('DEV',20,train_clean[40:]),('TEST',40,test)]:
            added=0
            for row in pool:
                text=row['text'].strip();key=re.sub(r'\W+',' ',text.casefold()).strip()
                if key in seen and split!='TEST':
                    duplicates.append({'label':label,'split':split,'text_sha256':hashlib.sha256(text.encode()).hexdigest()});continue
                seen.add(key);chosen.append({'id':split.lower()+'-'+str(index)+'-'+str(added),
                    'split':split,'label':index,'intent':label,'text':text});added+=1
                if added==count:break
            if added!=count:raise ValueError('Insufficient unique source data for '+label+'/'+split)
    assert len(chosen)==600
    keysets={s:{normalize(r['text']) for r in chosen if r['split']==s} for s in ('TRAIN','DEV','TEST')}
    assert not(keysets['TRAIN']&keysets['DEV'] or keysets['TRAIN']&keysets['TEST'] or keysets['DEV']&keysets['TEST'])
    raw=json.dumps(chosen,ensure_ascii=False,separators=(',',':')).encode();(run/'selected-data.json').write_bytes(raw)
    (run/'LICENSE').write_bytes(raws['LICENSE'])
    report={'status':'PINNED_HUMAN_INTENT_DATA_PARTITIONED_BEFORE_MODEL_CALLS','source_commit':PIN,'sources':sources,
        'source_license':'CC-BY-4.0','attribution':'Casanueva et al., Efficient Intent Detection with Dual Sentence Encoders,2020; PolyAI',
        'labels':LABELS,'counts':{s:sum(r['split']==s for r in chosen) for s in ('TRAIN','DEV','TEST')},
        'duplicate_candidates_skipped_before_inference':duplicates,'selected_data_sha256':hashlib.sha256(raw).hexdigest(),
        'public_utterances_only':True,'model_or_learning_calls':0,'official_test_not_training':True,
        'full_Banking77_benchmark_claim':False,'raw_source_directory':str(local),
        'official_TEST_rows_preserved_normalized_unique_counts':{s:len(v) for s,v in keysets.items()},
        'official_TEST_duplicates_must_be_clustered_not_independent_samples':True}
    (run/'result.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (run/'source.py').write_bytes(Path(__file__).read_bytes());(run/'protocol.md').write_bytes((ROOT/'experiments/semantic-alif-local-prototypes-v1.md').read_bytes())
    (ROOT/'local/current-semantic-ALIF-data.txt').write_text(str(run),encoding='utf-8')
    print(json.dumps({'run':str(run),'status':report['status'],'counts':report['counts'],'duplicates_skipped':len(duplicates),'model_calls':0}),flush=True)


if __name__=='__main__':main()
