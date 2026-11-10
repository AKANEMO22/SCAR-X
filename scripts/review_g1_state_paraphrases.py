"""Explicit semantic review of the generated intent prompts; no label import."""
from pathlib import Path
import hashlib,json,sys
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
from scarx.neural.language import tokenize
from scarx.evaluation.runs import new_run,write_json

INTENTS=('greeting_vi','greeting_en','identity','gratitude','farewell','activity','capabilities','teaching')


def main():
    source=json.loads((root/'artifacts/antigravity-g1-state-latest.json').read_text(encoding='utf-8'))
    directory=(root/source['candidate_directory']).resolve()
    if not directory.is_relative_to(root/'data/candidates'):raise ValueError('Dataset scope')
    raw=directory/'prompts.json';assert hashlib.sha256(raw.read_bytes()).hexdigest()==source['sha256']
    corpus=json.loads((root/'data/dialogue/generation1.json').read_text(encoding='utf-8'))
    known={' '.join(tokenize(p)) for f in corpus['families'] for split in ('train','development','acceptance') for p in f[split]}
    data=json.loads(raw.read_text(encoding='utf-8'));records=[]
    # All48 prompts were read in full and checked against intended meaning.
    # Several duplicate known prompts; preserve/label rather than claim fresh.
    for row in data['items']:
        assert row['intent'] in INTENTS
        family=corpus['families'][INTENTS.index(row['intent'])]
        records.append({**row,'target':family['answer'],'source':'synthetic:actual Antigravity CLI; explicit Codex semantic review',
            'same_as_old_fixture': ' '.join(tokenize(row['prompt'])) in known,
            'reference_kind':'one project style response, not a factual/semantic oracle',
            'automatic_learning_allowed':False,'purpose':'diagnostic input only; all items may be inspected, not a new finaltest'})
    run=new_run(root,'g1-state-paraphrase-review',{'source_run':source['run'],'items':48,'new_training':False,
        'display_title':'Rà soát 48 prompts Antigravity · chưa học'},data_description='Explicit semantic intent review; no real user transcript')
    target=directory/'reviewed.json';write_json(target,{'items':records,'review':'All48 meaning matches intent; duplicate prompts explicitly marked. English questions may have the old Vietnamese style target; exact match is only a limited diagnostic.'})
    result={'run':run.relative_to(root).as_posix(),'source_run':source['run'],'data':target.relative_to(root).as_posix(),
        'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'items':48,'reviewed':True,'same_as_old_fixture':sum(r['same_as_old_fixture'] for r in records),
        'new_training':False,'automatic_learning_allowed':False,'decision':'REVIEWED_DIAGNOSTIC_PROMPTS_ONLY'}
    write_json(run/'result.json',result);write_json(root/'artifacts/g1-state-prompts-reviewed.json',result)
    (run/'report.md').write_text('# G1 state prompt review\n\nPublic synthetic prompts manually reviewed for intent only. Targets are existing project style references, not truth labels; no automatic training import.\n\n'+json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result))


if __name__=='__main__':main()
