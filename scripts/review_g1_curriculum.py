"""Full semantic review of160 generated prompts, quarantining ambiguous intents."""
from pathlib import Path
import sys,json,hashlib
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
from scarx.evaluation.runs import new_run,write_json
from generate_g1_state_paraphrases import INTENTS
from generate_g1_curriculum import key

QUARANTINE={
 'c1-greeting_vi-07':'Good-day wish is ambiguous between greeting and farewell.',
 'c1-greeting_vi-08':'Pleasure in conversation has no explicit greeting intent.',
 'c1-greeting_en-06':'Nice to talk may close or begin a conversation.',
 'c1-greeting_en-07':'Good-day wish is ambiguous between greeting and farewell.',
 'c1-farewell-05':'Have a nice day is ambiguous without conversation context.',
 'c1-teaching-06':'Giving instructions may mean ordinary task use rather than learning.',
 'c2-greeting_en-03':'Good-day wish is ambiguous between greeting and farewell.',
 'c2-greeting_en-09':'Good-day wish is ambiguous between greeting and farewell.',
 'c2-farewell-01':'Evening wish could be a greeting, not an explicit goodbye.',
 'c2-farewell-04':'Unnatural Vietnamese wording; poor language-teaching example.',
 'c2-farewell-08':'Evening wish ambiguous between greeting and farewell.',
 'c2-teaching-08':'Instructions may mean ordinary task use rather than learning.'}


def main():
    info=json.loads((root/'artifacts/antigravity-g1-curriculum-latest.json').read_text(encoding='utf-8'))
    directory=(root/info['candidate_directory']).resolve();assert directory.is_relative_to(root/'data/candidates')
    raw=(directory/'curriculum.json').read_bytes();assert hashlib.sha256(raw).hexdigest()==info['sha256'];rows=json.loads(raw)['items']
    corpus=json.loads((root/'data/dialogue/generation1.json').read_text(encoding='utf-8'));retained=[];quarantine=[]
    for row in rows:
        if row['id'] in QUARANTINE:quarantine.append({**row,'reason':QUARANTINE[row['id']]});continue
        family=corpus['families'][INTENTS.index(row['intent'])]
        retained.append({**row,'target':family['answer'],'source':'actual Antigravity CLI; explicit Codex semantic review, fixed project style target',
            'label_kind':'one stylistic response for known intent, not a fact oracle or general-language certification',
            'automatic_learning_allowed':False})
    assert len(retained)==148 and len(quarantine)==12
    assert len(set(key(r['prompt']) for r in retained))==len(retained)
    assert not ({key(r['prompt']) for r in retained if r['split']=='train'} & {key(r['prompt']) for r in retained if r['split']=='development'})
    counts={intent:{split:sum(r['intent']==intent and r['split']==split for r in retained) for split in ('train','development')} for intent in INTENTS}
    run=new_run(root,'g1-curriculum-semantic-review',{'display_title':'Rà soát curriculum · bỏ12câu mơ hồ, chưa học',
        'source_run':info['run'],'items':160,'retained':148,'quarantined':12,'new_training':False},
        data_description='All160public synthetic prompts read; no model performance used for review/labels')
    path=directory/'reviewed-curriculum.json';write_json(path,{'items':retained,'quarantine':quarantine,'reviewed_before_model_training':True,'counts':counts})
    result={'run':run.relative_to(root).as_posix(),'source_run':info['run'],'source_generation_run':info.get('source_generation_run'),
        'data':path.relative_to(root).as_posix(),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'decision':'CURATED_PUBLIC_STYLE_CURRICULUM','retained':148,'quarantined':12,'counts':counts,
        'train':sum(r['split']=='train' for r in retained),'development':sum(r['split']=='development' for r in retained),
        'new_training':False,'validation_pending':False,'automatic_learning_allowed':False,
        'limits':['Known8intent style targets, not factual labels; English questions may map to oldVietnamese style response.',
            'Train/dev exactsurface disjoint, not a guarantee of novel semantic templates or broad normal conversation.']}
    write_json(run/'result.json',result);write_json(root/'artifacts/g1-curriculum-reviewed.json',result)
    (run/'report.md').write_text('# Semantic review of small G1 curriculum\n\nAll160prompts inspected before studenttraining;12ambiguous/awkward examples quarantined, no performance-based relabeling.\n\n'+json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result))


if __name__=='__main__':main()
