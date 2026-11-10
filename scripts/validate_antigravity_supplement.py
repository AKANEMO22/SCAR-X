"""Audit generated candidates; never promotes synthetic labels to truth."""
from pathlib import Path
import collections, hashlib, json, re, sys

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'src'))
from scarx.neural.language import tokenize

pointer=json.loads((root/'artifacts/antigravity-supplement-latest.json').read_text(encoding='utf-8'))
candidate=(root/pointer['candidate_directory']).resolve()
assert candidate.is_relative_to((root/'data/candidates').resolve())
manifest=json.loads((candidate/'manifest.json').read_text(encoding='utf-8'))
for task in manifest['tasks']:
    assert task['name'] in ('dialogue','real_photo')
    assert hashlib.sha256((candidate/(task['name']+'.json')).read_bytes()).hexdigest()==task['sha256']
job=root/'local/antigravity-supplement-jobs'/manifest['created_utc']
dialogs=json.loads((candidate/'dialogue.json').read_text(encoding='utf-8'))['dialogs']
assert len(dialogs)==40 and len({d['id'] for d in dialogs})==40
vocabulary=set(); pairs=[]; by_split=collections.Counter(); labels=collections.defaultdict(set)
for d in dialogs:
    assert d['language'] in ('vi','en') and d['split'] in ('train','development')
    assert len(d['turns'])==6
    by_split[(d['language'],d['split'])]+=1
    for index,t in enumerate(d['turns']):
        assert t['role']==('user' if index%2==0 else 'assistant')
        assert isinstance(t['text'],str) and 0<len(t['text'])<=1024
        if index%2:
            target=tokenize(t['text']);assert 1<=len(target)<=24
            vocabulary.update(target)
            prompt=d['turns'][index-1]['text']
            labels[' '.join(tokenize(prompt))].add(t['text'])
            pairs.append({'id':d['id']+':'+str(index//2),'dialogue_id':d['id'],
              'scenario_id':d['scenario_id'],'split':d['split'],'language':d['language'],
              'turns':d['turns'][:index+1], 'target':t['text'],
              'source':'synthetic:antigravity','independently_verified':False,
              'automatic_learning_allowed':False})
assert len(vocabulary)<=200
train_scenarios={d['scenario_id'] for d in dialogs if d['split']=='train'}
dev_scenarios={d['scenario_id'] for d in dialogs if d['split']=='development'}
assert not train_scenarios&dev_scenarios
photos=json.loads((candidate/'real_photo.json').read_text(encoding='utf-8'))['images']
assert {p['image'] for p in photos}=={'china.jpg','flower.jpg'}
events=[json.loads(l) for l in (job/'real_photo.events.jsonl').read_text(encoding='utf-8').splitlines() if l.startswith('{')]
reads={}
for e in events:
    step=e.get('step_update',{})
    if step.get('step_type')=='tool' and step.get('tool_name')=='read_file' and step.get('state')=='DONE':
        p=Path(step.get('tool_info',{}).get('parameters',{}).get('Path','')).resolve()
        if p.parent==job.resolve() and p.name in ('china.jpg','flower.jpg'):reads[p.name]=True
image_audit=[]
# Only these exact images/annotations were reviewed in this work session.
reviewed_hashes={'china.jpg':'8378025ad2519d649d02e32bd98990db4ab572357d9f09841c2fbfbb4fefad29',
 'flower.jpg':'a77f6ec41e353afdf8bdff2ea981b2955535d8d83294f8cfa49cf4e423dd5638'}
for p in photos:
    assert p['pixels_seen'] is True and len(p['qa'])==8 and p['unknowns']
    assert (candidate/p['image']).read_bytes()==(job/p['image']).read_bytes()
    for qa in p['qa']:assert all(isinstance(qa[k],str) and qa[k] for k in ('question','answer','evidence_visible'))
    digest=hashlib.sha256((candidate/p['image']).read_bytes()).hexdigest()
    image_audit.append({'image':p['image'],'sha256':digest,
      'native_read_file_completed':reads.get(p['image'],False),
      'caption_and_qa_visually_reviewed_by_codex':manifest['created_utc']=='20261109T063155Z' and digest==reviewed_hashes.get(p['image']),
      'review_limit':'Broad objects/colors/relations match the displayed public photos. Exact lighting direction/natural light and distant boat type are not independently established.',
      'independently_verified_factual_labels':False,'automatic_learning_allowed':False})
result={'candidate_directory':pointer['candidate_directory'],'dialogs':len(dialogs),'dialogue_response_targets':len(pairs),
 'language_split_counts':{str(k):v for k,v in by_split.items()},'target_vocabulary_tokens':len(vocabulary),
 'held_out_scenarios':sorted(dev_scenarios),'conflicting_raw_prompts':sum(len(v)>1 for v in labels.values()),
 'evaluation_limit':'All train dialogues are Vietnamese; all development dialogues are English. Language and scenario shifts are confounded. Do not infer bilingual generalization or flatten multi-turn memory tasks into independent pairs.',
 'photos':image_audit,'visual_qa_pairs':16,'captions':2,
 'current_model_supports_normal_photo_qa':False,'trained':False,
 'decision':'CANDIDATES_ONLY; style review required; photo encoder/cross-modal learning missing; no live auto-import'}
(candidate/'dialogue-experiences.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in pairs),encoding='utf-8')
(candidate/'validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(root/'artifacts/antigravity-supplement-validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
pointer['validation_pending']=False
(root/'artifacts/antigravity-supplement-latest.json').write_text(json.dumps(pointer,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,ensure_ascii=True))
