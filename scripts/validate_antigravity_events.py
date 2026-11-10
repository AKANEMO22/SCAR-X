"""Audit small generated dialogue candidates; no automatic learning labels."""
from pathlib import Path
from collections import Counter,defaultdict
import json,hashlib,sys,re
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
from scarx.neural.language import tokenize
from scarx.neural.input_events import InputEvents

if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
pointer=json.loads((root/'artifacts/antigravity-events-latest.json').read_text(encoding='utf-8'))
candidate=(root/pointer['candidate_directory']).resolve();assert candidate.is_relative_to((root/'data/candidates').resolve())
manifest=json.loads((candidate/'manifest.json').read_text(encoding='utf-8'))
payload=(candidate/'dialogue.json').read_bytes();assert hashlib.sha256(payload).hexdigest()==manifest['sha256']
dialogs=json.loads(payload)['dialogs'];counts=Counter((d['split'],d['language']) for d in dialogs)
issues=[];vocab=set();ids=set();templates=defaultdict(set);inputs=defaultdict(set);experience=[]
if len(dialogs)!=40:issues.append('expected40dialogs')
expected={('train','vi'):20,('train','en'):12,('development','vi'):4,('development','en'):4}
if dict(counts)!=expected:issues.append('language_or_split_count_mismatch')
for d in dialogs:
    if d['id'] in ids:issues.append('duplicate_dialog_id:'+d['id'])
    ids.add(d['id']);templates[d['template_id']].add(d['split'])
    if len(d['turns'])!=6:issues.append('expected6turns:'+d['id']);continue
    for i,t in enumerate(d['turns']):
        if t['role']!=('user' if i%2==0 else 'assistant'):issues.append('role_alternation:'+d['id'])
        if not isinstance(t['text'],str) or not 1<=len(t['text'])<=1024:issues.append('invalid_text:'+d['id']);continue
        if i%2:
            target=tokenize(t['text']);vocab.update(target)
            if not 1<=len(target)<=24:issues.append('codec_target_length:'+d['id'])
            inputs[' '.join(tokenize(d['turns'][i-1]['text']))].add(d['split'])
            prefix=d['turns'][:i]
            try:InputEvents(3).dialogue(prefix)
            except ValueError:issues.append('input_event_budget:'+d['id'])
            experience.append({'id':d['id']+':'+str(i//2),'dialogue_id':d['id'],'split':d['split'],'language':d['language'],
              'scenario_id':d['scenario_id'],'template_id':d['template_id'],'messages':prefix,'target':t['text'],
              'source':'synthetic:antigravity','verified':False,'automatic_learning_allowed':False})
if len(vocab)>200:issues.append('target_vocabulary_above200')
overlap=[key for key,values in templates.items() if len(values)>1]
if overlap:issues.append('template_overlap')
duplicate_inputs=[key for key,values in inputs.items() if len(values)>1]
if duplicate_inputs:issues.append('exact_user_utterance_overlap_across_splits')
tool_steps=[s for batch in manifest.get('batches',[manifest]) for s in batch.get('tool_steps',[])]
expected_read=(root/'local/antigravity-event-dialogue-jobs'/manifest.get('created_utc','')/'dialogue.json').resolve()
unexpected=[]
for step in tool_steps:
    if step.get('name')=='finish':continue # CLI schema-output delivery, not file/network access.
    read_path=step.get('parameters',{}).get('Path')
    if not (manifest.get('repair_of') and step.get('name')=='read_file' and isinstance(read_path,str) and Path(read_path).resolve()==expected_read):unexpected.append(step.get('name'))
if unexpected:issues.append('unexpected_tool_steps_require_review')
result={'candidate_directory':pointer['candidate_directory'],'dialogs':len(dialogs),'responses':len(experience),
 'split_language_counts':{str(k):v for k,v in counts.items()},'target_vocabulary_tokens':len(vocab),
 'schema_and_split_screen_pass':not issues,'issues':issues,'cross_split_exact_user_inputs':duplicate_inputs,
 'template_overlap':overlap,'role_history_preserved':True,'independently_verified_labels':0,
 'unexpected_tool_access':unexpected,
 'quality_limit':'Counts and template IDs do not independently prove naturalness, semantic correctness or absence of near-template leakage. All labels still require style/content review.',
 'trained':(candidate/'usage.json').exists(),'experiment_usage':json.loads((candidate/'usage.json').read_text(encoding='utf-8')) if (candidate/'usage.json').exists() else None,
 'live_auto_import':False,'source':manifest['source'],'model':manifest['model']}
(candidate/'experiences.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in experience),encoding='utf-8')
(candidate/'validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(root/'artifacts/antigravity-events-validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
pointer['validation_pending']=False;pointer['schema_screen_pass']=not issues
(root/'artifacts/antigravity-events-latest.json').write_text(json.dumps(pointer,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,ensure_ascii=True))
