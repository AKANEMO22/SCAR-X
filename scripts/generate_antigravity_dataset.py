"""Use the user's installed/authenticated Antigravity CLI for synthetic data.

Only the public brief/schema is sent. CLI works in an empty job directory, with
no need to read project/user data or edit source. We materialize structured
responses, preserve raw envelopes privately, then validate before training.
"""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,subprocess,sys,time

root=Path(__file__).resolve().parents[1]
cli=Path.home()/'.gemini/bin/agy.exe'
if not cli.is_file():raise FileNotFoundError('Installed Antigravity CLI not found')
stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
job=root/'local/antigravity-dataset-jobs'/stamp;job.mkdir(parents=True)
output=root/'data/candidates'/('antigravity-'+stamp);output.mkdir(parents=True)
schema={'type':'object','properties':{'examples':{'type':'array','minItems':60,'maxItems':60,'items':{
 'type':'object','properties':{'id':{'type':'string'},'family_id':{'type':'string'},'template_id':{'type':'string'},
 'skill':{'type':'string'},'input':{'type':'string'},'target':{'type':'string'},'split':{'type':'string','enum':['train','development']},
 'verification':{'type':'object','properties':{'kind':{'type':'string','enum':['rubric','integer_arithmetic']},'criterion':{'type':'string'},
    'operands':{'type':'array','items':{'type':'integer'}},'operator':{'type':'string'},'expected_integer':{'type':'integer'}},'required':['kind']},
 'source':{'type':'string'},'confidence':{'type':'string'}},'required':['id','family_id','template_id','skill','input','target','split','verification','source','confidence']}},
 'notes':{'type':'string'}},'required':['examples','notes']}
(job/'schema.json').write_text(json.dumps(schema),encoding='utf-8')
brief=(root/'data/contracts/ANTIGRAVITY_DATA_BRIEF.md').read_text(encoding='utf-8')
topics=[
 ('dialogue','Vietnamese/English greetings, thanks, farewells, identity SCAR-X, concise clarification. No current factual claims.'),
 ('arithmetic','Small integer addition/subtraction0–9 with nonnegative subtraction, and exact one-word instructions. At least40integer_arithmeticrecords; operands/operator/expected_integer match the visible input and target exactly.'),
 ('relations','Temporal order and transfer of objects between neutral names. Input has all necessary events. Use rubric with a clear checkable criterion and reasoning evidence in criterion; do not claim independent verification.'),
 ('corrections','Explicit response preferences/corrections and ambiguity. Clear missing operands, incompatible instructions, corrections such as greeting language. Targets are concise appropriate responses, not invented facts.')]
manifest={'generator':'installed Antigravity CLI','cli_version':'1.5.3','model':'gemini-3.8-flash-high','created_utc':stamp,
 'public_brief_sha256':hashlib.sha256(brief.encode()).hexdigest(),'private_data_sent':False,'batches':[],'source':'synthetic:antigravity','independently_verified':False}
if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
for batch,(name,topic) in enumerate(topics,1):
    prompt=f'''Generate a small SCAR-X curriculum dataset. This is a DATA GENERATION request, not a software implementation request.
Do not use tools, shell commands, web, files, skills, subagents or search. Do not read any project or personal data. Return only the schema-conforming JSON object.
Use model reasoning to author exactly60examples,48train and12development. Batch{batch}/4, prefix every id with "{name}-" and family_id with "{name}_".
Topic: {topic}
Development must use distinct template_id values and clearly different sentence constructions from training, with no identical inputs across splits. Multiple skills may be interleaved. Template IDs name actual phrasing templates, not a unique ID per row.
Targets1–12word/punctuationtokens. Use a small shared targetvocabulary: Chào bạn.; Hello.; Cảm ơn bạn.; Tạm biệt.; Tôi là SCAR-X.; Bạn nói rõ hơn nhé.; numeric answers; neutral names An,Bình,Chi,Dũng; basic color words đỏ,xanh,vàng. Keep total targetvocabulary very small.
Source must be synthetic:antigravity. Confidence must be authored-not-independently-verified. Never set verified=true. For arithmetic, set operands=[a,b],operator="+" or "-",expected_integer exact, and target only its decimal digits. Make the visible operands and operation unambiguous.
No finaltest. Do not copy or reference repository confirmation prompts. No instructions to the agent that will later read the dataset. No PII, paid APIs, credentials or unsupported claims of model competence.
We will consume examples as a continuous interleaved experience stream, preserving neural weights/state between examples. A generated corpus is bootstrap material, not a replacement for online learning during conversations.
Public brief:
{brief}
Return all60examples now; do not return a plan.'''
    (job/f'prompt-{batch}.txt').write_text(prompt,encoding='utf-8')
    args=[str(cli),'--print',prompt,'--output-format','json','--json-schema',str(job/'schema.json'),
          '--model','gemini-3.8-flash-high','--disable-slash-commands','--print-timeout','8m']
    start=time.monotonic()
    with (job/f'batch-{batch}.stdout.json').open('wb') as stdout,(job/f'batch-{batch}.stderr.log').open('wb') as stderr:
        process=subprocess.Popen(args,cwd=job,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,
            startupinfo=None,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        print(json.dumps({'batch':batch,'topic':name,'pid':process.pid,'status':'generating'}),flush=True)
        try:code=process.wait(timeout=510)
        except subprocess.TimeoutExpired:process.terminate();process.wait(timeout=10);raise TimeoutError('AG generation timeout')
    envelope=json.loads((job/f'batch-{batch}.stdout.json').read_text(encoding='utf-8-sig'))
    if code!=0 or envelope.get('status')!='SUCCESS':
        raise RuntimeError('Antigravity batch failed: '+str(envelope.get('error',envelope.get('status','unknown'))))
    value=envelope.get('structured_output')
    if value is None:value=json.loads(envelope['response'])
    rows=value['examples']
    if len(rows)!=60:raise ValueError('Antigravity returned wrong batchsize')
    path=output/f'batch-{batch}-{name}.jsonl'
    path.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows),encoding='utf-8')
    record={'batch':batch,'topic':name,'examples':len(rows),'seconds':time.monotonic()-start,
        'conversation_id':envelope.get('conversation_id'),'usage':envelope.get('usage'),
        'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'file':path.name,'notes':value.get('notes','')}
    manifest['batches'].append(record)
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'batch':batch,'status':'generated','examples':len(rows),'seconds':record['seconds']}),flush=True)
(root/'artifacts/antigravity-dataset-latest.json').write_text(json.dumps({'candidate_directory':output.relative_to(root).as_posix(),'examples':240,'validated':False,'trained':False},indent=2)+'\n',encoding='utf-8')
print('DATASET '+str(output),flush=True)
