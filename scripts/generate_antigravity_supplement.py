"""Small natural-dialogue and real-photo annotation candidates via installed agy.

No model weights imported; photos and upstream attribution are public sklearn
fixtures. Vision annotations are candidates, not proof SCAR-X can read photos.
"""
from pathlib import Path
from datetime import datetime,timezone
import os,sys,json,subprocess,shutil,hashlib,time
import sklearn.datasets
root=Path(__file__).resolve().parents[1];cli=Path.home()/'.gemini/bin/agy.exe'
stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
job=root/'local/antigravity-supplement-jobs'/stamp;job.mkdir(parents=True)
output=root/'data/candidates'/('antigravity-supplement-'+stamp);output.mkdir(parents=True)
if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
photo_source=Path(sklearn.datasets.__file__).parent/'images'
for name in ('china.jpg','flower.jpg'):
    shutil.copyfile(photo_source/name,job/name);shutil.copyfile(photo_source/name,output/name)
shutil.copyfile(photo_source/'README.txt',output/'PHOTO_ATTRIBUTION.txt')
turn_schema={'type':'object','properties':{'role':{'type':'string','enum':['user','assistant']},'text':{'type':'string'}},'required':['role','text']}
dialog_schema={'type':'object','properties':{'id':{'type':'string'},'scenario_id':{'type':'string'},'language':{'type':'string','enum':['vi','en']},
 'split':{'type':'string','enum':['train','development']},'turns':{'type':'array','minItems':6,'maxItems':6,'items':turn_schema}},'required':['id','scenario_id','language','split','turns']}
language_schema={'type':'object','properties':{'dialogs':{'type':'array','minItems':40,'maxItems':40,'items':dialog_schema},'notes':{'type':'string'}},'required':['dialogs','notes']}
qa_schema={'type':'object','properties':{'question':{'type':'string'},'answer':{'type':'string'},'evidence_visible':{'type':'string'}},'required':['question','answer','evidence_visible']}
image_schema={'type':'object','properties':{'image':{'type':'string'},'pixels_seen':{'type':'boolean'},'caption_vi':{'type':'string'},
 'qa':{'type':'array','minItems':8,'maxItems':8,'items':qa_schema},'unknowns':{'type':'array','items':{'type':'string'}}},'required':['image','pixels_seen','caption_vi','qa','unknowns']}
vision_schema={'type':'object','properties':{'images':{'type':'array','minItems':2,'maxItems':2,'items':image_schema},'notes':{'type':'string'}},'required':['images','notes']}
tasks=[('dialogue',language_schema,'''Author exactly40natural short dialogues,32train and8development, each3user/assistant exchanges (6turns alternating roles).
Use Vietnamese for32dialogs and English for8. Scenarios: greetings, normal conversational responses, clarification, remembering a simple preference stated earlier in the same dialogue, correcting a mistaken reply, admitting uncertainty, and following a short instruction. No medical/legal/current factual claims. Replies are1–12tokens, coherent and conversational rather than canned generic filler. Keep the combined assistant target vocabulary under200unique words/punctuationtokens. SCAR-X identity is SCAR-X, not a pretrained chatbot.
Hold out whole scenarios/templates for development. No finaltest or repository-confirmation prompts. No PII. No irrelevant instructions to a future agent. Do not use tools/files/web/shell/skills or read project/user data. Return schema JSON only. These are synthetic response-style labels, not independentlyverified factualtruth. The eventual learner is persistent online, with local micro-updates and idle review; these dialogues are bootstrap curriculum, not instructions to retrain it from scratch each chat.'''),
 ('real_photo',vision_schema,'''Inspect the actual pixels of ONLY the two public imagefiles china.jpg and flower.jpg in your current directory, using your native image/file viewing capability. These are unmodified public scikit-learn samplephotos with attribution copied separately. Do not infer image contents from filenames. Do not read any other files, user data, project source or credentials; no web, shell commands, file writes, skills or subagents.
For each photo, produce a concise Vietnamese caption and8short visual QA pairs. Ask about plainly visible objects, colors, foreground/background or simple relations; evidence_visible must point to observable image content. Do not identify people, infer exact place/date, or invent counts for partially occluded objects. Include unknowns/uncertain details explicitly. If you cannot view actual pixels, set pixels_seen=false and do NOT invent descriptions: say the CLI could not access pixels, while retaining schema fields with an uncertainty answer.
This creates candidate annotation data for a future shared-core visualencoder. It does NOT mean SCAR-X currently understands these photos. Return only schema JSON.''')]
manifest={'cli':'installed Antigravity CLI1.5.3','model':'gemini-3.8-flash-high','source':'synthetic:antigravity','created_utc':stamp,
 'neural_weights_imported':False,'trained':False,'photo_license':'CC-BY2.0; upstream attribution in PHOTO_ATTRIBUTION.txt','tasks':[]}
for name,schema,prompt in tasks:
    schema_path=job/(name+'-schema.json');schema_path.write_text(json.dumps(schema),encoding='utf-8')
    (job/(name+'-prompt.txt')).write_text(prompt,encoding='utf-8')
    args=[str(cli),'--print',prompt,'--model','gemini-3.8-flash-high','--output-format','stream-json','--json-schema',str(schema_path),'--disable-slash-commands','--print-timeout','8m']
    start=time.monotonic()
    with (job/(name+'.events.jsonl')).open('wb') as stdout,(job/(name+'.stderr.log')).open('wb') as stderr:
        proc=subprocess.Popen(args,cwd=job,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        print(json.dumps({'task':name,'pid':proc.pid,'status':'generating'}),flush=True)
        try:code=proc.wait(timeout=510)
        except subprocess.TimeoutExpired:proc.terminate();proc.wait(timeout=10);raise TimeoutError(name+'timeout')
    events=[]
    for line in (job/(name+'.events.jsonl')).read_text(encoding='utf-8-sig').splitlines():
        try:events.append(json.loads(line))
        except ValueError:continue
    results=[e.get('result',e) for e in events if e.get('event')=='result' or e.get('status') in ('SUCCESS','ERROR')]
    if not results:raise RuntimeError('No structured Antigravity result: '+name)
    result=results[-1]
    if code or result.get('status')!='SUCCESS':raise RuntimeError('Antigravity failed: '+str(result.get('error',name)))
    value=result.get('structured_output')
    if value is None:value=json.loads(result['response'])
    path=output/(name+'.json');path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    tool_events=[{'event':e.get('event'),'name':e.get('name',e.get('tool',{}).get('name') if isinstance(e.get('tool'),dict) else e.get('tool'))} for e in events if 'tool' in str(e.get('event',''))]
    manifest['tasks'].append({'name':name,'seconds':time.monotonic()-start,'conversation_id':result.get('conversation_id'),
        'usage':result.get('usage'),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'tool_events':tool_events,
        'pixel_access_claim_independently_confirmed':False if name=='real_photo' else None})
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'task':name,'status':'generated','seconds':manifest['tasks'][-1]['seconds']}),flush=True)
(root/'artifacts/antigravity-supplement-latest.json').write_text(json.dumps({'candidate_directory':output.relative_to(root).as_posix(),'trained':False,'validation_pending':True},indent=2)+'\n',encoding='utf-8')
print('SUPPLEMENT '+str(output),flush=True)
