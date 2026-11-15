"""One approved, bounded, model-free P0 on owned real files/SQLite.

No model outputs: all actions/final text are declared engineering fixtures.
Imports the pure checker by path, avoiding scarx.__init__'s NumPy/neural import.
"""
import ast
from datetime import datetime,timezone
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sqlite3
import struct
import sys
import time
import uuid
import zipfile
import psutil

ROOT=Path(__file__).resolve().parents[1]
MAX_SECONDS=300;MAX_BYTES=10_000_000;MAX_RSS=128*1024*1024
SPEC=importlib.util.spec_from_file_location('_pure_outcome_contract',ROOT/'src/scarx/evaluation/outcome_contract.py')
CHECK=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(CHECK)
encode=CHECK.canonical
TEXT='ALPHA-472: owned fixture content.\n'


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as stream:stream.write(encode(value))


def source_state():
    names=['scripts/run_hybrid_outcome_contract_p0.py','scripts/audit_hybrid_outcome_contract_p0.py',
           'src/scarx/evaluation/outcome_contract.py','tests/test_outcome_contract.py',
           'experiments/hybrid-stateful-outcome-contract-p0-v1.md']
    return {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names}


def live_learned_state():
    path=ROOT/'local/hybrid-v1/hybrid.sqlite'
    with sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True) as db:
        pointer=json.loads(db.execute("SELECT payload FROM objects WHERE kind='runtime' AND id='active_checkpoint'").fetchone()[0])
        expected,raw=db.execute('SELECT digest,payload FROM checkpoints WHERE id=?',(pointer['id'],)).fetchone()
    if hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('Live checkpoint integrity failure')
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        members=archive.namelist()
        if len(set(members))!=len(members):raise ValueError('Duplicate checkpoint member')
        arrays={n:hashlib.sha256(archive.read(n)).hexdigest() for n in members if n!='metadata.npy'}
        meta=archive.read('metadata.npy');major=meta[6]
        length=struct.unpack('<H' if major==1 else '<I',meta[8:10] if major==1 else meta[8:12])[0]
        start=10 if major==1 else 12
        header=ast.literal_eval(meta[start:start+length].decode('latin1'))
        if header['descr']!='|u1' or header['fortran_order']:raise ValueError('Unexpected metadata dtype')
        value=json.loads(meta[start+length:]);value.pop('controller')
    return {'arrays_sha256':arrays,'learned_metadata_sha256':CHECK.digest(value),'updates':value['updates']}


def publish(run,title,detail,status='completed'):
    payload={'kind':'technical_verification','title':title,'detail':detail,'status':status,
             'run_id':run.name,'evidence':str(run.relative_to(ROOT)),
             'measured':True,'scope':'Owned engineering fixtures; not model answers or neural thoughts'}
    # The only write to the live store is an explicitly attributed technical log.
    with sqlite3.connect(ROOT/'local/hybrid-v1/hybrid.sqlite',timeout=5) as db:
        db.execute('INSERT INTO events(created,actor,kind,payload) VALUES(?,?,?,?)',
                   (time.time(),'orchestrator','work_trace.activity',encode(payload).decode()))
    with sqlite3.connect(ROOT/'local/work-trace.sqlite',timeout=5) as db:
        db.execute('CREATE TABLE IF NOT EXISTS trace(id INTEGER PRIMARY KEY,kind TEXT,payload TEXT)')
        db.execute('INSERT INTO trace(kind,payload) VALUES(?,?)',('activity',encode({**payload,'kind':'activity','time':time.time()}).decode()))
        db.execute('DELETE FROM trace WHERE id NOT IN (SELECT id FROM trace ORDER BY id DESC LIMIT 600)')


def cases():
    # Freeze expected contract properties before any fixture executes.
    return [
        ('01-correct-source','read',[],[('read_file',{'path':'requested.txt'})],{'task_effect_satisfied':True,'whole_task_completion':True}),
        ('02-wrong-source-same-content','read',[],[('read_file',{'path':'other.txt'})],{'task_effect_satisfied':False,'whole_task_completion':False}),
        ('03-success-without-effect','update',[],[('acknowledge',{})],{'task_effect_satisfied':False,'whole_task_completion':False}),
        ('04-partial-effect','update',[],[('write_value',{'id':'target','value':'new'})],{'task_effect_satisfied':False,'whole_task_completion':False}),
        ('05-correct-update','update',[],[('commit',{'id':'target','value':'new'})],{'task_effect_satisfied':True,'whole_task_completion':True}),
        ('06-wrong-record','update',[],[('commit',{'id':'other','value':'new'})],{'task_effect_satisfied':False,'whole_task_completion':False}),
        ('07-order-reference','update',['prepare','commit'],[('prepare',{}),('commit',{'id':'target','value':'new'})],{'task_effect_satisfied':True,'dependency_order_satisfied':True,'whole_task_completion':True}),
        ('07-order-reversed','update',['prepare','commit'],[('commit',{'id':'target','value':'new'}),('prepare',{})],{'task_effect_satisfied':True,'dependency_order_satisfied':False,'whole_task_completion':False}),
        ('08-denied-before-success','update',[],[('forbidden_update',{'id':'target'}),('commit',{'id':'target','value':'new'})],{'task_effect_satisfied':True,'forbidden_attempts':1,'whole_task_completion':False}),
        ('09-loop-before-correct-read','read',[],[('read_file',{'path':'missing.txt'})]*24+[('read_file',{'path':'requested.txt'})],{'task_effect_satisfied':True,'had_nonprogress_loop':True,'repeated_nonprogress_calls':23,'terminal_stuck':False,'whole_task_completion':False}),
        ('10-progressing-polls','job',[],[('poll',{})]*3,{'task_effect_satisfied':True,'had_nonprogress_loop':False,'whole_task_completion':True}),
        ('11-finaltext-without-proof','read',[],[],{'task_effect_satisfied':False,'status':'unverified','whole_task_completion':False}),
        ('12-wrong-clarification','clarify',[],[('ask_field',{'field':'login'})],{'task_effect_satisfied':False,'clarification_supported':False,'whole_task_completion':False}),
        ('12-supported-clarification','clarify',[],[('ask_field',{'field':'alarm_time'})],{'task_effect_satisfied':False,'clarification_supported':True,'whole_task_completion':False}),
    ]


class Fixture:
    def __init__(self,run,name,kind,order):
        self.directory=run/'fixtures'/name;self.directory.mkdir(parents=True)
        self.root=self.directory/'project';self.root.mkdir()
        (self.root/'requested.txt').write_bytes(TEXT.encode('utf-8'))
        (self.root/'other.txt').write_bytes(TEXT.encode('utf-8'))
        self.db=sqlite3.connect(self.root/'state.sqlite')
        self.db.executescript('''CREATE TABLE records(id TEXT PRIMARY KEY,value TEXT,status TEXT);
            INSERT INTO records VALUES('target','old','pending'),('other','untouched','pending');
            CREATE TABLE preparation(done INTEGER);INSERT INTO preparation VALUES(0);
            CREATE TABLE job(progress INTEGER,status TEXT);INSERT INTO job VALUES(0,'running');
            CREATE TABLE request(alarm_time TEXT);INSERT INTO request VALUES(NULL);
            CREATE TABLE clarifications(field TEXT);''');self.db.commit()
        task_id=uuid.uuid4().hex
        self.contract={'format':CHECK.FORMAT,'task_id':task_id,'scope':'owned:'+name,
            'kind':kind,'source':'requested.txt' if kind=='read' else 'target',
            'value':hashlib.sha256(TEXT.encode()).hexdigest() if kind=='read' else 'new',
            'required_order':order,'missing_field':'alarm_time' if kind=='clarify' else None}
        self.snapshots={};initial=self.capture()
        self.trace={'format':CHECK.FORMAT,'task_id':task_id,'scope':self.contract['scope'],
            'initial_snapshot':initial,'final_snapshot':initial,'events':[],
            'final_text':'Declared fixture status: completed' if name.startswith(('06-','11-')) else '',
            'final_text_origin':'fixed_fixture_not_model'}

    def capture(self):
        files={}
        for path in sorted(self.root.glob('*.txt')):
            if path.is_symlink() or not path.resolve().is_relative_to(self.root.resolve()):raise ValueError('File escape')
            raw=path.read_bytes();files[path.name]={'text':raw.decode('utf-8'),'sha256':hashlib.sha256(raw).hexdigest()}
        value={'files':files,'records':[dict(zip(('id','value','status'),row)) for row in self.db.execute('SELECT id,value,status FROM records ORDER BY id')],
            'prepared':self.db.execute('SELECT done FROM preparation').fetchone()[0],
            'job':dict(zip(('progress','status'),self.db.execute('SELECT progress,status FROM job').fetchone())),
            'request':{'alarm_time':self.db.execute('SELECT alarm_time FROM request').fetchone()[0]},
            'request_schema':[row[1] for row in self.db.execute('PRAGMA table_info(request)')],
            'clarifications':[row[0] for row in self.db.execute('SELECT field FROM clarifications ORDER BY rowid')]}
        key=CHECK.digest(value)
        if key not in self.snapshots:
            write_json(self.directory/'snapshots'/(key+'.json'),value);self.snapshots[key]=value
        return key

    def call(self,tool,args):
        cutoff=time.time();view=CHECK.observed_prefix(self.trace['events'],cutoff)
        before=self.capture();started=time.time();timer=time.perf_counter();executed=tool
        try:
            if tool=='forbidden_update':
                executed='deny';raise PermissionError('Owned fixture registry disallows this proposed operation')
            if tool=='read_file':
                path=self.root/args['path']
                if path.is_symlink() or not path.resolve().is_relative_to(self.root.resolve()):raise ValueError('Path escape')
                result={'ok':True,'source':args['path'],'text':path.read_text(encoding='utf-8')}
            elif tool in ('write_value','commit'):
                if tool=='commit':
                    self.db.execute("UPDATE records SET value=?,status='committed' WHERE id=?",(args['value'],args['id']))
                else:self.db.execute('UPDATE records SET value=? WHERE id=?',(args['value'],args['id']))
                self.db.commit();result={'ok':True,'changed_rows':self.db.execute('SELECT changes()').fetchone()[0]}
            elif tool=='prepare':
                self.db.execute('UPDATE preparation SET done=1');self.db.commit();result={'ok':True,'prepared':True}
            elif tool=='acknowledge':result={'ok':True,'scope':'acknowledgement only; no requested state mutation'}
            elif tool=='poll':
                self.db.execute("UPDATE job SET progress=min(progress+1,3),status=CASE WHEN progress+1>=3 THEN 'completed' ELSE 'running' END")
                self.db.commit();row=self.db.execute('SELECT progress,status FROM job').fetchone()
                result={'ok':True,'progress':row[0],'status':row[1]}
            elif tool=='ask_field':
                self.db.execute('INSERT INTO clarifications VALUES(?)',(args['field'],));self.db.commit()
                result={'ok':True,'pending_field':args['field'],'sent_to_real_person':False}
            else:raise ValueError('Unknown tool fixture')
        except (FileNotFoundError,PermissionError) as exc:
            result={'ok':False,'error':type(exc).__name__,'message':str(exc)}
        returned=time.time();duration=(time.perf_counter()-timer)*1000
        after=self.capture()
        event={'index':len(self.trace['events']),'task_id':self.contract['task_id'],'scope':self.contract['scope'],
            'decision_id':uuid.uuid4().hex,'selector':'fixed_fixture','proposed_action':tool,'executed_action':executed,
            'decision_cutoff':cutoff,'started_epoch':started,'returned_epoch':returned,'duration_ms':duration,
            'tool':tool,'arguments':args,'result':result,'result_sha256':CHECK.digest(result),
            'before_snapshot':before,'after_snapshot':after,'observations_before_decision':view}
        self.trace['events'].append(event);self.trace['final_snapshot']=after


def main():
    started=time.monotonic();start_epoch=time.time();stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run=ROOT/'runs'/(stamp+'-outcome-contract-p0-v1-'+uuid.uuid4().hex[:6]);run.mkdir()
    marker=ROOT/'local/outcome-contract-p0-v1-execution.json'
    write_json(marker,{'run':str(run),'started_epoch':start_epoch,'single_approved_run':True})
    source=source_state();live_before=live_learned_state();definitions=cases()
    locked=[{'name':n,'kind':k,'required_order':o,'actions':a,'expected':e} for n,k,o,a,e in definitions]
    write_json(run/'locked-cases.json',locked)
    for name in source:
        dest=run/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/name).read_bytes())
    write_json(run/'manifest.json',{'format':CHECK.FORMAT,'source_sha256':source,
        'source_git_sha':os.environ.get('SCARX_OUTCOME_SOURCE_SHA'),
        'locked_cases_sha256':hashlib.sha256((run/'locked-cases.json').read_bytes()).hexdigest(),
        'started_epoch':start_epoch,'max_seconds':MAX_SECONDS,'max_artifact_bytes':MAX_BYTES,
        'max_RSS_bytes':MAX_RSS,'fixtures':12,'traces':14,'no_neural_or_VLM_calls':True,
        'live_before':live_before,'user_approval':'làm đi,15/11/2026; isolated P0 only',
        'literature_blob':'af9faacc65c0e700a52093fb0c5d77ac6c36c449'})
    allowed_sqlite={ROOT/'local/hybrid-v1/hybrid.sqlite',ROOT/'local/work-trace.sqlite'}
    blocked=[]
    def audit_guard(event,args):
        if event.startswith('socket.') or event in ('subprocess.Popen','os.system','os.posix_spawn'):
            blocked.append(event);raise PermissionError('No network/process/model calls in P0')
        if event=='import' and str(args[0]).split('.')[0] in ('numpy','cupy','torch','scarx'):
            blocked.append('neural_import');raise PermissionError('Pure P0 disallows neural packages')
        if event=='sqlite3.connect':
            if str(args[0])==(ROOT/'local/hybrid-v1/hybrid.sqlite').resolve().as_uri()+'?mode=ro':return
            p=Path(args[0]).resolve()
            if not p.is_relative_to(run.resolve()) and p not in allowed_sqlite:
                blocked.append('external_database');raise PermissionError('Database outside P0 scope')
        if event=='open':
            path,mode,flags=args
            writing=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC)))
            if writing and isinstance(path,(str,bytes,os.PathLike)):
                p=Path(os.fsdecode(path)).resolve()
                if not p.is_relative_to(run.resolve()):
                    blocked.append('external_write');raise PermissionError('Write outside P0 artifact directory')
    sys.addaudithook(audit_guard)
    report={'run_id':run.name,'cases':[],'gate_pass':False,'failures':[],
            'fixture_not_model_answers':True,'neural_forward_or_learning_calls':0,'native_VLM_ASR_calls':0,
            'scientific_or_production_PASS':False}
    peak=psutil.Process().memory_info().rss
    def check_budget():
        nonlocal peak
        memory=psutil.Process().memory_info();peak=max(peak,memory.rss,getattr(memory,'peak_wset',0))
        if time.monotonic()-started>MAX_SECONDS or peak>MAX_RSS:raise RuntimeError('Time/RSS budget exceeded')
        if sum(p.stat().st_size for p in run.rglob('*') if p.is_file())>MAX_BYTES:raise RuntimeError('Artifact budget exceeded')
    try:
        publish(run,'Outcome contract P0 started','12 fixtures /14 traces; real owned files/SQLite,0 model calls.','started')
        for name,kind,order,actions,expected in definitions:
            check_budget();fixture=Fixture(run,name,kind,order)
            try:
                write_json(fixture.directory/'contract.json',fixture.contract)
                for tool,args in actions:check_budget();fixture.call(tool,args)
                write_json(fixture.directory/'trace.json',fixture.trace)
                actual=CHECK.evaluate(fixture.trace,fixture.snapshots,fixture.contract)
                write_json(fixture.directory/'outcome.json',actual)
                if any(actual.get(k)!=v for k,v in expected.items()):raise AssertionError('Locked outcome mismatch: '+name)
                report['cases'].append({'name':name,'expected':expected,'actual':actual})
                publish(run,'P0 '+name,json.dumps(actual,ensure_ascii=False))
            finally:fixture.db.close()
        live_after=live_learned_state();write_json(run/'live-state-audit.json',{'before':live_before,'after':live_after,'equal':live_before==live_after})
        if live_before!=live_after:raise RuntimeError('Live learned state changed; no restore performed')
        if source_state()!=source:raise RuntimeError('Source changed during execution')
        if blocked:raise RuntimeError('Unexpected side-effect attempt')
        forbidden_modules=[n for n in sys.modules if n.split('.')[0] in ('numpy','cupy','torch','scarx')]
        if forbidden_modules:raise RuntimeError('Unexpected neural import')
        check_budget();report.update(gate_pass=True,verdict='P0_EXACT_OUTCOME_CONTRACT_FIXTURES_PASS_NOT_NEURAL_SKILL',
            fixtures=12,traces=14,tool_calls=sum(c['actual']['tool_calls'] for c in report['cases']),
            live_learned_state_equal=True,neural_packages_loaded=forbidden_modules)
    except Exception as exc:
        report['failures'].append(type(exc).__name__+': '+str(exc));report['verdict']='P0_STOPPED_NOT_PASS'
    finally:
        report.update(elapsed_seconds=time.monotonic()-started,peak_process_RSS_bytes=peak,blocked_attempts=blocked)
        write_json(run/'result.json',report)
        publish(run,'Outcome contract P0 '+('completed' if report['gate_pass'] else 'stopped'),
                json.dumps({k:v for k,v in report.items() if k!='cases'},ensure_ascii=False),
                'completed' if report['gate_pass'] else 'failed')
    print(json.dumps({'run':str(run),**{k:v for k,v in report.items() if k!='cases'}},ensure_ascii=False),flush=True)
    return 0 if report['gate_pass'] else 1


if __name__=='__main__':raise SystemExit(main())
