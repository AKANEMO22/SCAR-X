"""Owned P1A executor. Neural selection and fixture tool production stay separate."""
import copy
import importlib.util
from pathlib import Path
import time

from .neural_outcome_bridge import ADAPTER,MACROS
from .outcome_contract import canonical,digest,evaluate

ROOT=Path(__file__).resolve().parents[3]
spec=importlib.util.spec_from_file_location('_p1a_owned_P0_tools',ROOT/'scripts/run_hybrid_outcome_contract_p0.py')
P0=importlib.util.module_from_spec(spec);spec.loader.exec_module(P0)


class OwnedMacroSandbox:
    def __init__(self,run,name,task,journal,charge):
        request=task['request'];kind={'read_file':'read','poll':'job','create_alarm':'clarify'}[request['tool']]
        self.fixture=P0.Fixture(run,name,kind,[])
        if kind=='read':self.fixture.contract['source']=request['arguments']['path']
        if kind=='job':
            self.fixture.db.execute("UPDATE job SET progress=2,status='running'");self.fixture.db.commit()
        self.fixture.trace['initial_snapshot']=self.fixture.trace['final_snapshot']=self.fixture.capture()
        self.task=copy.deepcopy(task);self.journal=journal;self.charge=charge
        self.task_id=self.fixture.contract['task_id'];self.scope=self.fixture.contract['scope']
        journal.register_context(self.task_id,self.scope,self.task)
        registry={'read_file':{'arguments':['path'],'bounded_bytes':2048},'poll':{'arguments':[]},
                  'create_alarm':{'arguments':['alarm_time']}}
        (self.fixture.root/'registry.json').write_bytes(canonical(registry))

    def _query(self,tool,operation):
        f=self.fixture;cutoff=time.time();before=f.capture();started=time.time();timer=time.perf_counter()
        result=operation();returned=time.time();duration=(time.perf_counter()-timer)*1000
        after=f.capture()
        # A raw executor event is not itself a claim that a neural policy chose
        # it; the outer receipt binds the immutable neural decision separately.
        import uuid
        event={'index':len(f.trace['events']),'task_id':self.task_id,'scope':self.scope,
            'decision_id':uuid.uuid4().hex,'selector':'fixed_fixture',
            'proposed_action':tool,'executed_action':tool,'decision_cutoff':cutoff,
            'started_epoch':started,'returned_epoch':returned,'duration_ms':duration,
            'tool':tool,'arguments':{},'result':result,'result_sha256':digest(result),
            'before_snapshot':before,'after_snapshot':after,
            'observations_before_decision':P0.CHECK.observed_prefix(f.trace['events'],cutoff)}
        f.trace['events'].append(event);f.trace['final_snapshot']=after

    def dispatch(self,decision):
        self.journal.check_decision(decision)
        if decision['task_id']!=self.task_id or decision['scope']!=self.scope:
            raise ValueError('Executor belongs to another task/scope')
        if self.journal.has_receipt(decision['id']):raise ValueError('Decision already settled; no repeated dispatch')
        active=self.journal.context(self.task_id,self.scope,self.task,allow_revoked=True)
        attempt=self.charge('dispatch',decision['seed'],'action_macro:'+decision['id'])
        before=self.fixture.capture();start=len(self.fixture.trace['events']);proposed=decision['proposed_action']
        if not active:
            executed='deny';origin='runtime_scope_guard';reason='scope lease revoked after proposal'
        else:
            if proposed not in decision['allowed']:raise ValueError('Action was not permitted')
            executed=proposed;origin='neural_policy';reason=None
            request=self.task['request'];tool=request['tool'];f=self.fixture
            if proposed=='show_schema':
                def registry():
                    import json
                    data=json.loads((f.root/'registry.json').read_text(encoding='utf-8'))
                    return {'ok':True,'schema':data[tool]}
                self._query('show_schema',registry)
            elif proposed=='inspect_status':
                if tool=='read_file':
                    path=f.root/request['arguments']['path']
                    if path.exists() and path.stat().st_size>2048:raise ValueError('Source exceeds bound')
                    f.call('read_file',copy.deepcopy(request['arguments']))
                elif tool=='poll':
                    self._query('inspect_status',lambda:{'ok':True,**dict(zip(('progress','status'),
                        f.db.execute('SELECT progress,status FROM job').fetchone()))})
                else:
                    self._query('inspect_status',lambda:{'ok':True,'missing_fields':[
                        row[1] for row in f.db.execute('PRAGMA table_info(request)')
                        if f.db.execute('SELECT '+row[1]+' FROM request').fetchone()[0] is None]})
            elif proposed=='retry':
                if tool not in ('read_file','poll'):raise ValueError('No permitted retry implementation')
                if tool=='read_file':
                    path=f.root/request['arguments']['path']
                    if path.exists() and path.stat().st_size>2048:raise ValueError('Source exceeds bound')
                f.call(tool,copy.deepcopy(request['arguments']))
            elif proposed=='clarify':
                fields=[row[1] for row in f.db.execute('PRAGMA table_info(request)')
                        if f.db.execute('SELECT '+row[1]+' FROM request').fetchone()[0] is None]
                if tool!='create_alarm' or len(fields)!=1:raise ValueError('No unique supported missing field')
                f.call('ask_field',{'field':fields[0]})
            elif proposed not in ('defer','finish'):
                raise ValueError('Unknown macro')
        after=self.fixture.capture()
        outcome=evaluate(self.fixture.trace,self.fixture.snapshots,self.fixture.contract)
        new_events=self.fixture.trace['events'][start:]
        receipt={'decision_id':decision['id'],'task_id':self.task_id,'scope':self.scope,
            'adapter':ADAPTER,'macros':MACROS,'proposed_action':proposed,'executed_action':executed,
            'selector_origin':origin,'guard_override':not active,'guard_reason':reason,
            'before_snapshot':before,'after_snapshot':after,'outcome':outcome,
            'raw_tool_events':[{'tool_event_id':e['decision_id'],'sha256':digest(e)} for e in new_events],
            'trace_sha256':digest(self.fixture.trace),'learning_authorized':False,
            'dispatch_attempt':attempt['attempt'],
            'neural_credit_eligible':False,'created':time.time()}
        self.journal.record_receipt(decision,receipt)
        return copy.deepcopy(receipt)

    def observations(self):
        result=[];streak=0;previous=None
        for event in self.fixture.trace['events'][-8:]:
            signature=digest([event['tool'],event['arguments'],event['result']])
            streak=streak+1 if previous==signature else 1;previous=signature
            data=event['result']
            result.append({'tool':event['tool'],'reported_status':'ok' if data.get('ok') is True else 'error',
                'error_type':data.get('error',''),'empty_result':False,'spillover':False,
                'streak':streak,'probe':None,'returned_epoch':event['returned_epoch']})
        return result

    def save(self):
        f=self.fixture
        P0.write_json(f.directory/'public-task.json',self.task)
        P0.write_json(f.directory/'contract.json',f.contract)
        P0.write_json(f.directory/'trace.json',f.trace)

    def close(self):self.fixture.db.close()
