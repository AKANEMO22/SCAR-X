"""Generation 1 session: one live core, bounded idle scheduler and atomic state."""
from collections import deque
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import copy
import io
import json
from pathlib import Path
import sqlite3
import threading
import time
import uuid
import zipfile
import numpy as np
import psutil
from ..neural.unified import UnifiedCore
from ..runtime.resources import ResourceMonitor
from .locking import DirectoryLease
from .workload import plan_workload,finish_workload
from ..learning.dialogue import DialogueLearning,normalized

MODEL = 'scarx-generation1'
ROOT = Path(__file__).resolve().parents[3]

# Product prompts whose currently-correct answers count as a protected skill.
# Used only to decide WHICH passing pairs to guard; it is not an answer table and
# is never consulted during generation. A pair is added only if the model already
# answers it correctly, so this cannot manufacture capability.
PRODUCT_GUARD_PROMPTS = {
    'hi': 'hello',
    'hello': 'hello',
    'chào': 'chào bạn',
    'chào bạn': 'chào bạn',
    'bạn là ai': 'scar-x',
    'bạn tên gì': 'scar-x',
    'cảm ơn': 'không có gì',
    'tạm biệt': 'tạm biệt',
}


class LearningGuardRejected(ValueError):
    def __init__(self,metrics):
        super().__init__('Learning guard rejected: lost a previously correct probe')
        self.metrics=metrics


class Generation:
    def __init__(self, directory, seed=17, background=True, release=None):
        self.directory = Path(directory); self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock(); self.queue_lock = threading.Lock()
        self.pending = 0; self.stop_event = threading.Event(); self.closed = False
        self.workload = None
        self.dialogue = DialogueLearning()
        self.trace = deque(maxlen=256); self.sequence = 0
        self.idle_enabled = True; self.idle_hz = 20; self.idle_steps = 0
        self.probe = None; self.replay_cursor = 0; self.memories = []
        self.rng = np.random.default_rng(seed); self.active = None
        self.limit_status = {'enforced': False}; self.state = 'idle'; self.error = None
        self.lease = DirectoryLease(self.directory/'session.lock')
        self.db = None; self.monitor = None
        try:
            self.db = sqlite3.connect(self.directory/'generation1.sqlite', check_same_thread=False)
            self.db.execute('PRAGMA journal_mode=WAL'); self.db.execute('PRAGMA synchronous=FULL')
            self.db.executescript('''CREATE TABLE IF NOT EXISTS checkpoints(id TEXT PRIMARY KEY,created TEXT,action TEXT,sha TEXT,payload BLOB);
                CREATE TABLE IF NOT EXISTS runtime(singleton INTEGER PRIMARY KEY CHECK(singleton=1),sha TEXT,payload BLOB);
                CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY,created TEXT,action TEXT,details TEXT);''')
            from .learning_channel import initialize as initialize_learning_channel
            initialize_learning_channel(self)
            row = self.db.execute('SELECT sha,payload FROM runtime WHERE singleton=1').fetchone()
            if row:
                if hashlib.sha256(row[1]).hexdigest() != row[0]: raise ValueError('Session checksum mismatch; giữ database để phục hồi.')
                self._unpack(row[1])
            else:
                descriptor = Path(release) if release else ROOT/'artifacts/generation1-release.json'
                info = json.loads(descriptor.read_text(encoding='utf-8'))
                checkpoint = (ROOT/info['checkpoint']).resolve()
                if not checkpoint.is_relative_to((ROOT/'artifacts').resolve()): raise ValueError('Release path outside artifacts')
                payload = checkpoint.read_bytes()
                if hashlib.sha256(payload).hexdigest() != info['sha256']: raise ValueError('Release checksum mismatch')
                self.model = UnifiedCore.loads(payload)
                self.memories = [{'prompt': p, 'source': 'project authored probe; no automatic label'} for p in ('chào bạn', 'hi', 'bạn là ai', 'bạn học được không')]
                self._commit('init')
            self.model.on_tick = self._tick
            self.process = psutil.Process(); self.process.cpu_percent(None)
            self.monitor = ResourceMonitor(interval=.5).__enter__()
            self.thread = threading.Thread(target=self._run, name='scarx-shared-core-idle', daemon=True)
            if background: self.thread.start()
        except BaseException:
            if self.monitor: self.monitor.__exit__(None,None,None)
            if self.db: self.db.close()
            self.lease.close(); raise

    def _pack(self):
        data = self.model.dumps()
        meta = {'format': 'scarx-generation1-session-v2', 'sha256': hashlib.sha256(data).hexdigest(),
                'dialogue_learning':self.dialogue.dumps(),
                'rng': self.rng.bit_generator.state, 'memories': self.memories,
                'probe': self.probe, 'replay_cursor': self.replay_cursor,
                'idle_enabled': self.idle_enabled, 'idle_hz': self.idle_hz,
                'idle_steps': self.idle_steps, 'sequence': self.sequence, 'active': self.active}
        stream = io.BytesIO()
        with zipfile.ZipFile(stream,'w') as archive:
            archive.writestr(zipfile.ZipInfo('model.zip'),data)
            archive.writestr(zipfile.ZipInfo('session.json'),json.dumps(meta,ensure_ascii=False))
        payload=stream.getvalue()
        if len(payload)>3_000_000:raise MemoryError('Session vượt3MBserialized; giữ phiên đã commit, chưa ghi snapshot không thể phục hồi.')
        return payload

    def _unpack(self, payload):
        if len(payload)>3_000_000: raise ValueError('Session quá lớn')
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            if len(archive.namelist())!=2 or set(archive.namelist())!={'model.zip','session.json'} or sum(i.file_size for i in archive.infolist())>3_000_000:
                raise ValueError('Session format')
            data=archive.read('model.zip'); meta=json.loads(archive.read('session.json'))
        if meta.get('format') not in ('scarx-generation1-session','scarx-generation1-session-v2') or hashlib.sha256(data).hexdigest()!=meta['sha256']: raise ValueError('Session checksum')
        dialogue=DialogueLearning.loads(meta.get('dialogue_learning'))
        model=UnifiedCore.loads(data)
        memories=meta['memories']
        if not isinstance(memories,list) or len(memories)>128 or any(not isinstance(m,dict) or not isinstance(m.get('prompt'),str) or not 1<=len(m['prompt'])<=2048 or not isinstance(m.get('source'),str) for m in memories): raise ValueError('Invalid memory')
        if type(meta['idle_enabled']) is not bool or type(meta['idle_hz']) is not int or not 1<=meta['idle_hz']<=30: raise ValueError('Invalid idle policy')
        for name in ('idle_steps','sequence','replay_cursor'):
            if type(meta[name]) is not int or meta[name]<0: raise ValueError('Invalid controller counter')
        probe=meta['probe']
        if probe is not None:
            if not isinstance(probe,dict) or not isinstance(probe.get('prompt'),str) or not 1<=len(probe['prompt'])<=2048 or type(probe.get('position')) is not int or not 0<=probe['position']<=8 or type(probe.get('previous')) is not int or not 0<=probe['previous']<len(model.vocabulary) or not isinstance(probe.get('words'),list) or len(probe['words'])>8 or any(t not in model.ids for t in probe['words']): raise ValueError('Invalid pending probe')
        rng=np.random.default_rng(); rng.bit_generator.state=meta['rng']
        if hasattr(self,'model'):
            # Restore the serving objects in place. A rejected micro-update
            # changes no learner identity; it restores full saved arrays/state.
            serving=self.model;serving_core=serving.core
            serving_core.__dict__.clear();serving_core.__dict__.update(model.core.__dict__)
            serving.__dict__.clear();serving.__dict__.update(model.__dict__);serving.core=serving_core
            self.model=serving
        else:self.model=model
        self.model.on_tick=self._tick; self.rng=rng;self.dialogue=dialogue
        self.memories=memories; self.probe=probe
        for name in ('idle_enabled','idle_hz','idle_steps','sequence','replay_cursor','active'):setattr(self,name,meta[name])

    def _persist(self):
        payload=self._pack()
        self.db.execute('INSERT OR REPLACE INTO runtime VALUES(1,?,?)',(hashlib.sha256(payload).hexdigest(),payload))

    def _commit(self, action, details=None):
        previous=self.active; identifier=uuid.uuid4().hex; self.active=identifier
        try:
            if self.db.execute('SELECT COUNT(*) FROM checkpoints').fetchone()[0]>=2000:raise ValueError('Đã đạt2000checkpoint; sao lưu/quản lý trước khi học tiếp.')
            payload=self._pack(); now=datetime.now(timezone.utc).isoformat()
            with self.db:
                self.db.execute('INSERT INTO checkpoints VALUES(?,?,?,?,?)',(identifier,now,action,hashlib.sha256(payload).hexdigest(),payload))
                self._persist()
                self.db.execute('INSERT INTO events(created,action,details) VALUES(?,?,?)',(now,action,json.dumps(details or {},ensure_ascii=False)))
        except BaseException:
            self.active=previous; raise
        return identifier

    @contextmanager
    def attention(self,mode='foreground'):
        with self.queue_lock:self.pending+=1
        try:
            with self.lock:
                if self.closed:raise RuntimeError('Session closed')
                previous=self.state;self.state=mode
                try:yield
                finally:self.state=previous if self.pending>1 else ('idle' if self.idle_enabled else 'paused')
        finally:
            with self.queue_lock:self.pending-=1

    def _tick(self, kind, source):
        c=self.model.core; self.sequence+=1
        self.trace.append({'id':self.sequence,'time':time.time(),'tick':self.model.ticks,
            'mode':self.state,'kind':kind,'source':str(source or 'recurrent dynamics; no external input')[:160],
            'spikes':[int(s) if i<self.model.active_count else 0 for i,s in enumerate(c.spikes)],'voltage':np.round(c.voltage,4).tolist(),
            'spiking':int(np.count_nonzero(c.spikes[:self.model.active_count])),'computed':self.model.active_count,'allocated':c.config.hidden,
            'updates':c.updates,'eligibility_norm':float(np.linalg.norm(c.elig_rec)),
            'weights_updated':False})

    def idle_once(self):
        # Caller holds session lock. Exactly one bounded quantum of SAME core.
        if self.dialogue.enabled:
            if any(l['status']=='queued' and l['remaining']>0 for l in self.dialogue.lessons):
                self._online_microstep('idle_consolidation');self.idle_steps+=1;return
            if self.idle_steps and self.idle_steps%max(40,self.idle_hz*3)==0:
                self._review_experience();self.idle_steps+=1;return
        self.model.select_capacity(48)
        self.idle_steps+=1
        if self.probe is None and self.memories and self.idle_steps%20==0:
            memory=self.memories[self.replay_cursor%len(self.memories)]; self.replay_cursor+=1
            self.probe={'prompt':memory['prompt'],'position':0,'previous':1,'words':[]}
            self.model.boundary()
        elif self.probe is not None:
            probe=self.probe
            p=self.model.word_step(probe['previous'],self.model.context(probe['prompt']),probe['position'],source=probe['prompt'])
            token=int(np.argmax(p)); probe['position']+=1
            if token!=0:probe['words'].append(self.model.vocabulary[token]); probe['previous']=token
            if token==0 or probe['position']>=8:
                self.sequence+=1
                self.trace.append({'id':self.sequence,'time':time.time(),'mode':'idle','kind':'probe_result',
                    'source':probe['prompt'],'generated':' '.join(probe['words']),'verified':False,
                    'weights_updated':False,'note':'Neural probe output; not a transcript of semantic thoughts or a learning label.'})
                self.probe=None
        else:self.model.tick(kind='idle')

    def _run(self):
        last_save=time.monotonic()
        while not self.stop_event.wait(1/max(1,self.idle_hz)):
            if self.pending or not self.lock.acquire(timeout=.002):continue
            try:
                if self.pending or not self.idle_enabled:continue
                if self.idle_steps % max(1,self.idle_hz)==0:self.monitor.check()
                self.state='idle'; self.idle_once()
                if time.monotonic()-last_save>=30:
                    with self.db:self._persist()
                    last_save=time.monotonic()
            except Exception as exc:
                self.error=f'{type(exc).__name__}: {exc}'; self.idle_enabled=False; self.state='paused'
            finally:self.lock.release()

    def _begin_workload(self,prompt='',pixels=None,budget=None,request_id=None):
        load=plan_workload(self.model,prompt,pixels,budget,request_id)
        before=self._pack();old_h=self.model.core.config.hidden
        try:
            self.monitor.check();self.model.select_capacity(load['requested_neurons'])
            if self.model.core.config.hidden>old_h:self._commit('capacity-grow',{'from':old_h,'to':self.model.core.config.hidden,'reason':load['allocation_reason'],'quality_gain_proven':False})
        except BaseException:self._unpack(before);raise
        self.workload={**load,'allocated_neurons':self.model.core.config.hidden}
        self.sequence+=1
        self.trace.append({'id':self.sequence,'time':time.time(),'mode':'foreground','kind':'capacity_change',
            'computed':self.model.active_count,'allocated':self.model.core.config.hidden,'grew_from':old_h,
            'weights_updated':False,'source':load['allocation_reason']})

    def chat(self, prompt, max_words=24, neuron_budget=None, request_id=None, session_id='web', record_experience=True, observation=None):
        if type(max_words) is not int or not 1<=max_words<=32:raise ValueError('Giới hạn sinh1–32token.')
        if not isinstance(session_id,str) or not 1<=len(session_id)<=120 or not session_id.isascii():raise ValueError('SessionID không hợp lệ.')
        with self.attention():
            self.monitor.check(); self.probe=None
            self._begin_workload(prompt,budget=neuron_budget,request_id=request_id)
            correction=self.dialogue.correction(session_id,prompt) if self.dialogue.enabled and record_experience else None
            if correction:
                self.dialogue.enqueue(session_id,correction['prompt'],correction['target'],correction['evidence'])
                with self.db:self._persist()
                self._online_microstep('online_feedback')
                with self.db:self._persist()
                self.workload=finish_workload(self.model,self.workload,self.process.memory_info().rss)
                return {'text':'[Học từ lời sửa] Đã ghi nhận đáp án bạn muốn. Tôi đang cập nhật cục bộ và sẽ kiểm tra lại khi rảnh.',
                    'tokens':[],'stopped':True,'model':MODEL,'checkpoint':self.active,'source':'learning controller acknowledgement, not neural language generation',
                    'quality_gate_passed':False,'workload':self.workload}
            began=time.perf_counter();ticks_before=self.model.ticks
            result=self.model.generate(prompt,max_words)
            question=self.dialogue.observe(session_id,prompt,result) if self.dialogue.enabled and record_experience else None
            self.workload=finish_workload(self.model,self.workload,self.process.memory_info().rss)
            if record_experience:
                self.memories.append({'prompt':prompt,'source':'user input; content not automatically verified'})
                self.memories=self.memories[-128:]
            # Persist the evolving recurrent state without duplicating model objects.
            measured={'prompt':prompt,'generated':result['text'],'checkpoint':self.active,
                'neural_ticks':self.model.ticks-ticks_before,'seconds':time.perf_counter()-began,'timer':'perf_counter',
                'active':self.model.active_count,'allocated':self.model.core.config.hidden,
                'input_scope':'Latest user text only','target':None,'qwen':None,
                **(observation or {})}
            with self.db:
                self._persist()
                self._observation_event('answer-trace',measured,200)
            if question:
                return {**result,'text':'[Hỏi để học] '+question['question'],'neural_draft':result['text'],
                    'source':'learning controller clarification; draft is unverified','learning_question':question,
                    'model':MODEL,'checkpoint':self.active,'quality_gate_passed':False,'workload':self.workload}
            return {**result,'model':MODEL,'checkpoint':self.active,'quality_gate_passed':False,'workload':self.workload}

    def vision(self, data):
        from .pilots import PilotLab
        pixels,digest=PilotLab.pixels(data.get('image'))
        with self.attention():
            self.monitor.check(); self.probe=None
            self._begin_workload(data.get('prompt',''),pixels,data.get('neuron_budget'),data.get('request_id'))
            result=self.model.image(pixels)
            self.workload=finish_workload(self.model,self.workload,self.process.memory_info().rss)
            with self.db:self._persist()
            return {**result,'image_sha256':digest,'model':MODEL,'checkpoint':self.active,'workload':self.workload,'presentation':'server-formatted digit classification from shared core'}

    def teach(self, data, _automatic=False):
        if data.get('confirmation') is not True:raise ValueError('Cần xác nhận lời dạy rõ ràng.')
        prompt=data.get('prompt'); answer=data.get('answer'); repeats=data.get('repeats',16)
        if not isinstance(answer,str) or type(repeats) is not int or not 1<=repeats<=32:raise ValueError('Câu trả lời dạngtext; repeats1–32.')
        self.model.context(prompt)
        with self.attention('learning' if _automatic else 'foreground'):
            self.monitor.check(); original=self._pack(); self.probe=None
            try:
                # The guard must run on EVERY teaching path. It was previously
                # enabled only for _automatic (idle) lessons, so explicit teaching
                # from the web form and Hermes replaced weights with no regression
                # check at all. Measured consequence: a lesson was ACCEPTED while
                # 'chào bạn' went from 'Chào bạn.' to 'Bạn có bạn.'
                # This does not loosen the guard; it stops bypassing it.
                guard=self._guard_pairs(prompt)
                guard_before=self._guard_scores(self.model,guard) if guard else []
                # Capture the "before" text on a CLONE, not on the live core.
                # Observation must not perturb what it observes: running
                # generate() on the live core advanced its continuous recurrent
                # state, and that alone was enough to make the same weights read
                # 'Tôi là scar-x là' instead of 'Tôi là scar-x.'. Measured with a
                # step-isolation test: train_pair alone produced the correct
                # answer 3/3 times, while generate()-then-train broke the read.
                # The clone is discarded, so the session keeps ADR0006 continuity.
                try:
                    probe=UnifiedCore.loads(self.model.dumps())
                    output_before=probe.generate(prompt)
                    del probe
                except Exception:
                    output_before=self.model.generate(prompt)
                began=time.monotonic()
                planned=plan_workload(self.model,prompt,budget=data.get('neuron_budget'))
                self.model.select_capacity(planned['requested_neurons'])
                additions=self.model.expand_vocabulary(answer)
                before={name:getattr(self.model.core,name).copy() for name in ('w_in','w_rec','w_out')}
                updates_before=self.model.core.updates;losses=[]
                for _ in range(repeats):
                    losses.append(float(self.model.train_pair(prompt,answer))); self.monitor.check()
                    if time.monotonic()-began>15:raise TimeoutError('Teaching15s budget')
                output_after=self.model.generate(prompt)
                delta={name:float(np.linalg.norm(getattr(self.model.core,name)-old)) for name,old in before.items()}
                changed={name:int(np.count_nonzero(getattr(self.model.core,name)!=old)) for name,old in before.items()}
                metrics={'prompt':prompt,'target':answer,'passes':repeats,'losses':losses,'loss_mean':float(np.mean(losses)),
                    'loss_definition':'Mean teacher-forced token negative log likelihood, natural log, including EOS; not accuracy',
                    'update_steps':self.model.core.updates-updates_before,'changed_synapses':changed,
                    'attempted_weight_delta':delta,'before':output_before['text'],'after_attempt':output_after['text'],
                    'guard_checked':bool(guard),'guard_probe_count':len(guard)}
                if guard:
                    after_scores=self._guard_scores(self.model,guard)
                    # Bit 1 = content_ok, bit 2 = complete_ok.
                    # Only a CONTENT regression blocks an update. Losing the final
                    # punctuation is reported separately and does not block, because
                    # a truncated form must stay fixable - otherwise the guard
                    # protects the defect and refuses every repair (round 18).
                    content_before=[bool(s&1) for s in guard_before]
                    content_after=[bool(s&1) for s in after_scores]
                    complete_before=[bool(s&2) for s in guard_before]
                    complete_after=[bool(s&2) for s in after_scores]
                    metrics['guard_lost']=sum(b and not a for b,a in zip(content_before,content_after))
                    metrics['guard_style_lost']=sum(
                        b and not a for b,a,keep in zip(complete_before,complete_after,content_after) if keep)
                    if metrics['guard_lost']:
                        # Record the rejection BEFORE rolling back so the passive
                        # observer can show why a lesson was refused. Without this
                        # the UI saw no learning event at all and a refused lesson
                        # was indistinguishable from one that never ran.
                        rejected=dict(metrics)
                        rejected.update(accepted=False,rolled_back=True,
                            weight_delta={'w_in':0.,'w_rec':0.,'w_out':0.},
                            reason='learning guard rejected: lost a previously correct probe')
                        try:
                            self.sequence+=1
                            self.trace.append({'id':self.sequence,'time':time.time(),
                                'kind':'learning_rejected','mode':'foreground',
                                'weights_updated':False,'details':rejected,'source':prompt})
                        except Exception:pass
                        raise LearningGuardRejected(rejected)
                metrics.update(accepted=True,rolled_back=False,weight_delta=delta)
                self.memories.append({'prompt':prompt,'source':'explicit user lesson, not independently verified','answer':answer})
                self.memories=self.memories[-128:]
                if not _automatic:
                    item=self.dialogue.enqueue(str(data.get('session_id','web')),prompt,answer,'explicit teaching form')
                    item['remaining']=0;item['status']='learned'
                else:
                    self.dialogue.accepted_updates+=1
                    item=next((l for l in reversed(self.dialogue.lessons) if normalized(l['prompt'])==normalized(prompt) and l['status']!='superseded'),None)
                    if item:
                        item.pop('guard_rejection_signature',None);item.pop('same_guard_rejections',None)
                    if item and normalized(output_after['text'])==normalized(answer):item['status']='learned';item['remaining']=0
                    elif item and item['remaining']==0:
                        item['status']='needs_help'
                        self.dialogue.ask(item['session'],item['prompt'],'bounded local updates did not converge; need guidance')
                checkpoint=self._commit('online-learn' if _automatic else 'teach',{**metrics,'repeats':repeats,'verified':False})
                self.sequence+=1;self.trace.append({'id':self.sequence,'time':time.time(),'kind':'learning_update','mode':'foreground','weights_updated':True,'delta':delta,'source':prompt})
                self.workload=finish_workload(self.model,planned,self.process.memory_info().rss)
                return {'before':output_before,'after':output_after,'weight_delta':delta,'learning_metrics':metrics,'new_vocabulary_tokens':additions,'workload':self.workload,'checkpoint':checkpoint,'verified':False,'note':'Explicit human teaching; factual correctness is not automatically established.'}
            except BaseException:
                # Keep the rejection record (if any) so the observer can explain a
                # refusal; only discard the in-flight working trace otherwise.
                rejected_events=[e for e in self.trace if e.get('kind')=='learning_rejected']
                self._unpack(original)
                self.trace.clear()
                self.trace.extend(rejected_events[-4:])
                raise

    def _guard_pairs(self,changing_prompt):
        """Candidate pairs a new update must not break.

        Coverage fix: the corpus list alone omitted prompts the product already
        answers correctly, so a lesson could erase a working reply unseen
        (measured: 'chào bạn' was absent while the product answered it).

        This method is deliberately PURE: it does not call generate() and does
        not touch recurrent state. Candidate product prompts are appended with
        their EXPECTED answer, and the caller's existing before/after scoring
        naturally ignores any candidate that was not already passing, because a
        pair only counts as lost when it passed before and fails after.
        So adding a candidate cannot reject more, and it cannot protect a wrong
        answer: a pair the model does not answer correctly is never "lost".

        Threshold and scoring are unchanged.
        """
        pairs=[]
        path=ROOT/'data/dialogue/generation1.json'
        if path.exists():
            corpus=json.loads(path.read_text(encoding='utf-8'))
            pairs=[(f['train'][0],f['answer']) for f in corpus['families']]
        pairs += [(l['prompt'],l['target']) for l in self.dialogue.lessons if l['status']=='learned'][-8:]
        seen={normalized(p) for p,_ in pairs}
        for prompt,expect in PRODUCT_GUARD_PROMPTS.items():
            if normalized(prompt) in seen:continue
            pairs.append((prompt,expect));seen.add(normalized(prompt))
        return [(p,t) for p,t in pairs if normalized(p)!=normalized(changing_prompt)][:16]

    @staticmethod
    def _pair_verdict(output,target):
        """Two independent criteria, per the user's decision on 2026-11-10.

        content_ok : every required word is present and the reply does not loop.
                     This is what must never regress.
        complete_ok: the reply actually finishes (ends on . ! ? or matches the
                     target exactly). Trailing punctuation and repetition are NOT
                     a finished answer.

        The guard protects content, because a truncated form must remain
        fixable: 'Hello' and 'Hello.' are the same content, so a lesson that turns
        'Hello' into 'Hello.' is not a regression and must be allowed. Repetition
        ('Chào bạn. chào bạn') and lost words ARE regressions and must be blocked.
        Previously a single normalized() comparison conflated the two, so the
        guard protected the defect and refused the fix (measured: all three
        remaining lessons blocked, docs/STATUS.md round 18).
        """
        from ..neural.language import tokenize
        raw=tokenize(output)
        # Detect repetition on normalized tokens: 'Chào bạn. chào bạn' splits with
        # punctuation attached, which hides the repeated block.
        words=[t for t in raw]
        norm=' '.join(words).strip(' .!?')
        tgt_norm=' '.join(tokenize(target)).strip(' .!?')
        if not norm:
            return {'content_ok':False,'complete_ok':False,'repetition':False}
        flat=[]
        for t in words:
            core=t.strip('.,!?;:')
            if core:
                flat.append(core.lower())
        looping=False
        n=len(flat)
        for size in range(1,max(1,n//2)+1):
            for start in range(n-2*size+1):
                if flat[start:start+size]==flat[start+size:start+2*size]:
                    looping=True;break
            if looping:break
        content_ok = (norm==tgt_norm) and not looping
        # A reply only counts as complete if it really ends on punctuation.
        # 'Hello' vs target 'Hello.' is content-ok but NOT complete.
        complete_ok = bool(output.strip()) and output.strip()[-1] in '.!?'
        return {'content_ok':content_ok,'complete_ok':complete_ok,'repetition':looping}

    @staticmethod
    def _guard_scores(model,pairs):
        """Verdicts per pair as a bitmask: 3=content+complete, 1=content only."""
        clone=UnifiedCore.loads(model.dumps());scores=[]
        for prompt,target in pairs:
            # Diagnostic-only canonical state. The live recurrent core is never
            # reset by runtime chat/idle/learning. No testset or retrieval here.
            clone.core.reset_state();clone.last_active[:]=clone.ticks
            clone.select_capacity(min(clone.core.config.hidden,plan_workload(clone,prompt)['requested_neurons']))
            v=Generation._pair_verdict(clone.generate(prompt)['text'],target)
            scores.append((1 if v['content_ok'] else 0)|(2 if v['complete_ok'] else 0))
        return scores

    def _online_microstep(self,source):
        lesson=next((l for l in self.dialogue.lessons if l['status']=='queued' and l['remaining']>0),None)
        if lesson is None:return None
        before=self._pack();identifier=lesson['id'];display_load=self.workload
        lesson['remaining']-=1;lesson['attempts']+=1
        try:
            result=self.teach({'prompt':lesson['prompt'],'answer':lesson['target'],'repeats':1,'confirmation':True},_automatic=True)
            current=next(l for l in self.dialogue.lessons if l['id']==identifier)
            record={**result['learning_metrics'],'accepted':True,'source':source,'lesson':identifier,'checkpoint':result['checkpoint']}
        except Exception as exc:
            self._unpack(before);current=next(l for l in self.dialogue.lessons if l['id']==identifier)
            current['remaining']=max(0,current['remaining']-1);current['attempts']+=1
            if current['remaining']==0:current['status']='needs_help'
            self.dialogue.rejected_updates+=1
            record={**getattr(exc,'metrics',{}),'accepted':False,'rolled_back':True,'source':source,'lesson':identifier,'reason':str(exc),
                'prompt':current['prompt'],'target':current['target'],'weight_delta':{'w_in':0.,'w_rec':0.,'w_out':0.}}
            if isinstance(exc,LearningGuardRejected):
                signature=hashlib.sha256(self.model.dumps()+current['prompt'].encode('utf-8')+b'\0'+current['target'].encode('utf-8')).hexdigest()
                streak=current.get('same_guard_rejections',0)+1 if current.get('guard_rejection_signature')==signature else 1
                current['guard_rejection_signature']=signature;current['same_guard_rejections']=streak
                record['identical_guard_rejections']=streak
                if streak>=2:
                    current['remaining']=0;current['status']='needs_help'
                    record['stopped_identical_proposal']=True
                    record['reason']+='; stopped repeating the identical rejected proposal, guard unchanged'
        if current['status']=='needs_help':
            self.dialogue.ask(current['session'],current['prompt'],'local learning did not pass the guard/converge within the bounded attempts; need guidance')
        self.workload=display_load
        self.sequence+=1;self.trace.append({'id':self.sequence,'time':time.time(),'kind':'online_learning','mode':source,
            'source':current['prompt'],'weights_updated':record['accepted'],'details':record})
        if not record['accepted']:
            try:
                with self.db:
                    self._persist()
                    self._observation_event('online-reject',record,800)
            except BaseException:self._unpack(before);raise
        return record

    def _observation_event(self,action,details,limit):
        self.db.execute('INSERT INTO events(created,action,details) VALUES(?,?,?)',
            (datetime.now(timezone.utc).isoformat(),action,json.dumps(details,ensure_ascii=False,allow_nan=False)))
        self.db.execute('DELETE FROM events WHERE action=? AND id NOT IN (SELECT id FROM events WHERE action=? ORDER BY id DESC LIMIT ?)',(action,action,limit))

    def _review_experience(self):
        before=self._pack()
        try:self._review_experience_inner()
        except BaseException:self._unpack(before);raise

    def _review_experience_inner(self):
        candidates=[('lesson',l) for l in self.dialogue.lessons if l['status'] in ('learned','needs_help')]
        candidates += [('episode',e) for e in self.dialogue.episodes if not e['reviewed']]
        if not candidates:return
        kind,item=candidates[self.dialogue.cursor%len(candidates)];self.dialogue.cursor+=1
        self.model.select_capacity(min(self.model.core.config.hidden,plan_workload(self.model,item['prompt'])['requested_neurons']))
        output=self.model.generate(item['prompt'])
        expected=item['target'] if kind=='lesson' else item['neural_output']
        mismatch=normalized(output['text'])!=normalized(expected)
        self.dialogue.review_count+=1
        if kind=='episode':item['reviewed']=True
        if mismatch and kind=='lesson' and item['status']=='learned' and item.get('replays',0)<3:
            item['replays']=item.get('replays',0)+1;item['status']='queued';item['remaining']=4
        elif mismatch:
            self.dialogue.ask(item['session'],item['prompt'],
                'idle neural prediction differs from an explicit target' if kind=='lesson' else 'idle output changed; both generated outputs are unverified, ask the user instead of choosing a self-label')
        self.sequence+=1;self.trace.append({'id':self.sequence,'time':time.time(),'kind':'idle_review','mode':'idle',
            'source':item['prompt'],'generated':output['text'],'mismatch':mismatch,'reference_kind':kind,
            'reference_is_verified_truth':kind=='lesson' and item.get('independently_verified',False),'weights_updated':False})
        with self.db:self._persist()

    def pending_question(self,session):
        if not isinstance(session,str) or not 1<=len(session)<=120 or not session.isascii():raise ValueError('SessionID không hợp lệ.')
        with self.lock:
            q=self.dialogue.pending(session) if self.dialogue.ask_enabled else None
            return {'question':copy.deepcopy(q),'learning_enabled':self.dialogue.enabled}

    def answer_question(self,data):
        session=data.get('session_id','web');question=data.get('question_id');answer=data.get('answer')
        with self.attention():
            before=self._pack()
            try:
                lesson=self.dialogue.feedback(session,question,answer)
                with self.db:self._persist()
            except BaseException:self._unpack(before);raise
            if lesson and self.dialogue.enabled:
                for _ in range(2):self._online_microstep('online_human_answer')
            with self.db:self._persist()
            return {'recorded':True,'learned_from_human_label':lesson is not None,
                'independently_verified':lesson.get('independently_verified',False) if lesson else False,
                'accepted_microsteps':self.dialogue.accepted_updates,'rejected_microsteps':self.dialogue.rejected_updates,
                'checkpoint':self.active,'note':'Queued bounded local learning and idle review; not a claim that retention is solved.'}

    def configure_learning(self,data):
        enabled=data.get('enabled',self.dialogue.enabled);ask=data.get('ask_enabled',self.dialogue.ask_enabled)
        if type(enabled) is not bool or type(ask) is not bool:raise ValueError('Learning controls require booleans')
        with self.attention():
            before=self._pack()
            try:
                self.dialogue.enabled=enabled;self.dialogue.ask_enabled=ask
                with self.db:self._persist()
            except BaseException:self._unpack(before);raise
        return {'learning_enabled':enabled,'proactive_questions':ask,'unverified_self_training':False}

    def configure(self, data):
        enabled=data.get('enabled',self.idle_enabled); hz=data.get('hz',self.idle_hz)
        if type(enabled) is not bool or type(hz) is not int or not 1<=hz<=30:raise ValueError('enabled boolean, hz1–30.')
        with self.attention():
            original=self._pack()
            try:
                self.idle_enabled=enabled;self.idle_hz=hz;self.error=None
                with self.db:self._persist()
            except BaseException:self._unpack(original);raise
        return {'enabled':enabled,'hz':hz,'idle_weight_learning':self.dialogue.enabled,'unverified_self_training':False}

    def checkpoint(self):
        with self.attention():
            self.monitor.check()
            return {'checkpoint':self._commit('manual')}

    def restore(self, identifier):
        with self.attention():
            self.monitor.check();original=self._pack()
            row=self.db.execute('SELECT sha,payload FROM checkpoints WHERE id=?',(identifier,)).fetchone()
            if not row or hashlib.sha256(row[1]).hexdigest()!=row[0]:raise ValueError('Checkpoint không hợp lệ.')
            try:
                old_sequence=self.sequence;self._unpack(row[1]);self.sequence=max(old_sequence,self.sequence)
                new=self._commit('restore',{'source':identifier});self.trace.clear()
                return {'checkpoint':new,'restored_from':identifier}
            except BaseException:self._unpack(original);raise

    def status(self, after=0):
        with self.lock:
            memory=self.process.memory_info()
            return {'product':'SCAR-X · Thế hệ đầu','model':MODEL,'cores_loaded':1,'hidden':self.model.core.config.hidden,
                'active_neurons':self.model.active_count,'max_neurons':self.model.max_neurons,'workload':self.workload,
                'synapses':self.model.core.w_in.size+self.model.core.w_rec.size+self.model.core.w_out.size,
                'state':self.state,'ticks':self.model.ticks,'updates':self.model.core.updates,
                'idle':{'enabled':self.idle_enabled,'hz':self.idle_hz,'quanta':self.idle_steps,'weight_learning':self.dialogue.enabled,'error':self.error},
                'learning':{'enabled':self.dialogue.enabled,'accepted_updates':self.dialogue.accepted_updates,'rejected_updates':self.dialogue.rejected_updates,
                    'reviews':self.dialogue.review_count,'queued_lessons':sum(l['status']=='queued' for l in self.dialogue.lessons),
                    'pending_questions':sum(q['status']=='open' for q in self.dialogue.questions),'unverified_self_training':False},
                'pending_messages':self.pending,'ram_bytes':memory.rss,'commit_bytes':getattr(memory,'private',memory.vms),
                'cpu_percent_one_core':self.process.cpu_percent(None),'resources':self.monitor.report(),
                'memory_limit':self.limit_status,'checkpoint':self.active,'memories':len(self.memories),
                'trace':[r for r in self.trace if r['id']>after], 'trace_cursor':self.sequence,
                'trace_capacity':256,'recurrent_indices':self.model.core.indices.tolist(),
                'recurrent_weights':np.round(self.model.core.w_rec,4).tolist(),'general_chat':False,'general_vision':False,
                'note':'Only selected neuron rows/readout columns are computed; all allocated weights remain resident. Idle dynamics and guarded labeled learning use the same core.'}

    def checkpoints(self):
        with self.lock:return [{'id':r[0],'created':r[1],'action':r[2]} for r in self.db.execute('SELECT id,created,action FROM checkpoints ORDER BY rowid DESC LIMIT 40')]

    def export(self):
        with self.attention():return self._pack()

    def close(self):
        self.stop_event.set()
        if getattr(self,'thread',None) and self.thread.is_alive():self.thread.join(timeout=5)
        with self.lock:
            if self.closed:return
            try:
                with self.db:self._persist()
            finally:
                self.closed=True;self.db.close();self.monitor.__exit__(None,None,None);self.lease.close()
