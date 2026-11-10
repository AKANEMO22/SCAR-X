"""Persistent human-in-the-loop local learning, not a semantic truth oracle.

Questions and acknowledgements are controller messages. Neural answers remain
the shared core's outputs. Idle review compares supplied targets or past neural
outputs; agreement with its own past output is NEVER evidence of correctness.
"""
import copy
import re
import time
import uuid
from ..neural.language import tokenize
from .verifiers import parse_integer_expression,integer_answer


KNOWN_PROBES={'hi','hello','chào','chào bạn','xin chào','bạn là ai','bạn tên gì','cảm ơn','tạm biệt','bạn học được không'}


def normalized(text):return ' '.join(tokenize(text)).strip(' .!?')


class DialogueLearning:
    def __init__(self):
        self.enabled=True;self.ask_enabled=True
        self.episodes=[];self.lessons=[];self.questions=[];self.cursor=0
        self.accepted_updates=0;self.rejected_updates=0;self.review_count=0
        self.last_review_tick=0

    def dumps(self):return copy.deepcopy(self.__dict__)

    @classmethod
    def loads(cls,state):
        value=cls()
        if state is None:return value
        if not isinstance(state,dict) or set(state)!=set(value.__dict__):raise ValueError('Invalid dialogue state fields')
        if type(state['enabled']) is not bool or type(state['ask_enabled']) is not bool:raise ValueError('Invalid learning policy')
        for name in ('cursor','accepted_updates','rejected_updates','review_count','last_review_tick'):
            if type(state[name]) is not int or state[name]<0:raise ValueError('Invalid learning clock')
        for name,limit in (('episodes',64),('lessons',64),('questions',16)):
            rows=state[name]
            if not isinstance(rows,list) or len(rows)>limit:raise ValueError('Unbounded learning state')
            for row in rows:
                if not isinstance(row,dict) or not isinstance(row.get('id'),str) or len(row['id'])>80 or not isinstance(row.get('session'),str) or len(row['session'])>120:raise ValueError('Invalid experience identity')
                if not isinstance(row.get('prompt'),str) or not 1<=len(row['prompt'])<=2048:raise ValueError('Invalid experience prompt')
                if name=='lessons' and (not isinstance(row.get('target'),str) or not 1<=len(row['target'])<=1024 or not 1<=len(tokenize(row['target']))<=24 or any(len(t)>64 for t in tokenize(row['target'])) or type(row.get('remaining')) is not int or not 0<=row['remaining']<=32):raise ValueError('Invalid lesson')
                if name=='episodes' and (not isinstance(row.get('neural_output'),str) or len(row['neural_output'])>8192 or type(row.get('reviewed')) is not bool):raise ValueError('Invalid episode')
                if name=='lessons':
                    if 'guard_rejection_signature' in row and (not isinstance(row['guard_rejection_signature'],str) or re.fullmatch(r'[0-9a-f]{64}',row['guard_rejection_signature']) is None):raise ValueError('Invalid guard signature')
                    if 'same_guard_rejections' in row and (type(row['same_guard_rejections']) is not int or not 1<=row['same_guard_rejections']<=32):raise ValueError('Invalid repeated guard count')
                    if row.get('status') not in ('queued','learned','needs_help','superseded') or type(row.get('attempts')) is not int or not 0<=row['attempts']<=2000 or not 0<=row.get('replays',0)<=3:raise ValueError('Invalid lesson progress')
                    if row.get('independently_verified'):
                        expr=parse_integer_expression(row['prompt'])
                        if expr is None or normalized(row['target'])!=str(integer_answer(expr)):raise ValueError('Checkpoint label lacks arithmetic proof')
                if name=='questions' and (not isinstance(row.get('question'),str) or len(row['question'])>3000 or row.get('status') not in ('open','answered','skipped')):raise ValueError('Invalid question')
        value.__dict__.update(copy.deepcopy(state));return value

    def pending(self,session):
        return next((q for q in reversed(self.questions) if q['session']==session and q['status']=='open'),None)

    def ask(self,session,prompt,reason,force=False,user_turn=False):
        if not self.ask_enabled:return None
        current=normalized(prompt)
        pending=next((q for q in reversed(self.questions) if q['session']==session and q['status']=='open'),None)
        if pending:
            if normalized(pending['prompt'])==current:return pending
            # A question about a DIFFERENT prompt must never answer this turn.
            # ADR0009 allows at most one open question per session, so retire the
            # stale one instead of stacking a second.
            pending['status']='skipped'
            pending['superseded_by_prompt']=prompt[:240]
        blocked=next((l for l in reversed(self.lessons) if l['status']=='needs_help' and normalized(l['prompt'])==current),None)
        skipped=any(q['session']==session and normalized(q['prompt'])==current and q['status']=='skipped' for q in self.questions)
        # ADR0009: a skipped item is not re-raised merely because a background
        # review is running (no nagging), but a NEW explicit user turn may ask
        # again. `force` alone must not resurrect a skipped prompt.
        if skipped and not user_turn:return None
        if blocked is not None and not force:return None
        if len(self.questions)>=16:
            removable=next((i for i,q in enumerate(self.questions) if q['status']!='open'),None)
            if removable is None:return None
            self.questions.pop(removable)
        text=(f'SCAR-X đã lưu lời dạy cho «{prompt[:240]}» nhưng chưa học ổn định trong guard. Nếu bạn muốn sửa nhãn mong muốn, gửi đáp án ngắn; nếu không, chọn “bỏ qua” để tiếp tục trò chuyện.' if blocked else
              f'SCAR-X cần bạn hướng dẫn cho: «{prompt[:240]}». Bạn muốn dạy câu trả lời nào, hoặc có bằng chứng nào để kiểm tra? Nếu chưa biết, trả lời “bỏ qua”.')
        question={'id':uuid.uuid4().hex,'session':session,'prompt':prompt,'reason':reason,'status':'open',
            'question':text,
            'source':'learning controller; not a neural chain-of-thought transcript'}
        self.questions.append(question);return question

    def observe(self,session,prompt,result):
        row={'id':uuid.uuid4().hex,'session':session,'prompt':prompt,'neural_output':result['text'],'reviewed':False}
        self.episodes.append(row);self.episodes=self.episodes[-64:]
        target=next((l['target'] for l in reversed(self.lessons) if normalized(l['prompt'])==normalized(prompt)),None)
        words=result.get('tokens',[])
        bad=not result['text'].strip() or not result.get('stopped',False) or any(words.count(w)>3 for w in set(words))
        if target is not None and normalized(target)!=normalized(result['text']):bad=True
        if normalized(prompt) not in KNOWN_PROBES and target is None:bad=True
        if bad:return self.ask(session,prompt,'unsupported coverage, unstable output or mismatch with an explicit lesson; not a truth verdict',force=True,user_turn=True)
        return None

    def correction(self,session,text):
        match=re.fullmatch(r'\s*không\s+được\s+(.+?)\s*[,;]?\s*phải\s+là\s+(.+?)\s*',text,re.I)
        if not match:return None
        previous=next((e for e in reversed(self.episodes) if e['session']==session),None)
        if previous is None or normalized(match[1]) not in normalized(previous['neural_output']):return None
        return {'prompt':previous['prompt'],'target':match[2].strip(),'evidence':text}

    def enqueue(self,session,prompt,target,evidence,kind='explicit_response_correction'):
        if not isinstance(target,str) or not 1<=len(target)<=1024 or not 1<=len(tokenize(target))<=24 or any(len(t)>64 for t in tokenize(target)):raise ValueError('Lời dạy cần1–24token ngắn, tối đa1024ký tự; không gửi lệnh.')
        # Reject a provably false numeric label rather than treating a checkbox
        # or explicit user correction as proof of a mathematical fact.
        arithmetic=parse_integer_expression(prompt);independent=False
        plain=normalized(target)
        if arithmetic and re.fullmatch(r'-?\d+',plain):
            expected=integer_answer(arithmetic)
            if int(plain)!=expected:raise ValueError('Đáp án mâu thuẫn với phép tính kiểm chứng được; cần làm rõ, chưa cập nhật weights.')
            kind='integer_arithmetic_checked';independent=True
        item={'id':uuid.uuid4().hex,'session':session,'prompt':prompt,'target':target,'evidence':str(evidence)[:2048],
            'kind':kind if independent else 'explicit_response_correction','independently_verified':independent,'remaining':32,'attempts':0,'status':'queued'}
        # A revised desired response supersedes the previous response for this
        # prompt; retain its provenance, not conflicting targets in replay.
        for old in self.lessons:
            if normalized(old['prompt'])==normalized(prompt):old['status']='superseded';old['remaining']=0
        self.lessons.append(item);self.lessons=self.lessons[-64:]
        return item

    def feedback(self,session,question_id,answer):
        q=self.pending(session)
        if not q or q['id']!=question_id:raise ValueError('Câu hỏi không còn chờ phản hồi trong phiên này.')
        if not isinstance(answer,str):raise ValueError('Phản hồi cần dạngtext.')
        if normalized(answer) in ('bỏ qua','không biết','skip','chưa biết'):
            q['status']='skipped';return None
        lesson=self.enqueue(session,q['prompt'],answer,'answer to controller question '+q['id'])
        q['status']='answered';return lesson
