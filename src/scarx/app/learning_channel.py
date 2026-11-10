"""Bounded durable correspondence; transcript is not neural knowledge/retrieval."""
from datetime import datetime,timezone
import hashlib,json,re

LIMIT=512


def initialize(app):
    app.db.execute('CREATE TABLE IF NOT EXISTS learning_messages(id TEXT PRIMARY KEY,created TEXT NOT NULL,payload TEXT NOT NULL)')


def append(app,identifier,role,text,**fields):
    payload={'id':identifier,'role':role,'text':text,**fields}
    app.db.execute('INSERT OR IGNORE INTO learning_messages VALUES(?,?,?)',
      (identifier,datetime.now(timezone.utc).isoformat(),json.dumps(payload,ensure_ascii=False)))


def synchronize(app):
    for q in app.dialogue.questions:
        append(app,'q:'+q['id'],'assistant',q['question'],question_id=q['id'],session_id=q['session'],
          prompt=q['prompt'],source='learning controller',kind='question')
    # Keep pending questions visible while bounding communication history.
    protected=['q:'+q['id'] for q in app.dialogue.questions if q['status']=='open']
    excess=app.db.execute('SELECT COUNT(*) FROM learning_messages').fetchone()[0]-LIMIT
    if excess>0:
        candidates=app.db.execute('SELECT id FROM learning_messages ORDER BY rowid').fetchall()
        ids=[(r[0],) for r in candidates if r[0] not in protected][:excess]
        app.db.executemany('DELETE FROM learning_messages WHERE id=?',ids)


def feed(app):
    with app.lock:
        with app.db:synchronize(app)
        questions={q['id']:q for q in app.dialogue.questions};lessons={l['id']:l for l in app.dialogue.lessons}
        recent=app.db.execute('SELECT rowid,created,payload FROM learning_messages ORDER BY rowid DESC LIMIT 200').fetchall()
        indexed={r[0]:r for r in recent}
        for q in questions.values():
            if q['status']=='open':
                row=app.db.execute('SELECT rowid,created,payload FROM learning_messages WHERE id=?',('q:'+q['id'],)).fetchone()
                if row:indexed[row[0]]=row
        messages=[]
        for _,created,raw in sorted(indexed.values()):
            item=json.loads(raw);item.pop('request_fingerprint',None);item['created']=created
            if item.get('question_id'):
                q=questions.get(item['question_id']);item['question_status']=q['status'] if q else 'archived'
            if item.get('lesson_id'):
                lesson=lessons.get(item['lesson_id']);item['lesson_status']=lesson['status'] if lesson else 'outside_current_snapshot'
                item['independently_verified']=bool(lesson and lesson.get('independently_verified'))
            messages.append(item)
        return {'model':'scarx-generation1','messages':messages,'history_limit':LIMIT,
          'pending_questions':sum(q['status']=='open' for q in app.dialogue.questions),
          'learning':{'enabled':app.dialogue.enabled,'accepted_updates':app.dialogue.accepted_updates,
            'rejected_updates':app.dialogue.rejected_updates,'queued_lessons':sum(l['status']=='queued' for l in app.dialogue.lessons)},
          'source':'same shared neural core; questions and acknowledgements are controller messages'}


def post(app,data):
    if not isinstance(data,dict):raise ValueError('Cần JSON object.')
    identifier=data.get('client_id');kind=data.get('kind','chat')
    if not isinstance(identifier,str) or re.fullmatch(r'[A-Za-z0-9_-]{8,80}',identifier) is None:raise ValueError('client_id không hợp lệ.')
    if kind=='project_lesson':
        from .project_supervision import post as post_project
        return post_project(app,data)
    if kind not in ('chat','lesson','reply'):raise ValueError('Loại tin nhắn không hợp lệ.')
    text=data.get('text','');prompt=data.get('prompt','');answer=data.get('answer',text)
    if not isinstance(text,str) or len(text)>2048:raise ValueError('Tin nhắn tối đa2048ký tự.')
    body={k:data.get(k) for k in ('kind','text','prompt','answer','question_id','session_id')}
    fingerprint=hashlib.sha256(json.dumps(body,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    message_id='u:'+identifier
    with app.attention('learning'):
        existing=app.db.execute('SELECT payload FROM learning_messages WHERE id=?',(message_id,)).fetchone()
        if existing:
            if json.loads(existing[0]).get('request_fingerprint')!=fingerprint:raise ValueError('Mã tin nhắn đã dùng cho nội dung khác.')
            return {'deduplicated':True,'feed':feed(app)}
        if kind=='chat':
            app.model.context(text)
            with app.db:append(app,message_id,'user',text,kind=kind,source='human message',request_fingerprint=fingerprint)
            result=app.chat(text,session_id='hermes-learning')
            if not result.get('learning_question'):
                with app.db:append(app,'a:'+identifier,'assistant',result['text'],kind='chat',source=result.get('source','neural output; unverified'))
        else:
            from ..learning.dialogue import normalized
            skipping=kind=='reply' and isinstance(answer,str) and normalized(answer) in ('bỏ qua','không biết','skip','chưa biết')
            if not app.dialogue.enabled and not skipping:raise ValueError('Học đang tạm dừng. Bật học trước khi gửi lời dạy.')
            before=app._pack()
            try:
                if kind=='reply':
                    session=data.get('session_id');qid=data.get('question_id')
                    if not isinstance(session,str) or not isinstance(qid,str):raise ValueError('Thiếu session/câu hỏi cần trả lời.')
                    lesson=app.dialogue.feedback(session,qid,answer)
                    label='Trả lời: '+answer;fields={'question_id':qid,'session_id':session}
                else:
                    app.model.context(prompt)
                    lesson=app.dialogue.enqueue('hermes-learning',prompt,answer,'explicit lesson from learning tab')
                    label=prompt+'\n→ '+answer;fields={'prompt':prompt,'session_id':'hermes-learning'}
                if lesson:fields['lesson_id']=lesson['id']
                with app.db:
                    append(app,message_id,'user',label,kind=kind,source='human feedback; not automatically factual truth',request_fingerprint=fingerprint,**fields)
                    app._persist()
            except BaseException:
                app._unpack(before);raise
            if lesson:
                for _ in range(2):app._online_microstep('learning_tab_feedback')
            with app.db:
                append(app,'a:'+identifier,'assistant','Đã ghi nhận lời dạy; các cập nhật được kiểm tra trước khi giữ lại.' if lesson else 'Đã bỏ qua câu hỏi, không tạo nhãn học.',
                  kind='acknowledgement',source='learning controller',**({'lesson_id':lesson['id']} if lesson else {}))
        return {'deduplicated':False,'feed':feed(app)}
