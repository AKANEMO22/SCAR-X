"""Explicit project-fixture supervision, never a response lookup at inference."""
import hashlib,json,re
from pathlib import Path
from ..learning.dialogue import normalized


def example(data):
    family=data.get('family');index=data.get('example',0)
    if type(family) is not int or not 0<=family<8 or type(index) is not int or not 0<=index<8:raise ValueError('Invalid project example index')
    path=Path(__file__).resolve().parents[3]/'data/dialogue/generation1.json';raw=path.read_bytes()
    entry=json.loads(raw)['families'][family]
    return entry['train'][index],entry['answer'],hashlib.sha256(raw).hexdigest()


def post(app,data):
    from .learning_channel import append,feed
    identifier=data.get('client_id')
    if not isinstance(identifier,str) or re.fullmatch(r'[A-Za-z0-9_-]{8,80}',identifier) is None:raise ValueError('Invalid client ID')
    prompt,target,sha=example(data);message_id='project:'+identifier
    fingerprint=hashlib.sha256(json.dumps({'prompt':prompt,'target':target,'source_sha256':sha},sort_keys=True).encode()).hexdigest()
    with app.attention('learning'):
        previous=app.db.execute('SELECT payload FROM learning_messages WHERE id=?',(message_id,)).fetchone()
        if previous:
            if json.loads(previous[0]).get('request_fingerprint')!=fingerprint:raise ValueError('Client ID already used for another project example')
            return {'deduplicated':True,'feed':feed(app)}
        if not app.dialogue.enabled:raise ValueError('Learning is paused')
        override=next((r for r in reversed(app.dialogue.lessons) if normalized(r['prompt'])==normalized(prompt) and r['status']!='superseded'),None)
        if override and normalized(override['target'])!=normalized(target):
            return {'skipped':True,'reason':'Existing personal response override; project example not queued','weights_updated':False}
        checkpoint=app.checkpoint()['checkpoint'];before=app._pack()
        source='project-authored basic dialogue fixture under user-authorized teaching; not a human message or independently verified fact'
        try:
            lesson=app.dialogue.enqueue('project-supervision',prompt,target,source+'; sha256='+sha)
            with app.db:
                append(app,message_id,'supervisor',prompt+' → '+target,kind='project_lesson',source=source,
                    request_fingerprint=fingerprint,lesson_id=lesson['id'],source_sha256=sha)
                app._persist()
        except BaseException:app._unpack(before);raise
        records=[app._online_microstep('project_fixture_supervision') for _ in range(2)]
        return {'deduplicated':False,'checkpoint_before':checkpoint,'lesson_id':lesson['id'],'source':source,
            'source_sha256':sha,'updates':[r for r in records if r],'feed':feed(app)}
