"""Bounded measurements from registered runners, consumed by the passive UI."""
import json, sqlite3, time
from pathlib import Path
from .experiment_status import RUN_ID


def append(run, kind, data):
    run=Path(run).resolve()
    if run.parent.name!='runs' or not RUN_ID.fullmatch(run.name) or kind not in ('answer','learning','activity'):
        raise ValueError('Expected registered run and measured event')
    record={**data,'time':time.time(),'kind':kind,'run_id':run.name}
    payload=json.dumps(record,ensure_ascii=False,allow_nan=False)
    if len(payload.encode())>32768:raise ValueError('Trace event exceeds32KB')
    path=run.parent.parent/'local/work-trace.sqlite';path.parent.mkdir(parents=True,exist_ok=True)
    with sqlite3.connect(path,timeout=5) as db:
        db.execute('CREATE TABLE IF NOT EXISTS trace(id INTEGER PRIMARY KEY,kind TEXT,payload TEXT)')
        db.execute('INSERT INTO trace(kind,payload) VALUES(?,?)',(kind,payload))
        db.execute('DELETE FROM trace WHERE id NOT IN (SELECT id FROM trace ORDER BY id DESC LIMIT 600)')


def activity(run, title, detail, *, status='completed', evidence=None):
    """Technical work log, explicitly separate from neural measurements/thoughts."""
    if status not in ('started','completed','failed') or not isinstance(title,str) or not title.strip() or len(title)>200:
        raise ValueError('Expected a bounded technical activity and status')
    if not isinstance(detail,str) or len(detail)>3000:
        raise ValueError('Activity detail exceeds limit')
    append(run,'activity',{'title':title,'detail':detail,'status':status,'evidence':evidence,
        'scope':'Nhật ký công việc kỹ thuật; không phải suy nghĩ neuron'})


def read(root):
    path=Path(root)/'local/work-trace.sqlite'
    if not path.exists():return []
    with sqlite3.connect(path.as_uri()+'?mode=ro',uri=True,timeout=2) as db:
        return [json.loads(row[0]) for row in db.execute('SELECT payload FROM trace ORDER BY id DESC LIMIT 200')]
