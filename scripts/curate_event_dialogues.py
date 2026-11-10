"""Explicit versioned editorial repair of teacher data, not a learned lookup."""
from pathlib import Path
from datetime import datetime,timezone
import copy,json,hashlib,sys
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
from scarx.neural.language import tokenize
if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
source=root/'data/candidates/antigravity-events-repair-20261109T093620Z'
raw=(source/'dialogue.json').read_bytes();original=json.loads(raw);value=copy.deepcopy(original)
by_id={d['id']:d for d in value['dialogs']};edits=[]


def edit(identifier,index,text,reason):
    before=by_id[identifier]['turns'][index]['text'];by_id[identifier]['turns'][index]['text']=text
    edits.append({'dialogue_id':identifier,'turn':index,'before':before,'after':text,'reason':reason})


edit('b1-03',3,'Bạn gửi nội dung cần giải thích nhé.','Request missing source text instead of promising an explanation without seeing it')
edit('b2-dialog-03',3,'Bạn gửi nội dung tài liệu nhé.','Request the missing report, no assumed access to yesterday file')
edit('b2-dialog-08',1,'Yes.','Text-only acknowledgement; do not train unsupported auditory perception claim')
edit('b3-d07',1,'Thank you. Please give the time.','No invented admission about a prior answer absent from the dialogue')
edit('b1-07',5,'You are welcome.','Common short response vocabulary; same conversational intent')
edit('b1-10',3,'Please review the list.','Short direct instruction without extra unique vocabulary')
edit('b4-d10',2,'So it has three sides?','Original repair made user disagree with a correct answer; use a coherent follow-up')
edit('b4-d10',3,'That is correct.','Confirm the grounded geometry statement concisely')
replacement=[('user','Tôi chọn màu đỏ.'),('assistant','Tôi đã ghi nhận.'),('user','Đổi thành màu vàng.'),
 ('assistant','Màu bạn chọn là vàng.'),('user','Tôi đang chọn màu nào?'),('assistant','Màu vàng.')]
for i,(_,text) in enumerate(replacement):edit('b4-d04',i,text,'Replace ambiguous pigment/light mixing correction with an explicit in-dialogue preference update')
tokens={t for d in value['dialogs'] for turn in d['turns'][1::2] for t in tokenize(turn['text'])}
if len(tokens)>200:raise ValueError('Curated corpus still exceeds fixed200target vocabulary; do not relax threshold')
stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');target=root/'data/candidates'/('antigravity-events-curated-'+stamp);target.mkdir(parents=True)
path=target/'dialogue.json';path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
manifest={'source':'synthetic:antigravity + explicit Codex editorial curation','model':'gemini-3.8-flash-low','created_utc':stamp,
 'parent_directory':source.relative_to(root).as_posix(),'parent_sha256':hashlib.sha256(raw).hexdigest(),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
 'content_review':{'reviewer':'Codex','scope':'Read every six-turn public dialogue; no independent factual-truth certification','edits':edits,
   'remaining_limits':['exact-match text style has multiple valid answers','template IDs do not rule out semantic near-duplicates','neural memory/unlimited recall not established','no live auto-import']},
 'target_vocabulary_tokens':len(tokens),'data_contract':'data/contracts/ANTIGRAVITY_DIALOGUE_EVENTS_V1.md','trained':False,'independently_verified':False}
(target/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(root/'artifacts/antigravity-events-latest.json').write_text(json.dumps({'candidate_directory':target.relative_to(root).as_posix(),'validation_pending':True,'trained':False},indent=2)+'\n',encoding='utf-8')
print(json.dumps({'candidate':str(target),'target_vocabulary':len(tokens),'editorial_edits':len(edits)}))
