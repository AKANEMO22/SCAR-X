"""Opt-in trained language/vision labs; separate from V0.1's active model.

Language quality gate failed: raw generation is exposed only as an explicitly
experimental model. Vision is digits-only, not general screenshot understanding.
"""
import base64
import copy
from datetime import datetime,timezone
import hashlib
import io
import json
from pathlib import Path
import time
import uuid
import numpy as np
from ..neural.core import SpikingCore
from ..neural.language import WordLanguageCore,tokenize

LANGUAGE='scarx-language-lab-v0.2'
VISION='scarx-digits-lab-v0.2'


class PilotLab:
    def __init__(self,app):
        self.app=app;self.db=app.db;self.root=Path(__file__).resolve().parents[3]
        self.loaded={}
        self.db.executescript('''
          CREATE TABLE IF NOT EXISTS pilot_checkpoints(id TEXT PRIMARY KEY,kind TEXT NOT NULL,created TEXT NOT NULL,
            sha256 TEXT NOT NULL,payload BLOB NOT NULL,details TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS pilot_active(kind TEXT PRIMARY KEY,id TEXT NOT NULL);
        ''')

    def available(self):
        source=self.root/'artifacts/pilot-release.json'
        return json.loads(source.read_text(encoding='utf-8')) if source.exists() else {}

    def _load(self,kind):
        if kind in self.loaded:return self.loaded[kind]
        row=self.db.execute('SELECT p.sha256,p.payload FROM pilot_active a JOIN pilot_checkpoints p ON p.id=a.id WHERE a.kind=?',(kind,)).fetchone()
        if row:
            if hashlib.sha256(row[1]).hexdigest()!=row[0]:raise ValueError('Pilot checkpoint checksum sai; không tự reset.')
            model=WordLanguageCore.loads(row[1]) if kind=='language' else SpikingCore.loads(row[1])
        else:
            release=self.available()
            if kind not in release:raise ValueError('Checkpoint pilot chưa có trên máy.')
            path=(self.root/release[kind]['checkpoint']).resolve()
            if not path.is_relative_to(self.root):raise ValueError('Checkpoint path ngoài project')
            model=WordLanguageCore.load(path) if kind=='language' else SpikingCore.load(path)
            with self.db:self._commit(kind,model,{'action':'load-trained-pilot','source':release[kind]})
        if kind=='vision' and (model.config.inputs,model.config.outputs,model.config.hidden)!=(64,10,48):raise ValueError('Sai digit core')
        self.loaded[kind]=model
        return model

    def _commit(self,kind,model,details):
        if self.db.execute('SELECT count(*) FROM pilot_checkpoints').fetchone()[0]>=2000:raise ValueError('Giới hạn2000pilotcheckpoint; không tự xóa bản cũ.')
        payload=model.dumps()
        if len(payload)>8_000_000:raise MemoryError('Pilot snapshot quá lớn')
        identifier=uuid.uuid4().hex
        self.db.execute('INSERT INTO pilot_checkpoints VALUES(?,?,?,?,?,?)',(identifier,kind,datetime.now(timezone.utc).isoformat(),hashlib.sha256(payload).hexdigest(),payload,json.dumps(details,ensure_ascii=False)))
        self.db.execute('INSERT OR REPLACE INTO pilot_active VALUES(?,?)',(kind,identifier))
        return identifier

    def status(self):
        with self.app.lock:
            active={r[0]:r[1] for r in self.db.execute('SELECT kind,id FROM pilot_active')}
            return {'release':self.available(),'active':active,'language_model':LANGUAGE,'vision_model':VISION,
                    'warning':'Ngôn ngữ pilot chưa đạt gate; chỉ44từ ban đầu, có thể trộn/lặp câu. Vision chỉ chữ số viết tay, không đọc giao diện.'}

    def language(self,prompt):
        if not isinstance(prompt,str) or not prompt.strip() or len(prompt)>2048:raise ValueError('Nhập câu từ1–2048ký tự.')
        with self.app.lock:
            model=self._load('language'); result=WordLanguageCore.loads(model.dumps()).generate(prompt)
            result.update({'experimental':True,'quality_gate_passed':False,'vocabulary_size':len(model.vocabulary),
                           'checkpoint':self.db.execute('SELECT id FROM pilot_active WHERE kind=?',('language',)).fetchone()[0]})
            with self.db:self.app._event('language-predict',{'prompt':prompt,'generated':result['text'],'quality_gate_passed':False})
            return result

    def teach_language(self,data):
        prompt=data.get('prompt');answer=data.get('answer');repeats=data.get('repeats',16)
        if data.get('confirmation') is not True:raise ValueError('Cần chủ động xác nhận ví dụ muốn dạy; đây không phải bằng chứng đúng khách quan.')
        if not isinstance(prompt,str) or not 1<=len(prompt)<=2048 or not isinstance(answer,str) or not 1<=len(tokenize(answer))<=24:raise ValueError('Câu hỏi1–2048ký tự; câu trả lời1–24token.')
        if type(repeats) is not int or not 1<=repeats<=32:raise ValueError('Số vòng dạy từ1–32.')
        with self.app.lock:
            self.app.monitor.check();original=self._load('language');candidate=WordLanguageCore.loads(original.dumps())
            before=candidate.generate(prompt);added=candidate.expand_vocabulary(answer)
            fingerprint=candidate.core.fingerprint();start=time.monotonic()
            for _ in range(repeats):
                candidate.train_pair(prompt,answer,mode='full');self.app.monitor.check()
                if time.monotonic()-start>15:raise TimeoutError('Đợt dạy vượt15s; state cũ giữ nguyên.')
            after=candidate.generate(prompt)
            result={'before':before,'after':after,'new_vocabulary_tokens':added,'weights_changed':candidate.core.fingerprint()!=fingerprint,
                    'updates':candidate.core.updates-original.core.updates,'note':'Ví dụ do người dùng cung cấp, chưa kiểm chứng nội dung; không tự dùng làm factual truth/replay.'}
            with self.db:
                self.app.store.observe(json.dumps({'prompt':prompt,'answer':answer,'evidence':'explicit-human-teaching-not-independently-verified'},ensure_ascii=False),'scarx-language-lab:user',kind='correction',verified=False,commit=False)
                result['checkpoint']=self._commit('language',candidate,{'action':'teach','prompt':prompt,'result':result})
                self.app._event('language-teach',result)
            self.loaded['language']=candidate
            return result

    @staticmethod
    def pixels(data_uri):
        from PIL import Image,ImageOps
        if not isinstance(data_uri,str) or not data_uri.startswith(('data:image/png;base64,','data:image/jpeg;base64,')) or len(data_uri)>700_000:raise ValueError('Chỉ nhận ảnhPNG/JPEG dataURI local≤500KB.')
        payload=base64.b64decode(data_uri.split(',',1)[1],validate=True)
        if len(payload)>500_000:raise ValueError('Ảnh quá lớn')
        try:image=Image.open(io.BytesIO(payload))
        except OSError as exc:raise ValueError('Không đọc được ảnhPNG/JPEG.') from exc
        with image:
            if image.format not in ('PNG','JPEG'):raise ValueError('Định dạng ảnh không được hỗ trợ.')
            if image.width*image.height>1_048_576 or min(image.size)<8:raise ValueError('Ảnh tối thiểu8×8, tối đa1Mpixel.')
            if max(image.size)/min(image.size)>3:raise ValueError('Ảnh ngoài phạm vi chữ số; chưa có model đọc screenshot/giao diện.')
            gray=image.convert('L'); small=np.asarray(gray.resize((64,64)),np.float32)
            border=np.concatenate((small[0],small[-1],small[:,0],small[:,-1]))
            if float(np.median(border))>127:gray=ImageOps.invert(gray);small=255-small
            occupancy=float(np.mean(small>96))
            if not .04<occupancy<.65:raise ValueError('Ảnh không giống miền chữ số đã học; chưa có khả năng mô tả ảnh tự do.')
            pixels=np.asarray(gray.resize((8,8)),np.float32)/255*16
        return pixels,hashlib.sha256(payload).hexdigest()

    def vision(self,data):
        pixels,digest=self.pixels(data.get('image'))
        sequence=np.repeat((pixels.reshape(-1)/16-.3)[None,:],4,axis=0)
        with self.app.lock:
            model=self._load('vision');p=SpikingCore.loads(model.dumps()).forward(sequence)
            return {'digit':int(np.argmax(p)),'probabilities':p.tolist(),'pixels':pixels.tolist(),'image_sha256':digest,
                    'source':'Trained48-ALIF neural digit classifier; presentation is formatted by server, not language generation.',
                    'scope':'digits-only; not screenshot/OCR/visualQA','out_of_domain_check':'Fixed occupancy/aspect heuristic, not a validated neural OOD detector.',
                    'checkpoint':self.db.execute('SELECT id FROM pilot_active WHERE kind=?',('vision',)).fetchone()[0]}

    def checkpoints(self):
        with self.app.lock:
            return [{'id':r[0],'kind':r[1],'created':r[2],'details':json.loads(r[3])} for r in self.db.execute('SELECT id,kind,created,details FROM pilot_checkpoints ORDER BY rowid DESC LIMIT 20')]

    def examples(self):
        source=self.root/'artifacts/pilots/vision-examples.json'
        return json.loads(source.read_text(encoding='utf-8')) if source.exists() else []

    def restore(self,identifier):
        with self.app.lock:
            row=self.db.execute('SELECT kind,sha256,payload FROM pilot_checkpoints WHERE id=?',(identifier,)).fetchone()
            if row is None or hashlib.sha256(row[2]).hexdigest()!=row[1]:raise ValueError('Pilotcheckpoint không hợp lệ')
            model=WordLanguageCore.loads(row[2]) if row[0]=='language' else SpikingCore.loads(row[2])
            with self.db:new=self._commit(row[0],model,{'action':'restore','source':identifier})
            self.loaded[row[0]]=model
            return {'kind':row[0],'checkpoint':new,'restored_from':identifier}
