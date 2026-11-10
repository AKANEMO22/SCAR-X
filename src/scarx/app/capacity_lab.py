"""Real bounded allocation/compute demonstration on a checkpoint clone."""
import threading,time,hashlib,json,os,tempfile
from pathlib import Path
import numpy as np
from ..neural.unified import UnifiedCore
from .workload import plan_workload,finish_workload


def cases():
    short='Chào bạn.'
    story='An đến thư viện để đọc sách. Bình mang theo một cuốn sổ. Hai bạn chọn một bàn gần cửa sổ rồi cùng ghi lại những điều vừa đọc.'
    items=[('greeting','Chào hỏi ngắn',short),('summary','Tóm tắt một đoạn','Tóm tắt đoạn sau: '+story)]
    items.append(('relations','Theo dõi nhiều sự kiện','Ai giữ cuốn sổ ở cuối câu chuyện? '+story+' An đưa sổ cho Bình. Bình đưa sổ cho Chi. Chi đi sang phòng bên cạnh, gặp Dũng rồi đưa cuốn sổ cho Dũng. Dũng đọc một trang và đưa sổ lại cho An. Hãy theo dõi thứ tự thay vì chỉ đếm tên.'))
    items.append(('long_input','Tải đầu vào dài','Tóm tắt và phân biệt những sự kiện lặp lại trong các đoạn sau. '+story+' '+story+' '+story+' '+story))
    items.append(('return','Quay về đầu vào ngắn',short))
    return items


class CapacityLab:
    def __init__(self,app):
        self.app=app;self.lock=threading.Lock();self.stop=threading.Event();self.thread=None
        self.value={'status':'idle','steps':[],'frames':[],'note':'Chưa chạy. Thử trên bản sao checkpoint, không sửa phiên trò chuyện.'}
        self.history_path=app.directory/'capacity-observation.json'
        if self.history_path.exists():
            try:
                if self.history_path.stat().st_size>1500000:raise ValueError('Observation too large')
                saved=json.loads(self.history_path.read_text(encoding='utf-8'))
                value=saved['observation']
                if saved['format']!='scarx-capacity-observation-v1' or value['status'] not in ('completed','cancelled','failed') or len(value['steps'])>5:raise ValueError('Observation format')
                for row in value['steps']:
                    if not 48<=row['active']<=row['allocated']<=128 or len(row['frames'])>24:raise ValueError('Observation count')
                    for frame in row['frames']:
                        if len(frame['spikes'])!=row['allocated'] or len(frame['voltage'])!=row['allocated'] or frame['allocated']!=row['allocated'] or frame['computed']!=row['active']:raise ValueError('Observation frame shape')
                        if any(s not in (0,1) for s in frame['spikes']) or not np.isfinite(frame['voltage']).all():raise ValueError('Observation frame values')
                value['restored_history']=True;self.value=value
            except (OSError,ValueError,KeyError,TypeError):self.value['note']='Kết quả quan sát cũ không đọc được; có thể chạy lại mà không ảnh hưởng checkpoint neural.'

    def _save(self,value=None):
        with self.lock:payload=json.dumps({'format':'scarx-capacity-observation-v1','observation':self.value if value is None else value},ensure_ascii=False,allow_nan=False).encode()
        if len(payload)>1500000:raise ValueError('Observation exceeds1.5MB')
        fd,tmp=tempfile.mkstemp(dir=self.app.directory,suffix='.observation.tmp')
        try:
            with os.fdopen(fd,'wb') as stream:stream.write(payload);stream.flush();os.fsync(stream.fileno())
            os.replace(tmp,self.history_path)
        finally:
            if os.path.exists(tmp):os.unlink(tmp)

    def status(self):
        with self.lock:return json.loads(json.dumps(self.value,ensure_ascii=False))

    def start(self,origin='release'):
        if origin not in ('release','current','live'):raise ValueError('Origin release/current/live')
        with self.lock:
            if self.thread is not None and self.thread.is_alive():return {'started':False,'reason':'Một lượt đang chạy','status':self.value['status']}
        with self.app.lock:
            checkpoint_before=self.app.checkpoint()['checkpoint'] if origin=='live' else None
            if origin in ('current','live'):payload=self.app.model.dumps()
            else:
                root=Path(__file__).resolve().parents[3];info=json.loads((root/'artifacts/generation1-release.json').read_text(encoding='utf-8'))
                path=(root/info['checkpoint']).resolve()
                if not path.is_relative_to((root/'artifacts').resolve()):raise ValueError('Release path')
                payload=path.read_bytes()
                if hashlib.sha256(payload).hexdigest()!=info['sha256']:raise ValueError('Release checksum')
        self.stop.clear()
        with self.lock:
            self.value={'status':'running','origin':origin,'started':time.time(),'steps':[],'frames':[],'checkpoint_before':checkpoint_before,
              'note':('Đang chạy trên lõi dùng trong Hermes. Đã lưu checkpoint trước lượt thử; chỉ xử lý task, không học hay tạo lời dạy/câu hỏi giả. Cấp phát mới được giữ lại.' if origin=='live' else
                'Bản sao checkpoint, không thay phiên trò chuyện.')+' Chọn theo tải đầu vào, chưa đo độ khó ngữ nghĩa.'}
            self.thread=threading.Thread(target=self._run,args=(payload,origin),daemon=True,name='scarx-capacity-lab');self.thread.start()
        return {'started':True,'status':'running'}

    def _run(self,payload,origin):
        try:
            model=self.app.model if origin=='live' else UnifiedCore.loads(payload)
            identity=id(model.core);initial_updates=model.core.updates
            for index,(identifier,title,prompt) in enumerate(cases()):
                if self.stop.is_set():break
                while self.app.pending and not self.stop.wait(.02):pass
                frames=[]
                def on_tick(kind,source):
                    if len(frames)>=24:return
                    c=model.core
                    frames.append({'kind':kind,'tick':model.ticks,'computed':model.active_count,'allocated':c.config.hidden,
                      'spikes':[int(v) if i<model.active_count else 0 for i,v in enumerate(c.spikes)],'voltage':np.round(c.voltage,4).tolist()})
                if origin=='live':
                    with self.app.attention():
                        allocated_before=model.core.config.hidden;start_ticks=model.ticks;updates_before=model.core.updates;began=time.perf_counter()
                        callback=model.on_tick
                        def live_tick(kind,source):
                            if callback:callback(kind,source)
                            on_tick(kind,source)
                        model.on_tick=live_tick
                        try:generated=self.app.chat(prompt,max_words=8,session_id='capacity-observer',record_experience=False)
                        finally:model.on_tick=callback
                        planned=dict(self.app.workload)
                        if model.core.updates!=updates_before:raise RuntimeError('Observation task unexpectedly learned')
                else:
                    allocated_before=model.core.config.hidden;planned=plan_workload(model,prompt)
                    model.select_capacity(planned['requested_neurons']);start_ticks=model.ticks;began=time.perf_counter()
                    model.on_tick=on_tick
                    try:generated=model.generate(prompt,max_words=8)
                    finally:model.on_tick=None
                workload=finish_workload(model,planned,self.app.process.memory_info().rss)
                row={'id':identifier,'title':title,'prompt':prompt,'words':len(prompt.split()),'allocated_before':allocated_before,
                  'allocated':model.core.config.hidden,'active':model.active_count,'dormant':model.core.config.hidden-model.active_count,
                  'grew_by':model.core.config.hidden-allocated_before,'max_neurons':model.max_neurons,'neural_ticks':model.ticks-start_ticks,
                  'inference_seconds':time.perf_counter()-began,'neural_arrays_bytes':model.core.memory_bytes(),
                  'timing_includes_persistence':origin=='live',
                  'process_rss_bytes':workload['process_rss_bytes'],'generated':generated['text'],'quality_verified':False,
                  'reason':planned['allocation_reason'],'same_core_identity':id(model.core)==identity,'weights_updated':False,'frames':frames}
                with self.lock:
                    self.value['steps'].append(row);self.value['frames']=frames;self.value['current']=title;self.value['progress']={'completed':index+1,'total':len(cases())}
                # Display dwell is separate from actual inference time above.
                # It lets the user see each measured transition without pretending
                # that waiting is additional neural reasoning.
                if self.stop.wait(.9):break
            if origin!='live' and model.core.updates!=initial_updates:raise RuntimeError('Demo unexpectedly learned')
            with self.lock:
                terminal=json.loads(json.dumps(self.value,ensure_ascii=False))
            terminal.update(status='cancelled' if self.stop.is_set() else 'completed',finished=time.time(),display_dwell_seconds=.9)
            terminal['note']=terminal['note'].replace('Đang chạy trên lõi','Đã chạy trên lõi')
            # Expose completion only after the durable record can be restored.
            self._save(terminal)
            with self.lock:self.value=terminal
        except Exception as exc:
            with self.lock:self.value.update(status='failed',error=str(exc)[:300],finished=time.time())

    def close(self):
        self.stop.set()
        if self.thread:self.thread.join(timeout=1)
