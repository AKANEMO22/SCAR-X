"""Passive capacity-study observer: actual events only, no model/SQLite access."""
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
import argparse,json,os
PAGE='''<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>SCAR-X · H64/H128</title>
<style>body{font:16px system-ui;background:#101923;color:#e2eaf2;margin:28px;max-width:1450px}h1{font-size:26px}small{color:#adc1d2}pre{white-space:pre-wrap;background:#1b2937;padding:18px;border-radius:10px;max-height:320px;overflow:auto}table{width:100%;border-collapse:collapse}th,td{text-align:left;padding:10px;border-bottom:1px solid #354a5b}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}#status{color:#91dfbd}@media(max-width:800px){.grid{grid-template-columns:1fr}}svg{background:#1b2937;border-radius:8px;width:100%;height:220px}</style>
<h1>SCAR-X · Thử dung lượng H64 và H128</h1><p id="status">Đang đọc log thật…</p><small>ALIF · local synaptic learning · CPU-first / 2GB. Mỗi trial dùng một core, không phải nhiều model ghép để trả lời. Đây là bài sự kiện2nhãn, chưa phải chat/vision tổng quát. Trang không điều khiển training.</small>
<div class="grid"><section><h2>Tiến độ / neuron / tài nguyên</h2><pre id="progress"></pre></section><section><h2>Cập nhật học thật</h2><pre id="learning"></pre><svg id="loss" viewBox="0 0 600 220" aria-label="Loss gần nhất"></svg></section></div>
<h2>So sánh đã hoàn tất</h2><small>Điểm từng cell, không phải kết luận cuối. Reused H48 được ghi rõ, không giả đang học lại.</small><table><thead><tr><th>H / arm / budget / task</th><th>Seed</th><th>Direct</th><th>Probe</th><th>OOD direct/probe</th><th>Updates / CPU</th><th>Nguồn</th></tr></thead><tbody id="scores"></tbody></table>
<h2>Đầu ra neural thật so với oracle</h2><table><thead><tr><th>H / arm / task / seed</th><th>Sự kiện đầu vào</th><th>Output</th><th>Nhãn</th></tr></thead><tbody id="answers"></tbody></table>
<h2>Nhật ký kỹ thuật</h2><pre id="activity"></pre><h2>Audit / kết luận khi đủ dữ liệu</h2><pre id="audit">Chưa có audit cuối.</pre>
<script>function show(id,v){document.getElementById(id).textContent=typeof v==='string'?v:JSON.stringify(v,null,2)}function cell(tr,v){let x=document.createElement('td');x.textContent=typeof v==='string'?v:JSON.stringify(v);tr.append(x)}async function get(p){let r=await fetch(p,{cache:'no-store'});return r.ok?await r.json():null}function chart(items){let s=document.getElementById('loss');s.replaceChildren();let a=items.map(x=>x.loss).filter(Number.isFinite);if(a.length<2)return;let max=Math.max(...a,.001);let line=document.createElementNS('http://www.w3.org/2000/svg','polyline');line.setAttribute('points',a.map((v,i)=>`${20+i*560/(a.length-1)},${195-v/max*175}`).join(' '));line.setAttribute('fill','none');line.setAttribute('stroke','#78d9b7');line.setAttribute('stroke-width','3');s.append(line);let t=document.createElementNS('http://www.w3.org/2000/svg','text');t.textContent='Loss gần nhất · '+a.length+' điểm thực';t.setAttribute('x','20');t.setAttribute('y','215');t.setAttribute('fill','#adc1d2');s.append(t)}async function refresh(){try{let p=await get('/progress.json');if(p){show('status',p.status+' · '+p.run);show('progress',p)}let e=await get('/events.json');if(e){show('learning',e.learning.slice(-4));chart(e.learning);show('activity',e.activity||[]);let a=document.getElementById('answers');a.replaceChildren();for(let x of e.answers.slice(-12)){let tr=document.createElement('tr');cell(tr,x.height+' / '+x.arm+' / '+x.task+' / '+x.seed);cell(tr,x.input);cell(tr,x.prediction);cell(tr,x.target);a.append(tr)}let b=document.getElementById('scores');b.replaceChildren();for(let x of (e.completed||[]).slice(-36)){let tr=document.createElement('tr');cell(tr,x.height+' / '+x.arm+' / '+x.budget+' / '+x.task);cell(tr,x.seed);cell(tr,(100*x.metrics.id_direct).toFixed(2)+'%');cell(tr,(100*x.metrics.id_probe).toFixed(2)+'%');cell(tr,(100*x.metrics.ood_direct).toFixed(2)+'% / '+(100*x.metrics.ood_probe).toFixed(2)+'%');cell(tr,x.episodes+' / '+x.cpu_seconds.toFixed(3)+'s');cell(tr,x.reused?'H48 cũ · hash verified':'Huấn luyện mới');b.append(tr)}}let a=await get('/audit.json');if(a)show('audit',a)}catch(e){show('status','Chưa đọc được log: '+e.message)}}refresh();setInterval(refresh,1500)</script></html>'''

def main():
    p=argparse.ArgumentParser();p.add_argument('--active-file',default='local/capacity-observer-active.json');p.add_argument('--port',type=int,default=8773);a=p.parse_args()
    if a.port in (8767,8771,8772):raise ValueError('Do not occupy existing observers')
    root=Path(__file__).resolve().parents[1];active=(root/a.active_file).resolve()
    if not active.is_relative_to(root/'local'):raise ValueError('Owned active pointer only')
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            route=self.path.split('?',1)[0];mime='application/json; charset=utf-8'
            if route=='/':body=PAGE.encode('utf-8');mime='text/html; charset=utf-8'
            elif route in ('/progress.json','/events.json','/audit.json'):
                run=Path(json.loads(active.read_text(encoding='utf-8'))['run']).resolve()
                if run.parent!=root/'runs' or not(run/'manifest.json').exists():self.send_error(403);return
                source=run/route[1:]
                if not source.exists():self.send_error(404);return
                if source.stat().st_size>2_000_000:self.send_error(413);return
                body=source.read_bytes()
            else:self.send_error(404);return
            self.send_response(200);self.send_header('Content-Type',mime);self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
        def log_message(self,*args):pass
    server=ThreadingHTTPServer(('127.0.0.1',a.port),Handler)
    (root/'local/capacity-observer-owner.json').write_text(json.dumps({'pid':os.getpid(),'port':a.port,'scope':'Read-only research logs, no model or live database'},indent=2),encoding='utf-8')
    print(f'Capacity observer http://127.0.0.1:{a.port}',flush=True);server.serve_forever()
if __name__=='__main__':main()
