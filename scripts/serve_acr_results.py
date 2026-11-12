"""Passive, bounded localhost evidence viewer. Never loads or runs a model."""
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
import argparse,json,os

PAGE='''<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>SCAR-X · ACR evidence</title>
<style>body{background:#111820;color:#e3eaf0;font:16px system-ui;margin:24px;max-width:1350px}h1{font-size:25px}small{color:#adc0cf}nav a{color:#8cd8ef;margin-right:20px}pre{white-space:pre-wrap;background:#1b2733;border-radius:10px;padding:16px;max-height:370px;overflow:auto}img{max-width:100%;background:white;border-radius:8px}table{width:100%;border-collapse:collapse}td,th{padding:9px;border-bottom:1px solid #384858;text-align:left}#decision{color:#9de7bc}section{margin-top:28px}</style>
<h1>SCAR-X · Adaptive Credit Routing</h1><p id="state">Đang đọc log thật…</p><p id="decision"></p>
<small>H48 ALIF · một forward recurrent · học synapse cục bộ theo protocol nghiên cứu. Đây là bài sự kiện hai nhãn; chưa phải chat/vision tổng quát. Spike/alpha là đo hoạt động, không phải lời nghĩ hoặc ý thức. Trang không train model.</small>
<nav><a href="#answers">Đầu ra thật</a><a href="#learning">Học và tài nguyên</a><a href="#routing">Neuron / alpha</a><a href="#evidence">Kết quả đã kiểm</a></nav>
<section id="learning"><h2>Tiến độ và tài nguyên</h2><pre id="progress"></pre><h3>Cập nhật synapse gần nhất</h3><pre id="updates"></pre></section>
<section id="answers"><h2>Input, output neural và nhãn oracle</h2><table><thead><tr><th>Task / seed / profile</th><th>Sự kiện</th><th>Neural</th><th>Đáp án</th></tr></thead><tbody id="answerRows"></tbody></table><small>Không chạy Qwen hay model thay thế; không có đầu ra Qwen trong phép đo này.</small></section>
<section id="routing"><h2>Trace alpha thật</h2><img src="/router-alpha.png" alt="Alpha theo tick và neuron từ checkpoint thật"><p>Biểu đồ lấy từ final-checkpoint TRAIN replay, 0 cập nhật học; mean curves không phải causal ranking giữa các model có weights khác nhau.</p><details><summary>Alpha theo neuron/tick và tương quan</summary><pre id="router"></pre></details><h3>Static mixing đã đo</h3><img src="/static-credit-screen.png" alt="Direct, probe và OOD của static alpha"></section>
<section id="evidence"><h2>Kết quả và kiểm toán</h2><pre id="audit"></pre><details><summary>Stress/OOD và kiểm trace</summary><pre id="mechanism"></pre></details><details><summary>Báo cáo ACR</summary><pre id="report"></pre></details></section>
<script>function set(id,value){document.getElementById(id).textContent=typeof value==='string'?value:JSON.stringify(value,null,2)}async function get(path){let r=await fetch(path,{cache:'no-store'});return r.ok?await r.json():null}function cell(tr,value){let td=document.createElement('td');td.textContent=typeof value==='string'?value:JSON.stringify(value);tr.append(td)}async function refresh(){try{let p=await get('/progress.json');if(p){set('state',p.status+' · '+p.run);set('progress',p)}let e=await get('/events.json');if(e){set('updates',e.learning.slice(-5));let body=document.getElementById('answerRows');body.replaceChildren();for(let a of e.answers.slice(-12)){let tr=document.createElement('tr');cell(tr,a.task+' / '+a.seed+' / '+a.arm+' / '+a.budget);cell(tr,a.input);cell(tr,a.prediction);cell(tr,a.target);body.append(tr)}}let r=await get('/router-summary.json');if(r)set('router',r);let a=await get('/audit-summary.json');if(a){set('decision',a.decision+' · integrity '+a.integrity);set('audit',a)}let m=await get('/mechanism-summary.json');if(m)set('mechanism',m);let t=await fetch('/report.txt',{cache:'no-store'});if(t.ok)set('report',await t.text())}catch(e){set('state','Chưa đọc được log: '+e.message)}}refresh();setInterval(refresh,2500)</script></html>'''

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--port',type=int,default=8771);ap.add_argument('--owner-file',default='local/acr-results-observer-owner.json');args=ap.parse_args()
    root=Path(__file__).resolve().parents[1];run=(root/args.run).resolve()
    if args.port==8767 or run.parent!=root/'runs' or not(run/'manifest.json').exists():raise ValueError('Owned run, separate port required')
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            route=self.path.split('?',1)[0];mime='application/json; charset=utf-8';body=None
            if route=='/':body=PAGE.encode();mime='text/html; charset=utf-8'
            elif route in ('/progress.json','/events.json'):
                source=run/route[1:]
                if source.exists() and source.stat().st_size<=1_000_000:body=source.read_bytes()
            elif route in ('/router-alpha.png','/static-credit-screen.png'):
                source=run/'figures'/route[1:]
                if source.exists() and source.stat().st_size<=2_000_000:body=source.read_bytes();mime='image/png'
            elif route in ('/router-summary.json','/audit-summary.json','/mechanism-summary.json'):
                name={'/router-summary.json':'router-analysis-v1.json','/audit-summary.json':'audit-v1.json','/mechanism-summary.json':'mechanism-audit-v1.json'}[route]
                source=run/name
                if source.exists() and source.stat().st_size<=10_000_000:
                    data=json.loads(source.read_text(encoding='utf-8'))
                    if route=='/router-summary.json':data={k:data[k] for k in ('status','rollouts','checkpoint_files','new_learning_updates','summary','limitations')}
                    elif route=='/mechanism-summary.json':data={k:data[k] for k in ('status','diagnostic_files','stress_files','worst_mixture_error','max_physical_delta_error','mechanism_summary','stress_summary','scope')}
                    body=json.dumps(data,ensure_ascii=False,allow_nan=False).encode('utf-8')
            elif route=='/report.txt':
                source=root/'docs/research/ACR_DEV_RESULTS_20261112.md'
                if source.exists() and source.stat().st_size<=500_000:body=source.read_bytes();mime='text/plain; charset=utf-8'
            if body is None:self.send_error(404);return
            self.send_response(200);self.send_header('Content-Type',mime);self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
        def log_message(self,*args):pass
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    owner=(root/args.owner_file).resolve()
    if not owner.is_relative_to(root/'local'):raise ValueError('Owner receipt outside isolated local')
    owner.write_text(json.dumps({'pid':os.getpid(),'port':args.port,'run':run.name,'scope':'Passive evidence viewer, no model/SQLite/live state'},indent=2),encoding='utf-8')
    print(f'Actual ACR evidence http://127.0.0.1:{args.port}',flush=True);server.serve_forever()
if __name__=='__main__':main()
