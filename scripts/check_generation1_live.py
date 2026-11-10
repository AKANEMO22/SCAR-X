"""Check public loopback state and optional native Hermes model-picker API.

Session token is read into memory only and never printed or persisted.
"""
import argparse
import json
from pathlib import Path
import re
import sys
import threading
import urllib.request
import psutil
root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--hermes-port',type=int);args=parser.parse_args()
if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
def get(base,path,headers=None):return json.load(urllib.request.urlopen(urllib.request.Request(base+path,headers=headers or {}),timeout=30))
base='http://127.0.0.1:8765';models=get(base,'/v1/models');before=get(base,'/api/status')
threading.Event().wait(.65);after=get(base,'/api/status')
assert [r['id'] for r in models['data']]==['scarx-generation1']
assert after['cores_loaded']==1 and after['ticks']>before['ticks']
assert after['learning']['unverified_self_training'] is False
result={'models':['scarx-generation1'],'cores_loaded':1,'idle_advances':True,'idle_updates_delta':after['updates']-before['updates'],
  'idle_weight_learning':after['idle']['weight_learning'],'ram_bytes':after['ram_bytes'],
  'memory_limit':after['memory_limit'],'trace_capacity':after['trace_capacity'],
  'llama_server_process_count':sum(1 for p in psutil.process_iter(['name']) if (p.info['name'] or '').lower()=='llama-server.exe')}
if args.hermes_port:
    if not 1<=args.hermes_port<=65535:raise ValueError('Invalid Hermes port')
    base='http://127.0.0.1:'+str(args.hermes_port)
    html=urllib.request.urlopen(base+'/',timeout=10).read().decode('utf-8');headers={}
    for name,value in re.findall(r'window\.([A-Za-z_0-9]*(?:TOKEN|SESSION)[A-Za-z_0-9]*)\s*=\s*("[^"\n]*")',html,re.I):
        if 'SESSION' in name.upper() and 'TOKEN' in name.upper():headers['X-Hermes-Session-Token']=json.loads(value);break
    info=get(base,'/api/model/info',headers);options=get(base,'/api/model/options?explicit_only=true',headers)
    providers=options.get('providers',[])
    selected=next(p for p in providers if p.get('slug')=='scarx-local')
    assert info['model']=='scarx-generation1' and selected['models']==['scarx-generation1']
    assert selected['capabilities']['scarx-generation1']['reasoning'] is False
    result['hermes']={'configured_model':info['model'],'provider':info['provider'],'model_choices':selected['models'],
       'reasoning_control_supported':False,'scope':'verified native picker API; existing running chat may need a new session or /model command'}
(root/'artifacts/generation1-live-check.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
