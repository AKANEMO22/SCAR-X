from pathlib import Path
import hashlib
import json
import socket
import subprocess
import sys
import time
from urllib.request import urlopen,Request
import uuid
import zipfile

root=Path(__file__).resolve().parents[1];package=root/'artifacts/SCAR-X-V0.2-LAB.zip'
destination=root/'local'/('v02-package-'+uuid.uuid4().hex[:8])
with zipfile.ZipFile(package) as archive:
    assert all(p.startswith('scar-x/') and '..' not in Path(p).parts and '/local/' not in p for p in archive.namelist())
    manifest=json.loads(archive.read('scar-x/PACKAGE_MANIFEST.json'))
    assert manifest['experimental'] and manifest['language_quality_gate_passed'] is False
    assert all(hashlib.sha256(archive.read('scar-x/'+name)).hexdigest()==digest for name,digest in manifest['sha256'].items())
    archive.extractall(destination)
checkout=destination/'scar-x'
with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
process=subprocess.Popen([sys.executable,str(checkout/'run_app.py'),'--port',str(port),'--data-dir',str(checkout/'local/qa')],cwd=checkout,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
try:
    base=f'http://127.0.0.1:{port}';status=None
    for _ in range(80):
        if process.poll() is not None:raise RuntimeError(process.stderr.read().decode('utf-8'))
        try:status=json.load(urlopen(base+'/api/status',timeout=1));break
        except OSError:time.sleep(.1)
    assert status is not None
    def post(path,payload,auth=False):
        headers={'Content-Type':'application/json'}
        if auth:headers['X-SCARX-Token']=status['token']
        return json.load(urlopen(Request(base+path,data=json.dumps(payload).encode(),headers=headers),timeout=10))
    models=json.load(urlopen(base+'/v1/models'))['data'];ids=[m['id'] for m in models]
    assert set(ids)=={'scarx-signal-v0.1','scarx-language-lab-v0.2','scarx-digits-lab-v0.2'}
    text=post('/v1/chat/completions',{'model':'scarx-language-lab-v0.2','messages':[{'role':'user','content':'chào bạn'}]})
    assert text['choices'][0]['message']['content']=='Chào bạn, tôi là scar-x.'
    assert text['scarx']['quality_gate_passed'] is False
    examples=json.load(urlopen(base+'/api/lab/examples'))
    image=post('/api/lab/vision',{'image':examples[0]['image']},True)
    assert image['digit']==examples[0]['expected_digit']
    result={'package_sha256':hashlib.sha256(package.read_bytes()).hexdigest(),'manifest_verified':True,'extracted_release_started':True,
            'model_ids':ids,'trained_greeting':text['choices'][0]['message']['content'],'language_general_quality_gate':False,
            'digit_fixture_prediction':image['digit'],'digit_fixture_expected':examples[0]['expected_digit'],
            'personal_data_in_package':False,'os_memory_limit':status['resources']['os_limit'],'limitations':'Only supervised greeting demonstrated; visiondigits only, no generalchat/screenshot capability.'}
    (root/'artifacts/v02-package-check.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=True,indent=2))
finally:process.terminate();process.wait(timeout=10)
