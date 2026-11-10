"""Start the extracted release, optionally import its own synthetic demo model."""
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

root=Path(__file__).resolve().parents[1]
package=root/'artifacts/SCAR-X-MVP-v0.1.zip'
destination=root/'local'/('release-check-'+uuid.uuid4().hex[:8])
with zipfile.ZipFile(package) as archive:
    assert all(n.startswith('scar-x/') and '..' not in Path(n).parts and '/local/' not in n for n in archive.namelist())
    manifest=json.loads(archive.read('scar-x/PACKAGE_MANIFEST.json'))
    assert all(hashlib.sha256(archive.read('scar-x/'+name)).hexdigest()==digest for name,digest in manifest['sha256'].items())
    archive.extractall(destination)
checkout=destination/'scar-x'
with socket.socket() as sock:
    sock.bind(('127.0.0.1',0)); port=sock.getsockname()[1]
process=subprocess.Popen([sys.executable,str(checkout/'run_app.py'),'--port',str(port),'--data-dir',str(checkout/'local/qa'),
                          '--import-checkpoint',str(checkout/'artifacts/scarx-demo-v01.checkpoint.zip')],
                         cwd=checkout,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
try:
    status=None; base=f'http://127.0.0.1:{port}'
    for _ in range(80):
        if process.poll() is not None: raise RuntimeError(process.stderr.read().decode('utf-8'))
        try: status=json.load(urlopen(base+'/api/status',timeout=1)); break
        except (OSError,TimeoutError): time.sleep(.1)
    assert status and status['neurons']==48 and status['updates']==1120
    assets={name:urlopen(base+'/'+name,timeout=3).status for name in ('','app.js','style.css','responsive.css')}
    assert json.load(urlopen(base+'/v1/models'))['data'][0]['id']=='scarx-signal-v0.1'
    body={'model':'scarx-signal-v0.1','messages':[{'role':'user','content':'{"task":"A","signal":2}'}]}
    response=json.load(urlopen(Request(base+'/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'}),timeout=5))
    parsed=json.loads(response['choices'][0]['message']['content'])
    assert parsed['result']['retrieval_used'] is False
    result={'package_sha256':hashlib.sha256(package.read_bytes()).hexdigest(),'manifest_verified':True,'extracted_checkout_started':True,
            'assets':assets,'demo_neural_checkpoint_imported':True,'updates':status['updates'],'openai_backend_response':True,
            'os_memory_limit':status['resources']['os_limit'],'personal_data_included':False,'hermes_mode':'model-backend; structured JSON only'}
    (root/'artifacts/mvp-v01-package-check.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
finally: process.terminate(); process.wait(timeout=10)
