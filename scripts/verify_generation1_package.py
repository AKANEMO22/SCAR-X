"""Extract/hash/start the shipped package in a fresh local directory."""
from pathlib import Path
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
import uuid
import zipfile
root=Path(__file__).resolve().parents[1]
target=root/'artifacts/SCAR-X-Generation-1.zip'
directory=(root/'local'/('generation1-package-'+uuid.uuid4().hex[:8])).resolve()
assert directory.is_relative_to((root/'local').resolve())
directory.mkdir()
with zipfile.ZipFile(target) as archive:
    for member in archive.infolist():
        destination=(directory/member.filename).resolve()
        assert destination.is_relative_to(directory) and not Path(member.filename).is_absolute()
    manifest=json.loads(archive.read('scar-x/PACKAGE_MANIFEST.json'))
    assert all(hashlib.sha256(archive.read('scar-x/'+name)).hexdigest()==digest for name,digest in manifest['sha256'].items())
    assert not any('/local/' in n or n.endswith('/.env') or '/backups/' in n for n in archive.namelist())
    archive.extractall(directory)
checkout=directory/'scar-x'
with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
args=[sys.executable,str(checkout/'run_app.py'),'--port',str(port),'--data-dir','local/package-session']
startup=None
if os.name=='nt':
    startup=subprocess.STARTUPINFO();startup.dwFlags|=subprocess.STARTF_USESHOWWINDOW;startup.wShowWindow=0
output=(directory/'stdout.log').open('wb');errors=(directory/'stderr.log').open('wb')
process=subprocess.Popen(args,cwd=checkout,stdout=output,stderr=errors,startupinfo=startup)
base='http://127.0.0.1:'+str(port)
def call(path,data=None):
    req=urllib.request.Request(base+path,None if data is None else json.dumps(data).encode(),headers={'Content-Type':'application/json'})
    return json.load(urllib.request.urlopen(req,timeout=10))
try:
    deadline=time.monotonic()+15
    while True:
        try:models=call('/v1/models');break
        except OSError:
            if process.poll() is not None or time.monotonic()>deadline:raise RuntimeError('Packaged server did not start; inspect local package logs')
            time.sleep(.15)
    assert [m['id'] for m in models['data']]==['scarx-generation1']
    answer=call('/v1/chat/completions',{'model':'scarx-generation1','messages':[{'role':'user','content':'hi'}]})
    assert answer['choices'][0]['message']['content']
    status=call('/api/status');assert status['cores_loaded']==1 and status['idle']['enabled'] and status['ram_bytes']<2_000_000_000
    result={'package':str(target),'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'manifest_files':len(manifest['sha256']),
      'hashes_verified':True,'private_files_excluded':True,'fresh_extract_started':True,'models':['scarx-generation1'],'cores_loaded':1,
      'greeting':answer['choices'][0]['message']['content'],'ram_bytes':status['ram_bytes'],'memory_limit':status['memory_limit']}
    (root/'artifacts/generation1-package-check.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result))
finally:
    process.terminate()
    try:process.wait(timeout=5)
    except subprocess.TimeoutExpired:process.kill();process.wait()
    output.close();errors.close()
