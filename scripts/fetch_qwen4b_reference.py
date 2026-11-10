"""Pinned public comparison weights only; never installed as a SCAR-X backend."""
from pathlib import Path
import hashlib,json,time,urllib.request,sys

root=Path(__file__).resolve().parents[1]
directory=root/'local/benchmark-models';directory.mkdir(parents=True,exist_ok=True)
name='Qwen3-4B-Instruct-2507-Q4_K_M.gguf'
revision='4edb920b6f14e3b9284d4502a6485103d72cde05'
digest='8cdb57cbb880d313736a9bc4e3d3d2485f145b5e19cf33783746e753e82641fc'
size=2497280448
path=directory/name;partial=directory/(name+'.partial')
url='https://huggingface.co/lmstudio-community/Qwen3-4B-Instruct-2507-GGUF/resolve/'+revision+'/'+name
receipt={'base_model':'Qwen/Qwen3-4B-Instruct-2507','quantizer':'lmstudio-community','revision':revision,
 'file':name,'expected_bytes':size,'expected_sha256':digest,'purpose':'isolated frozen comparison only; no imported weights in SCAR-X'}
started=time.monotonic()
try:
    if not path.exists():
        previous=0
        with urllib.request.urlopen(url,timeout=45) as response,partial.open('wb') as output:
            total=0;hasher=hashlib.sha256()
            while True:
                block=response.read(1024*1024)
                if not block:break
                output.write(block);hasher.update(block);total+=len(block)
                if total>size:raise ValueError('Download larger than pinned artifact')
                if total-previous>=256*1024*1024:
                    print(json.dumps({'downloaded_bytes':total,'expected_bytes':size}),flush=True);previous=total
                if time.monotonic()-started>360:raise TimeoutError('Registered download limit360s')
        assert total==size and hasher.hexdigest()==digest,'Pinned artifact checksum/size mismatch'
        partial.replace(path)
    else:
        hasher=hashlib.sha256()
        with path.open('rb') as source:
            while block:=source.read(1024*1024):hasher.update(block)
        assert path.stat().st_size==size and hasher.hexdigest()==digest,'Existing comparison artifact does not match'
    receipt.update({'available':True,'sha256_verified':True,'bytes':path.stat().st_size,'local_path':str(path),'seconds':time.monotonic()-started})
except Exception as exc:
    receipt.update({'available':False,'error_type':type(exc).__name__,'http_status':getattr(exc,'code',None),'seconds':time.monotonic()-started})
(root/'artifacts/qwen4b-reference-download.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print(json.dumps(receipt),flush=True)
if not receipt['available']:sys.exit(1)
