"""Bounded public dataset retrieval; never execute pickle or archive paths."""
from pathlib import Path
import time,hashlib,json,urllib.request,tarfile
root=Path(__file__).resolve().parents[1];directory=root/'local/datasets/cifar10';directory.mkdir(parents=True,exist_ok=True)
path=directory/'cifar-10-binary.tar.gz';partial=path.with_suffix('.gz.partial')
url='https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz';expected='c32a1d4ab5d03f1284b67883e8d87530';start=time.monotonic()
if not path.exists():
    hasher=hashlib.md5();total=0;last=0
    with urllib.request.urlopen(url,timeout=45) as response,partial.open('wb') as out:
        final_url=response.url
        while block:=response.read(1024*1024):
            total+=len(block)
            if total>180000000 or time.monotonic()-start>360:raise RuntimeError('Registered download budget exceeded')
            hasher.update(block);out.write(block)
            if total-last>32*1024*1024:print(json.dumps({'download_bytes':total}),flush=True);last=total
    assert hasher.hexdigest()==expected,'Dataset MD5 mismatch'
    partial.replace(path)
else:
    hasher=hashlib.md5()
    with path.open('rb') as stream:
        while block:=stream.read(1024*1024):hasher.update(block)
    assert hasher.hexdigest()==expected,'Existing dataset checksum mismatch'
    final_url=url
target=directory/'data_batch_1.bin'
with tarfile.open(path,'r:gz') as archive:
    item=archive.getmember('cifar-10-batches-bin/data_batch_1.bin')
    assert item.isfile() and item.size==30730000
    file=archive.extractfile(item);data=file.read(item.size+1)
    assert len(data)==item.size;target.write_bytes(data)
receipt={'source_url':url,'resolved_download_url':final_url,'reference':'Alex Krizhevsky2009, Learning Multiple Layers of Features from Tiny Images',
 'md5_verified':expected,'archive_bytes':path.stat().st_size,'single_binary_batch_bytes':len(data),
 'batch_sha256':hashlib.sha256(data).hexdigest(),'local_batch':str(target),'original_test_opened':False,'dataset_redistributed':False,'seconds':time.monotonic()-start}
(root/'artifacts/rgb-object-dataset.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8');print(json.dumps(receipt),flush=True)
