"""Pinned public train parquet -> raw RGBrecords, no dataset remote code/pickle."""
from pathlib import Path
import hashlib,time,json,urllib.request,io
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
root=Path(__file__).resolve().parents[1];directory=root/'local/datasets/cifar10';directory.mkdir(parents=True,exist_ok=True)
revision='0b2714987fa478483af9968de7c934580d0bb9a2';sha='8428b53a88a11ac374111006708df51469e315a22ac6d66470afd9c78d2ae883';size=119705255
url=f'https://huggingface.co/datasets/uoft-cs/cifar10/resolve/{revision}/plain_text/train-00000-of-00001.parquet'
path=directory/'hf-train-pinned.parquet';partial=path.with_suffix('.parquet.partial');start=time.monotonic()
if not path.exists():
    digest=hashlib.sha256();total=0;last=0
    with urllib.request.urlopen(url,timeout=45) as response,partial.open('wb') as out:
        while block:=response.read(1024*1024):
            total+=len(block)
            if total>size or time.monotonic()-start>360:raise RuntimeError('Registered parquet download budget')
            out.write(block);digest.update(block)
            if total-last>=32*1024*1024:print(json.dumps({'bytes_downloaded':total,'expected':size}),flush=True);last=total
    if total!=size or digest.hexdigest()!=sha:raise ValueError('Pinned parquet checksum/size mismatch')
    partial.replace(path)
else:
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        while block:=stream.read(1024*1024):digest.update(block)
    if path.stat().st_size!=size or digest.hexdigest()!=sha:raise ValueError('Existing parquet checksum mismatch')
records=np.empty((10000,3073),np.uint8);count=0
table=pq.ParquetFile(path)
if table.metadata.num_rows!=50000:raise ValueError('Expected training rows')
for batch in table.iter_batches(batch_size=1024,columns=['img','label']):
    for row in batch.to_pylist():
        label=row['label'];image_bytes=row['img']['bytes']
        if type(label) is not int or not 0<=label<=9 or len(image_bytes)>500000:raise ValueError('Invalid image/label')
        with Image.open(io.BytesIO(image_bytes)) as image:
            if image.size!=(32,32) or image.format!='PNG':raise ValueError('Expected32x32PNG')
            rgb=np.asarray(image.convert('RGB'),np.uint8)
        records[count,0]=label;records[count,1:]=rgb.transpose(2,0,1).reshape(-1);count+=1
        if count==10000:break
    if count==10000:break
if count!=10000:raise ValueError('Incomplete batch')
target=directory/'hf-first-10000.bin';target.write_bytes(records.tobytes())
receipt={'source_url':'https://huggingface.co/datasets/uoft-cs/cifar10','revision':revision,'train_parquet_sha256':sha,'train_parquet_bytes':size,
 'reference':'AlexKrizhevsky2009, Learning Multiple Layers of Features from Tiny Images','local_batch':str(target),
 'binary_format':'first10000pinnedHFtrainrows converted to label+3072planarRGB; not asserted same order as officialbinarybatch1',
 'batch_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'single_binary_batch_bytes':target.stat().st_size,
 'original_test_opened':False,'dataset_redistributed':False,'license_status':'unknown on sourcecard; localresearch only',
 'protocol_amendment':'experiments/rgb-object-grounding-v1-source-amendment.md','seconds':time.monotonic()-start}
(root/'artifacts/rgb-object-dataset.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8');print(json.dumps(receipt),flush=True)
