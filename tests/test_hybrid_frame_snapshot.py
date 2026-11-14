from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
from types import SimpleNamespace
from PIL import Image
import pytest
from scarx.hybrid.frame_snapshot import publish_snapshot
from scarx.hybrid.screen import ScreenMemory
from scarx.hybrid.store import Store
from scarx.hybrid.media import VideoMemory


def jpeg(color):
    out=io.BytesIO();Image.new('RGB',(32,32),color).save(out,format='JPEG');return out.getvalue()


def test_distinct_frame_requests_never_change_a_previous_returned_path(tmp_path):
    with ThreadPoolExecutor(max_workers=8) as pool:
        items=list(pool.map(lambda value:(publish_snapshot(tmp_path,value),value),[jpeg('red'),jpeg('green')]*16))
    assert len({path for path,_ in items})==2
    for path,value in items:
        assert path.read_bytes()==value and path.name=='question-frame-'+hashlib.sha256(value).hexdigest()+'.jpg'
    assert not list(tmp_path.glob('*.tmp'))


def test_existing_snapshot_integrity_is_checked_and_payload_is_bounded(tmp_path):
    value=jpeg('red');path=publish_snapshot(tmp_path,value);path.write_bytes(b'corrupted owned fixture')
    with pytest.raises(ValueError,match='integrity'):publish_snapshot(tmp_path,value)
    for invalid in [b'',b'x'*2_000_001,'not bytes']:
        with pytest.raises(ValueError):publish_snapshot(tmp_path,invalid)


def test_actual_screen_snapshots_are_isolated_for_different_historical_sources(tmp_path):
    store=Store(tmp_path);stream='a'*32;folder=tmp_path/'archive'/stream;folder.mkdir(parents=True)
    red,green=folder/'red.png',folder/'green.png';Image.new('RGB',(32,32),'red').save(red);Image.new('RGB',(32,32),'green').save(green)
    store.put('stream',stream,{'id':stream,'kind':'screen','latest_frame':str(green)})
    memory=ScreenMemory.__new__(ScreenMemory);memory.store=store;memory.directory=tmp_path
    try:
        first=memory.snapshot(stream,[{'id':'red','source':str(red),'modality':'sampled_frame','start':1}]);before=first.read_bytes()
        second=memory.snapshot(stream,[{'id':'green','source':str(green),'modality':'sampled_frame','start':2}])
        assert first!=second and first.read_bytes()==before and first.read_bytes()!=second.read_bytes()
    finally:store.close()


def test_content_addressed_cache_counts_toward_archive_quota_and_keeps_original(tmp_path,monkeypatch):
    store=Store(tmp_path);engine=SimpleNamespace(store=store,config=SimpleNamespace(archive_bytes=0,archive_seconds=1,segment_seconds=1))
    memory=VideoMemory(engine);folder=memory.archive/'a';folder.mkdir();snapshot=publish_snapshot(folder,jpeg('red'))
    import os,time
    os.utime(snapshot,(time.time()-100,time.time()-100));original=tmp_path/'original.jpg';original.write_bytes(b'KEEP ORIGINAL')
    try:
        memory.prune();assert not snapshot.exists() and original.read_bytes()==b'KEEP ORIGINAL'
        assert store.get('expired_source',str(snapshot)) is not None
    finally:store.close()
