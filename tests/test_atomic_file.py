import pytest
from scarx.runtime import atomic_file


def test_sharing_error_retries_same_atomic_payload_without_exposing_partial_target(tmp_path,monkeypatch):
    source=tmp_path/'next.tmp';target=tmp_path/'journal.json';source.write_bytes(b'complete next');target.write_bytes(b'complete old')
    original=atomic_file.os.replace;calls=[]
    def sharing(a,b):
        calls.append((a,b));assert target.read_bytes()==b'complete old'
        if len(calls)<3:raise PermissionError('transient reader sharing')
        original(a,b)
    monkeypatch.setattr(atomic_file.os,'replace',sharing)
    atomic_file.replace_with_retry(source,target)
    assert len(calls)==3 and target.read_bytes()==b'complete next' and not source.exists()


def test_permanent_permission_error_is_bounded_and_preserves_old_file(tmp_path,monkeypatch):
    source=tmp_path/'next.tmp';target=tmp_path/'journal.json';source.write_bytes(b'next');target.write_bytes(b'old');calls=[]
    def denied(*args):calls.append(args);raise PermissionError('permanent deny')
    monkeypatch.setattr(atomic_file.os,'replace',denied)
    with pytest.raises(PermissionError):atomic_file.replace_with_retry(source,target,attempts=2)
    assert len(calls)==2 and target.read_bytes()==b'old' and source.read_bytes()==b'next'
