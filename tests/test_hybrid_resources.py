"""Hard-cap, GPU accounting and private-commit guard at the amended limit."""

from types import SimpleNamespace
import threading
import pytest
from scarx.hybrid import resources
from scarx.hybrid.config import MAX_RAM_BYTES


class GPU:
    def __init__(self):
        self.bytes = 0

    def sample(self, pids):
        return {"bytes": self.bytes, "available": True}

    def close(self):
        pass


def monitor(monkeypatch, rss, private, gpu=0):
    monkeypatch.setattr(resources, "SharedGPU", GPU)
    value = resources.ResourceMonitor()
    value.root = SimpleNamespace(
        pid=123,
        children=lambda recursive: [],
        memory_info=lambda: SimpleNamespace(rss=rss, private=private, vms=private),
        name=lambda: "fixture",
    )
    value.gpu.bytes = gpu
    return value


def test_cap_allows_exact_limit_and_rejects_one_more_byte(monkeypatch):
    m = monitor(monkeypatch, MAX_RAM_BYTES, 100)
    assert m.limit == 4_500_000_000
    assert not m.check()["exceeded"]
    m.gpu.bytes = 1
    with pytest.raises(MemoryError):
        m.check()
    assert m.last["conservative_ram_bytes"] == MAX_RAM_BYTES + 1
    m.close()


def test_private_commit_still_gated_when_rss_is_low(monkeypatch):
    m = monitor(monkeypatch, 100, MAX_RAM_BYTES + 1)
    with pytest.raises(MemoryError):
        m.check()
    m.close()


def test_async_breach_still_notifies_worker_shutdown(monkeypatch):
    m = monitor(monkeypatch, MAX_RAM_BYTES + 1, 100)
    called = threading.Event()
    m.on_exceed = called.set
    m.start()
    try:
        assert called.wait(1)
    finally:
        m.close()
