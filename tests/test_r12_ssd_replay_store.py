"""SCAR-X Round 12: Unit Tests for SSD Replay Store."""
import shutil
import tempfile
from pathlib import Path
import pytest
import numpy as np

from scarx.neural.r12_ssd_replay_store import SSDReplayStore


@pytest.fixture
def tmp_store():
    d = tempfile.mkdtemp()
    store = SSDReplayStore(d, capacity=10)
    yield store
    shutil.rmtree(d, ignore_errors=True)


def test_ssd_store_basic_add_and_batch(tmp_store):
    for i in range(5):
        seq = np.random.randn(20, 8).astype(np.float32)
        tmp_store.add_episode(seq, i % 2)

    assert len(tmp_store.entries) == 5
    batch = tmp_store.get_replay_batch(3, seed=42)
    assert len(batch) == 3
    assert batch[0][0].shape == (20, 8)


def test_ssd_capacity_limit(tmp_store):
    # Store has capacity 10
    for i in range(15):
        seq = np.random.randn(20, 8).astype(np.float32)
        tmp_store.add_episode(seq, i % 2)

    assert len(tmp_store.entries) == 10


def test_ssd_snapshot_and_rollback(tmp_store):
    for i in range(4):
        seq = np.full((20, 8), float(i), dtype=np.float32)
        tmp_store.add_episode(seq, i % 2)

    tmp_store.create_snapshot("checkpoint_phase1")

    # Add more episodes
    for i in range(4, 8):
        seq = np.full((20, 8), float(i), dtype=np.float32)
        tmp_store.add_episode(seq, i % 2)

    assert len(tmp_store.entries) == 8

    # Rollback to checkpoint_phase1
    tmp_store.rollback("checkpoint_phase1")
    assert len(tmp_store.entries) == 4
    assert np.allclose(tmp_store.entries[-1]["data"][0][0], 3.0)


def test_ssd_tamper_detection(tmp_store):
    seq = np.random.randn(20, 8).astype(np.float32)
    tmp_store.add_episode(seq, 1)

    # Tamper with the file
    with open(tmp_store.db_file, "r") as f:
        text = f.read()

    # Tamper data without updating checksum
    tampered = text.replace('"target": 1', '"target": 0')
    with open(tmp_store.db_file, "w") as f:
        f.write(tampered)

    # Reloading should raise ValueError due to checksum mismatch
    with pytest.raises(ValueError, match="checksum mismatch"):
        SSDReplayStore(tmp_store.store_dir)
