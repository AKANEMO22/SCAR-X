"""SCAR-X Round 13: Unit Tests for SSD Neural Engram Store."""
import tempfile
import shutil
from pathlib import Path
import pytest
import numpy as np

from scarx.neural.r13_modular_alif import ModularALIFModule
from scarx.neural.r13_neural_engram_store import NeuralEngramStore


@pytest.fixture
def tmp_store():
    d = tempfile.mkdtemp()
    store = NeuralEngramStore(d)
    yield store
    shutil.rmtree(d, ignore_errors=True)


def test_engram_save_reload_roundtrip(tmp_store):
    seed = 140011
    mod = ModularALIFModule("test_module", start_neuron_id=0, inputs=8, hidden=48, outputs=8, seed=seed)
    dummy_data = np.random.randn(10, 20, 8).astype(np.float32)
    mod.calibrate(dummy_data)

    # Do a training step
    loss = mod.learn_episode(dummy_data[0], 3)
    orig_w_in = mod.core.w_in.copy()
    orig_w_rec = mod.core.w_rec.copy()
    orig_w_out = mod.core.w_out.copy()

    # Save to SSD
    tmp_store.save_engram(mod, "engram_mod1")
    assert tmp_store.exists("engram_mod1")

    # Unload module
    tmp_store.unload_module(mod)
    assert mod.is_unloaded
    assert mod.core is None  # memory freed

    # Reload into fresh module
    fresh_mod = ModularALIFModule("test_module", start_neuron_id=0, inputs=8, hidden=48, outputs=8, seed=seed)
    tmp_store.load_engram("engram_mod1", fresh_mod)

    assert not fresh_mod.is_unloaded
    assert np.allclose(fresh_mod.core.w_in, orig_w_in)
    assert np.allclose(fresh_mod.core.w_rec, orig_w_rec)
    assert np.allclose(fresh_mod.core.w_out, orig_w_out)


def test_next_neural_tick_and_learning_equivalence(tmp_store):
    seed = 140011
    m1 = ModularALIFModule("mod1", start_neuron_id=0, inputs=8, hidden=48, outputs=8, seed=seed)
    dummy_data = np.random.randn(10, 20, 8).astype(np.float32)
    m1.calibrate(dummy_data)
    m1.learn_episode(dummy_data[0], 2)

    # Save m1
    tmp_store.save_engram(m1, "engram_equiv")

    # Load into m2
    m2 = ModularALIFModule("mod2", start_neuron_id=0, inputs=8, hidden=48, outputs=8, seed=seed)
    tmp_store.load_engram("engram_equiv", m2)

    # 1. Forward equivalence
    test_seq = np.random.randn(20, 8).astype(np.float32)
    p1, spk1 = m1.forward_sequence(test_seq)
    p2, spk2 = m2.forward_sequence(test_seq)
    assert np.allclose(p1, p2)
    assert np.allclose(spk1, spk2)

    # 2. Next learning step equivalence
    train_seq = np.random.randn(20, 8).astype(np.float32)
    loss1 = m1.learn_episode(train_seq, 5)
    loss2 = m2.learn_episode(train_seq, 5)
    assert abs(loss1 - loss2) < 1e-6
    assert np.allclose(m1.core.w_in, m2.core.w_in)
    assert np.allclose(m1.core.w_rec, m2.core.w_rec)
    assert np.allclose(m1.core.w_out, m2.core.w_out)


def test_tamper_detection(tmp_store):
    mod = ModularALIFModule("tamper_mod", start_neuron_id=0, inputs=8, hidden=48, outputs=8, seed=123)
    mod.calibrate(np.random.randn(5, 20, 8))
    tmp_store.save_engram(mod, "engram_tamper")

    # Modify file contents directly
    file_path = tmp_store._engram_path("engram_tamper")
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Tamper with weight values
    tampered = text.replace('"total_episodes": 0', '"total_episodes": 9999')
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(tampered)

    fresh = ModularALIFModule("fresh", start_neuron_id=0, inputs=8, hidden=48, outputs=8, seed=123)
    with pytest.raises(ValueError, match="checksum mismatch"):
        tmp_store.load_engram("engram_tamper", fresh)


def test_snapshot_and_rollback(tmp_store):
    mod = ModularALIFModule("mod_snap", start_neuron_id=0, inputs=8, hidden=48, outputs=8, seed=123)
    dummy_data = np.random.randn(5, 20, 8).astype(np.float32)
    mod.calibrate(dummy_data)
    mod.learn_episode(dummy_data[0], 1)
    orig_w_out = mod.core.w_out.copy()

    # Save & snapshot
    tmp_store.save_engram(mod, "engram_snap")
    tmp_store.create_snapshot("engram_snap", "v1")

    # Learn more and re-save
    mod.learn_episode(dummy_data[1], 4)
    tmp_store.save_engram(mod, "engram_snap")

    # Rollback to v1
    tmp_store.rollback("engram_snap", "v1")
    fresh = ModularALIFModule("fresh", start_neuron_id=0, inputs=8, hidden=48, outputs=8, seed=123)
    tmp_store.load_engram("engram_snap", fresh)

    assert np.allclose(fresh.core.w_out, orig_w_out)
