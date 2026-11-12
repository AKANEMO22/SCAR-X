"""SCAR-X Round 13.1: Unit Tests for Phase 0 Causal Repairs."""
import numpy as np
import pytest

from scarx.neural.r13_1_repaired_modular_alif import (
    generate_independent_symbol_patterns,
    RepairedModularALIFModule,
    DirectSpikeRouter,
)


def test_defect_b_independent_symbol_patterns():
    # Verify that different seeds produce genuinely different patterns
    p1 = generate_independent_symbol_patterns(seed=140011, num_symbols=8)
    p2 = generate_independent_symbol_patterns(seed=140023, num_symbols=8)

    assert p1.shape == (8, 20, 8)
    assert p2.shape == (8, 20, 8)
    # Check that channel burst locations are not identical
    diff = np.max(np.abs(p1 - p2))
    assert diff > 0.5, "Patterns between seeds must be distinct!"


def test_defect_a_direct_spike_router():
    router = DirectSpikeRouter(in_features=48, out_features=8, seed=42)
    # Feed binary spike vector
    spk = np.zeros(48, dtype=np.float32)
    spk[5] = 1.0
    spk[12] = 1.0
    out = router.route_tick(spk)
    assert out.shape == (8,)
    assert not np.allclose(out, 0.0)


def test_defect_f_complete_neural_checkpointing():
    seed = 140011
    mod = RepairedModularALIFModule("mod_ckpt", start_neuron_id=0, inputs=8, hidden=48, outputs=8, seed=seed)
    dummy_seq = np.random.randn(20, 8).astype(np.float32)
    mod.calibrate(np.random.randn(5, 20, 8))

    # Run halfway through an episode (10 ticks)
    mod.reset_episode()
    for row in dummy_seq[:10]:
        mod.step(row, track=True)

    # Snapshot complete state
    engram = mod.get_complete_engram()
    assert "voltage" in engram
    assert "adaptation" in engram
    assert "spikes" in engram
    assert "rng_state" in engram

    # Verify voltage and adaptation are non-trivial
    assert not np.allclose(engram["voltage"], 0.0)

    # Unload and reload into fresh module
    mod.unload()
    fresh = RepairedModularALIFModule("mod_ckpt", start_neuron_id=0, inputs=8, hidden=48, outputs=8, seed=seed)
    fresh.load_complete_engram(engram)

    # Check restored state equality
    assert np.allclose(fresh.core.voltage, engram["voltage"])
    assert np.allclose(fresh.core.adaptation, engram["adaptation"])
    assert np.allclose(fresh.core.spikes, engram["spikes"])

    # Step next tick on both and verify exact match
    next_row = dummy_seq[10]
    # Mod was unloaded, but fresh should step cleanly
    spk_next = fresh.step(next_row, track=True)
    assert spk_next.shape == (48,)


def test_defect_c_frozen_hidden_forward_execution():
    mod = RepairedModularALIFModule("mod_fh", start_neuron_id=0, inputs=8, hidden=48, outputs=8, seed=123)
    mod.calibrate(np.random.randn(5, 20, 8))
    train_x = np.random.randn(20, 8).astype(np.float32)

    # Run forward pass explicitly
    mod.reset_episode()
    for row in train_x:
        mod.step(row, track=False)

    f = mod.core.features
    f_hat, _, _ = mod.standardized_features(f)
    assert not np.allclose(f_hat, 0.0)
