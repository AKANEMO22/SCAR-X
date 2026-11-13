"""Actual CUDA parity/persistence, run explicitly in the GPU runtime environment."""

import json
import time
import numpy as np
import pytest
from scarx.hybrid.learner import LocalALIF, features
from scarx.hybrid.engine import HybridEngine
from scarx.hybrid.store import Store
from scarx.hybrid.config import HybridConfig
from types import SimpleNamespace


@pytest.mark.parametrize("seed", [260113, 260127, 260131])
def test_twenty_local_updates_match_cpu_and_cold_gpu_continuation(seed):
    cpu = LocalALIF(seed)
    gpu = LocalALIF(seed).to_device("gpu")
    assert all(type(a).__module__.startswith("cupy") for a in gpu.arrays().values())
    for key, a in cpu.arrays().items():
        np.testing.assert_array_equal(a, gpu.host(gpu.arrays()[key]))
    rng = np.random.default_rng(seed + 17)
    for stage in range(20):
        seq = rng.normal(0, 0.2, (8, 64)).astype(np.float32)
        target = stage % 6
        evidence = {"kind": "machine_verifier", "id": f"device-parity:{seed}:{stage}"}
        a = cpu.learn(seq, target, evidence=evidence)
        b = gpu.learn(seq, target, evidence=evidence)
        assert abs(a["loss"] - b["loss"]) < 3e-5
        for key, array in cpu.arrays().items():
            np.testing.assert_allclose(
                array,
                gpu.host(gpu.arrays()[key]),
                rtol=3e-5,
                atol=3e-5,
                err_msg=f"{seed}/{stage}/{key}",
            )
    restored, state = LocalALIF.loads(
        gpu.dumps({"purpose": "device parity"}), device="gpu"
    )
    assert state == {"purpose": "device parity"}
    assert restored.dumps() == gpu.dumps()
    evidence = {"kind": "machine_verifier", "id": "next step after GPU cold restore"}
    gpu.learn(seq, 1, evidence=evidence)
    restored.learn(seq, 1, evidence=evidence)
    assert restored.dumps() == gpu.dumps()


def test_guard_and_rollback_keep_gpu_numerics_and_restore_full_state(
    tmp_path, monkeypatch
):
    store = Store(tmp_path)
    monitor = SimpleNamespace(
        sample=lambda: {"rss_bytes": 40_000_000},
        check=lambda: {"rss_bytes": 40_000_000},
    )
    e = HybridEngine(
        store, SimpleNamespace(), monitor, HybridConfig(execution_mode="gpu")
    )
    e.model.to_device("gpu")
    initial = e.save(pinned=True)

    def no_cpu(*args, **kwargs):
        raise AssertionError("GPU mode tried a CPU neural step")

    monkeypatch.setattr("scarx.neural.core.SpikingCore.step", no_cpu)
    try:
        r = e._guarded_learn(
            features("verified example"),
            2,
            "action",
            1.0,
            {"kind": "machine_verifier", "id": "gpu-guard"},
        )
        assert r["accepted"] and e.model.device == "gpu"
        snapshot = e.model.dumps(e.state)

        def fail(*args, **kwargs):
            raise OSError("controlled disk write failure")

        monkeypatch.setattr(store, "save_snapshot", fail)
        r = e.cognition.consolidate_once()
        assert r["persistence_failed"] and e.model.dumps(e.state) == snapshot
        e.load(initial)
        assert e.model.device == "gpu" and e.model.updates == 0
        assert e.model.dumps(e.state) == store.snapshot(initial)
    finally:
        store.close()
