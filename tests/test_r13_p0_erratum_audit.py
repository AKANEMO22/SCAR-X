"""SCAR-X Round 13: Phase 0 Erratum & Metric Re-audit Tests."""
import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_recalculate_r12_protocol_forgetting():
    r12_path = ROOT / "artifacts" / "r12_continual_memory_results.json"
    assert r12_path.exists()

    with open(r12_path) as f:
        data = json.load(f)

    # 1. Verify Reservoir-64 Order 1 protocol mean forgetting is ~22.37pp
    res_64_o1 = data["order_1_arms"]["arm_d_reservoir_64"]
    o1_protocol_forg = []
    for e in res_64_o1:
        snaps = e["phase_snapshots"]
        tasks = e["task_order"][:-1]
        seed_forgs = []
        for t_idx, t in enumerate(tasks):
            max_p = max(snaps[p]["eval"][t]["id_direct"] for p in range(len(snaps)))
            fin = snaps[-1]["eval"][t]["id_direct"]
            seed_forgs.append(max_p - fin)
        o1_protocol_forg.append(np.mean(seed_forgs))

    mean_o1_proto = np.mean(o1_protocol_forg) * 100
    assert abs(mean_o1_proto - 22.37) < 0.1, f"Expected ~22.37pp, got {mean_o1_proto:.2f}pp"

    # 2. Verify Reservoir-64 Order 2 protocol mean forgetting is ~13.45pp
    res_64_o2 = data["order_2_arms"]["arm_d_reservoir_64"]
    o2_protocol_forg = []
    for e in res_64_o2:
        snaps = e["phase_snapshots"]
        tasks = e["task_order"][:-1]
        seed_forgs = []
        for t_idx, t in enumerate(tasks):
            max_p = max(snaps[p]["eval"][t]["id_direct"] for p in range(len(snaps)))
            fin = snaps[-1]["eval"][t]["id_direct"]
            seed_forgs.append(max_p - fin)
        o2_protocol_forg.append(np.mean(seed_forgs))

    mean_o2_proto = np.mean(o2_protocol_forg) * 100
    assert abs(mean_o2_proto - 13.45) < 0.1, f"Expected ~13.45pp, got {mean_o2_proto:.2f}pp"
