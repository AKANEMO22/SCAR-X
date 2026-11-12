"""Test whether Literature Adapters (BrainTrace) were bottlenecked by timescale/gating configuration."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from scarx.neural.core import Config
from scarx.neural.r10_independent_replication import IndependentLearningALIF
from scarx.evaluation.r8_benchmark_integrity import generate_strict_binding
from scarx.evaluation.r10_bottleneck_analysis import evaluate_model_standard


def test_braintrace_variants():
    seeds = (140011, 140017, 140023)
    results_default = []
    results_mtsg = []

    for seed in seeds:
        train_x, train_y = generate_strict_binding(seed + 200, 1024)
        refit_x, refit_y = generate_strict_binding(seed + 300, 256)
        id_x, id_y = generate_strict_binding(seed + 400, 512)
        ood_x, ood_y = generate_strict_binding(seed + 500, 512, ood=True)
        data = {"train": (train_x, train_y), "refit": (refit_x, refit_y), "id": (id_x, id_y), "ood": (ood_x, ood_y)}

        cfg = Config(inputs=8, outputs=2, hidden=48, fan_in=8, plastic_rows=12, seed=seed)

        # 1. Default BrainTrace (single slow timescale, no sensory gating)
        m_def = IndependentLearningALIF(cfg, method="braintrace")
        m_def.calibrate(train_x[:32])
        for ep in range(1024):
            idx = ep % len(train_x)
            m_def.learn_episode(train_x[idx], int(train_y[idx]))
        r_def = evaluate_model_standard(m_def, data)
        results_default.append(r_def)

        # 2. Matched BrainTrace (multi-timescale gamma_e in [0.25, 0.92] + sensory gating)
        m_mtsg = IndependentLearningALIF(cfg, method="braintrace")
        m_mtsg.trace_kind = "multi"
        m_mtsg.gate_mode = "activity"
        m_mtsg.gamma_e_vec = np.linspace(0.25, 0.92, 48, dtype=np.float32)[:, None]
        m_mtsg.calibrate(train_x[:32])
        for ep in range(1024):
            idx = ep % len(train_x)
            m_mtsg.learn_episode(train_x[idx], int(train_y[idx]))
        r_mtsg = evaluate_model_standard(m_mtsg, data)
        results_mtsg.append(r_mtsg)

        print(f"Seed: {seed}")
        print(f"  BrainTrace Default: Direct = {r_def['id_direct']*100:.2f}% | Probe = {r_def['id_probe']*100:.2f}% | OOD = {r_def['ood_direct']*100:.2f}%")
        print(f"  BrainTrace Matched: Direct = {r_mtsg['id_direct']*100:.2f}% | Probe = {r_mtsg['id_probe']*100:.2f}% | OOD = {r_mtsg['ood_direct']*100:.2f}%")
        print(f"  Delta: {r_mtsg['id_direct']*100 - r_def['id_direct']*100:+.2f} pp")

    mean_def = np.mean([r["id_direct"] for r in results_default]) * 100
    mean_mtsg = np.mean([r["id_direct"] for r in results_mtsg]) * 100
    print("\nSummary:")
    print(f"  BrainTrace Default Mean Direct: {mean_def:.2f}%")
    print(f"  BrainTrace Matched Mean Direct: {mean_mtsg:.2f}%")
    print(f"  Net Improvement: {mean_mtsg - mean_def:+.2f} pp")


if __name__ == "__main__":
    test_braintrace_variants()
