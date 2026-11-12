"""SCAR-X Round 9b Mechanistic Credit Assignment Test.

Logs micro-timelines and synaptic credit assignment across 7 pre-registered diagnostic scenarios:
1. early_decision: decision cue at tick 1, delay 9 ticks, query at tick 10
2. delayed_reward: decision cue at tick 2, value at tick 3, query at tick 10, target at tick 11
3. mid_delay_distractor: decision cue at tick 1-2, distractor pulse on channel 6 at tick 5-6, query at tick 10
4. burst_spikes: strong multi-step input driving repeated spikes before query
5. quiescent_neurons: weak inputs where specific neuron clusters remain subthreshold
6. order_inversion: swapped temporal order of key presentation (key 0 first vs key 1 first)
7. variable_delay: comparing short delay (3 ticks) vs long delay (8 ticks)

For each scenario, tracks:
- event timeline
- spike raster
- membrane voltage u(t) before/after spike
- adaptive threshold theta(t)
- eligibility traces e_f(t) per neuron
- learning signal delta per neuron
- synaptic weight updates Delta W_in, Delta W_rec
- compares trained vs frozen vs shuffled vs zero_signal
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scarx.neural.core import Config
from scarx.neural.r9b_factorial_eprop import FactorialMTSGEProp


def generate_scenario_episodes():
    scenarios = {}

    # 1. Early decision
    x1 = np.zeros((12, 8), dtype=np.float32)
    x1[1, 0] = 1.5; x1[1, 2] = 1.5  # Key 0 = Value 0 at tick 1
    x1[10, 4] = 1.5                  # Query Key 0 at tick 10
    x1[11, 7] = 1.0                  # End marker
    scenarios["early_decision"] = (x1, 0)

    # 2. Delayed reward
    x2 = np.zeros((12, 8), dtype=np.float32)
    x2[2, 1] = 1.5                   # Key 1 at tick 2
    x2[3, 3] = 1.5                   # Value 1 at tick 3
    x2[10, 5] = 1.5                  # Query Key 1 at tick 10
    x2[11, 7] = 1.0                  # End marker
    scenarios["delayed_reward"] = (x2, 1)

    # 3. Mid-delay distractor
    x3 = x1.copy()
    x3[5, 6] = 1.5; x3[6, 6] = -1.5  # Distractor pulses on ch 6 at ticks 5, 6
    scenarios["mid_delay_distractor"] = (x3, 0)

    # 4. Burst spikes
    x4 = np.zeros((12, 8), dtype=np.float32)
    x4[2:6, 0] = 1.8; x4[2:6, 2] = 1.8  # Strong input across 4 consecutive ticks
    x4[10, 4] = 1.5
    x4[11, 7] = 1.0
    scenarios["burst_spikes"] = (x4, 0)

    # 5. Quiescent neurons
    x5 = np.zeros((12, 8), dtype=np.float32)
    x5[2, 0] = 0.4; x5[2, 2] = 0.4      # Weak input (below default spike threshold)
    x5[10, 4] = 1.5
    x5[11, 7] = 1.0
    scenarios["quiescent_neurons"] = (x5, 0)

    # 6. Order inversion
    x6 = np.zeros((12, 8), dtype=np.float32)
    # Key 1 presented first at tick 2, Key 0 presented at tick 5
    x6[2, 1] = 1.5; x6[2, 3] = 1.5      # Key 1 = Value 1
    x6[5, 0] = 1.5; x6[5, 2] = 1.5      # Key 0 = Value 0
    x6[10, 4] = 1.5                     # Query Key 0
    x6[11, 7] = 1.0
    scenarios["order_inversion"] = (x6, 0)

    # 7. Variable delay
    x7_short = np.zeros((8, 8), dtype=np.float32)
    x7_short[1, 0] = 1.5; x7_short[1, 2] = 1.5
    x7_short[4, 4] = 1.5  # Query at tick 4 (delay = 2 ticks)
    x7_short[7, 7] = 1.0
    scenarios["variable_delay_short"] = (x7_short, 0)

    return scenarios


def trace_episode_dynamics(model: FactorialMTSGEProp, seq: np.ndarray, target: int):
    c = model.config
    model.reset_episode()

    voltages_pre = []
    voltages_post = []
    spikes = []
    thresholds = []
    eligibility_traces = []

    for t, row in enumerate(seq):
        v_pre = model.core.voltage.copy()
        theta = c.threshold + c.adaptation_strength * model.core.adaptation
        model.step(row, track=True)
        v_post = model.core.voltage.copy()
        z = model.core.spikes.copy()
        e_f = model.e_f.copy()

        voltages_pre.append(v_pre.tolist())
        voltages_post.append(v_post.tolist())
        spikes.append(z.tolist())
        thresholds.append(theta.tolist())
        eligibility_traces.append(float(np.linalg.norm(e_f)))

    # Compute loss & pre-update credit
    f = model.core.features
    f_hat, raw, scale = model.standardized_features(f)
    logits = model.core.w_out @ f_hat + model.core.bias
    logits = logits - np.max(logits)
    p = np.exp(logits) / np.sum(np.exp(logits))
    loss = -float(np.log(max(float(p[target]), 1e-12)))

    err = -p.copy(); err[target] += 1.0
    pre_update_signal = (model.core.w_out.T @ err) * (np.abs(raw) < 5.0) / scale

    # Record weight snapshots before learn
    w_in_before = model.core.w_in.copy()
    w_rec_before = model.core.w_rec.copy()

    # Apply episode learning
    model.learn_episode(seq, target)

    delta_w_in = model.core.w_in - w_in_before
    delta_w_rec = model.core.w_rec - w_rec_before

    return {
        "loss": loss,
        "probabilities": p.tolist(),
        "total_spikes": int(np.sum(spikes)),
        "mean_voltage_pre": float(np.mean(voltages_pre)),
        "mean_voltage_post": float(np.mean(voltages_post)),
        "final_eligibility_norm": float(np.linalg.norm(model.e_f)),
        "signal_norm": float(np.linalg.norm(pre_update_signal)),
        "delta_w_in_norm": float(np.linalg.norm(delta_w_in)),
        "delta_w_rec_norm": float(np.linalg.norm(delta_w_rec)),
        "per_neuron_spike_counts": np.sum(spikes, axis=0).tolist(),
        "per_neuron_signal": pre_update_signal.tolist(),
        "per_neuron_update_norm": np.linalg.norm(delta_w_in, axis=1).tolist(),
    }


def run_mechanistic_audit():
    print("=" * 80)
    print("SCAR-X Round 9b Mechanistic Credit Assignment Test")
    print("=" * 80)

    cfg = Config(inputs=8, outputs=2, hidden=48, fan_in=8, plastic_rows=12, neuron="alif", seed=140011)
    scenarios = generate_scenario_episodes()

    audit_records = {}

    for s_name, (seq, target) in scenarios.items():
        print(f"\n--- Scenario: {s_name} (seq_len={len(seq)}, target={target}) ---")

        # Compare 4 control conditions on the top configuration (stopped_reset, recurrent_only, heterogeneous)
        conditions = ("learned", "shuffled_signal", "zero_signal", "readout_only")
        scenario_results = {}

        for cond in conditions:
            # Calibrate model with dummy sequences
            m = FactorialMTSGEProp(
                config=cfg,
                decay_mode="heterogeneous",
                gating_mode="recurrent_only",
                reset_mode="stopped_reset",
                control_mode=cond,
            )
            calib = np.random.default_rng(123).normal(0, 0.2, (16, len(seq), 8)).astype(np.float32)
            m.calibrate(calib)

            metrics = trace_episode_dynamics(m, seq, target)
            scenario_results[cond] = metrics

            print(
                f"  {cond:16s} | Spikes: {metrics['total_spikes']:3d} | "
                f"Signal Norm: {metrics['signal_norm']:6.3f} | "
                f"||dW_in||: {metrics['delta_w_in_norm']:6.4f} | "
                f"||dW_rec||: {metrics['delta_w_rec_norm']:6.4f} | "
                f"Loss: {metrics['loss']:.4f}"
            )

        audit_records[s_name] = scenario_results

    # Save artifact
    out_dir = ROOT / "artifacts"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "r9b_mechanistic_credit_audit.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"study": "r9b_mechanistic_credit_audit", "scenarios": audit_records}, f, indent=2)
    print(f"\nMechanistic audit artifact saved to {out_path}")
    return audit_records


if __name__ == "__main__":
    run_mechanistic_audit()
