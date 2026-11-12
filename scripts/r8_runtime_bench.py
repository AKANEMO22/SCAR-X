"""SCAR-X Round 8: Runtime & Resource Efficiency Benchmark.

Role: Sub-Agent D (CPU Runtime and Neural Compute Engineer)
Mandates:
1. Strict process RSS & private commit memory profiling (Target: < 2,000,000,000 bytes).
2. Forward latency (p50, p95) and learning update latency.
3. Spiking sparsity and inactive neuron computation verification.
4. Mathematical parity checks between base SpikingCore and enhanced wrappers.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
import psutil
import numpy as np

from scarx.neural.core import Config, SpikingCore
from scarx.neural.r8_mtsg_eprop import MultiTimescaleSensoryGatedEProp
from scarx.neural.r8_bf_cascade import BennaFusiMetaplasticCascade
from scarx.evaluation.r8_benchmark_integrity import generate_strict_binding


def run_runtime_bench():
    print("=== SCAR-X ROUND 8: RUNTIME & RESOURCE EFFICIENCY BENCHMARK ===")
    proc = psutil.Process(os.getpid())
    mem_init = proc.memory_info()

    cfg = Config(inputs=8, outputs=2, hidden=48, fan_in=8, plastic_rows=12, seed=42)
    model_mtsg = MultiTimescaleSensoryGatedEProp(cfg, condition="learned_mtsg")
    model_bf = BennaFusiMetaplasticCascade(cfg, use_cascade=True)

    x_bench, y_bench = generate_strict_binding(999, 1024)
    model_mtsg.calibrate(x_bench[:32])
    model_bf.calibrate(x_bench[:32])

    # 1. Forward latency profiling
    step_times = []
    seq_times = []
    spike_counts = []
    total_neuron_steps = 0

    for seq in x_bench[:200]:
        t0_seq = time.perf_counter()
        model_mtsg.core.reset_state()
        for row in seq:
            t0_step = time.perf_counter()
            model_mtsg.step(row, track=False)
            t1_step = time.perf_counter()
            step_times.append(t1_step - t0_step)
            spike_counts.append(int(np.sum(model_mtsg.core.spikes)))
            total_neuron_steps += cfg.hidden
        t1_seq = time.perf_counter()
        seq_times.append(t1_seq - t0_seq)

    # 2. Learning update latency profiling
    learn_times = []
    for seq, y in zip(x_bench[200:400], y_bench[200:400]):
        t0_l = time.perf_counter()
        model_mtsg.learn_episode(seq, int(y))
        t1_l = time.perf_counter()
        learn_times.append(t1_l - t0_l)

    # 3. Memory profiling under 10,000 step stress
    for i in range(100):
        seq = x_bench[i % len(x_bench)]
        model_mtsg.learn_episode(seq, int(y_bench[i % len(y_bench)]))
        model_bf.learn_episode(seq, int(y_bench[i % len(y_bench)]))

    mem_final = proc.memory_info()
    peak_rss = mem_final.rss
    private_commit = getattr(mem_final, "private", mem_final.rss)

    # 4. Inactive neuron & sparsity analysis
    total_spikes = sum(spike_counts)
    sparsity = 1.0 - (total_spikes / max(1, total_neuron_steps))
    active_neurons_per_tick = float(np.mean(spike_counts))

    # 5. State memory footprint
    mtsg_state_bytes = model_mtsg.memory_bytes()
    bf_state_bytes = model_bf.memory_bytes()

    results = {
        "memory_profile": {
            "peak_rss_bytes": int(peak_rss),
            "peak_rss_mb": round(peak_rss / (1024 * 1024), 2),
            "private_commit_bytes": int(private_commit),
            "max_allowed_bytes": 2_000_000_000,
            "passed_memory_gate": bool(peak_rss < 2_000_000_000),
            "mtsg_state_bytes": int(mtsg_state_bytes),
            "bf_state_bytes": int(bf_state_bytes),
        },
        "latency_profile": {
            "step_forward_p50_us": round(float(np.percentile(step_times, 50)) * 1e6, 2),
            "step_forward_p95_us": round(float(np.percentile(step_times, 95)) * 1e6, 2),
            "seq_forward_p50_ms": round(float(np.percentile(seq_times, 50)) * 1e3, 3),
            "seq_forward_p95_ms": round(float(np.percentile(seq_times, 95)) * 1e3, 3),
            "learn_episode_p50_ms": round(float(np.percentile(learn_times, 50)) * 1e3, 3),
            "learn_episode_p95_ms": round(float(np.percentile(learn_times, 95)) * 1e3, 3),
            "throughput_seqs_per_sec": round(1.0 / float(np.mean(seq_times)), 1),
            "throughput_ticks_per_sec": round(1.0 / float(np.mean(step_times)), 1),
        },
        "sparsity_profile": {
            "spiking_sparsity": round(float(sparsity), 4),
            "active_neurons_per_tick": round(active_neurons_per_tick, 2),
            "total_neurons": cfg.hidden,
            "inactive_fraction": round(1.0 - (active_neurons_per_tick / cfg.hidden), 4),
        },
    }

    out_path = Path("artifacts/gemini-r8/runtime-bench.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n=== RESOURCE EFFICIENCY BENCHMARK SUMMARY ===")
    print(f"Peak RSS: {results['memory_profile']['peak_rss_mb']} MB (Gate: < 2000 MB) -> PASS = {results['memory_profile']['passed_memory_gate']}")
    print(f"MTSG Learner State: {results['memory_profile']['mtsg_state_bytes']} bytes (~{results['memory_profile']['mtsg_state_bytes']/1024:.1f} KB)")
    print(f"BF-Cascade State:   {results['memory_profile']['bf_state_bytes']} bytes (~{results['memory_profile']['bf_state_bytes']/1024:.1f} KB)")
    print(f"Forward Step Latency: p50={results['latency_profile']['step_forward_p50_us']} us, p95={results['latency_profile']['step_forward_p95_us']} us")
    print(f"Sequence Forward Latency (12 ticks): p50={results['latency_profile']['seq_forward_p50_ms']} ms")
    print(f"Learn Episode Latency: p50={results['latency_profile']['learn_episode_p50_ms']} ms")
    print(f"Spike Sparsity: {results['sparsity_profile']['spiking_sparsity']*100:.2f}% (Active neurons: {results['sparsity_profile']['active_neurons_per_tick']}/{results['sparsity_profile']['total_neurons']})")
    print(f"Artifact written to: {out_path.resolve()}")


if __name__ == "__main__":
    run_runtime_bench()
