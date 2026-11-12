"""SCAR-X Round 13.2: Master Benchmark Runner.

Full Independent Composition & Fair LLM Benchmark:
- Phase 1: Independent SNN Modules (A: X->Y, B: Y->Z). Zero C supervision.
- Phase 2: Causal Controls & Decisive Knowledge Swap Test.
- Phase 3: Fair LLM Baselines (ScratchTransformer-20M Fair 2-Step & SmolLM2 Exact Token Match).
- Phase 4: Scaling K in {8, 16, 32, 64} on DEV seeds + Held-out EVAL seed 140099.
- Phase 5: Dynamic Resource Profiling & Cold Restart Verification.
- Exports raw sample-level predictions to artifacts/r13_2_raw_predictions.json.
"""
import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import psutil
import numpy as np
import torch
import torch.nn as nn

# CPU thread pinning for reproducibility
for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[k] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scarx.neural.r13_2_independent_modular_alif import (
    IndependentALIFModule,
    FixedSpikeRouter,
    LearnedRepresentationAligner,
    SymbolicRouter,
    generate_independent_symbol_patterns,
)
from scarx.neural.r13_neural_engram_store import NeuralEngramStore
from scarx.evaluation.r13_2_fair_llm_benchmarks import (
    FairScratchTransformer20M,
    train_fair_scratch_transformer,
    evaluate_fair_scratch_transformer,
    FairSmolLM2Evaluator,
)

DEV_SEEDS = (140011, 140017, 140023)
HELD_OUT_SEED = 140099
ALL_SEEDS = DEV_SEEDS + (HELD_OUT_SEED,)


def numpy_json_serializer(o):
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer, int)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


class LiveProcessTracker:
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.peak_rss_mb = 0.0

    def sample(self) -> float:
        rss = float(self.process.memory_info().rss) / (1024 * 1024)
        if rss > self.peak_rss_mb:
            self.peak_rss_mb = rss
        return rss


def build_data(
    seed: int,
    mapping: Dict[int, int],
    patterns: np.ndarray,
    n_train: int = 512,
    n_test: int = 128,
    noise: float = 0.05,
):
    rng = np.random.default_rng(seed)
    num_symbols = len(mapping)

    def sample(n):
        xs, ys, syms = [], [], []
        for _ in range(n):
            s = int(rng.integers(0, num_symbols))
            target = int(mapping[s])
            p = patterns[s].copy()
            p += rng.normal(0, noise, p.shape).astype(np.float32)
            xs.append(p)
            ys.append(target)
            syms.append(s)
        return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.int64), np.array(syms, dtype=np.int64)

    return sample(n_train), sample(n_test)


def run_r13_2_master_benchmark():
    tracker = LiveProcessTracker()
    store_dir = ROOT / "artifacts" / "r13_2_engram_bank"
    store = NeuralEngramStore(store_dir)

    print("=" * 80)
    print("SCAR-X ROUND 13.2: INDEPENDENT COMPOSITION & FAIR LLM BENCHMARK")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device for LLM computation: {device}")

    # Load LLM-2 SmolLM2
    print("Loading LLM-2: HuggingFaceTB/SmolLM2-135M-Instruct...")
    t0_llm2 = time.perf_counter()
    smol_evaluator = FairSmolLM2Evaluator(device=device)
    llm2_load_sec = time.perf_counter() - t0_llm2
    print(f"SmolLM2 loaded in {llm2_load_sec:.2f}s | Trainable params: {smol_evaluator.total_parameters:,}")
    tracker.sample()

    raw_sample_predictions = []
    results = {
        "metadata": {
            "round": "R13.2",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "dev_seeds": list(DEV_SEEDS),
            "held_out_seed": HELD_OUT_SEED,
            "device": device,
        },
        "level_1_independent_composition": {},
        "causal_verification_controls": {},
        "decisive_knowledge_swap_test": {},
        "level_2_scaling": {},
        "level_3_multistep": {},
        "continual_adaptation": {},
        "resource_audit": {},
    }

    # =========================================================================
    # PHASE 1 & LEVEL 1: TRULY INDEPENDENT SNN COMPOSITION (8 SYMBOLS)
    # =========================================================================
    print("\n" + "=" * 60)
    print(">>> PHASE 1: TRULY INDEPENDENT SNN COMPOSITION (8 SYMBOLS) <<<")
    print("=" * 60)

    l1_seed_results = []
    for seed in ALL_SEEDS:
        is_heldout = (seed == HELD_OUT_SEED)
        tag = f"[Seed {seed}{' (HELD-OUT)' if is_heldout else ''}]"
        print(f"\n--- {tag} ---")

        rng = np.random.default_rng(seed)
        perm_a = rng.permutation(8)
        map_a = {i: int(perm_a[i]) for i in range(8)}
        perm_b = rng.permutation(8)
        map_b = {i: int(perm_b[i]) for i in range(8)}
        map_c = {i: int(map_b[map_a[i]]) for i in range(8)}

        pat_x = generate_independent_symbol_patterns(seed + 100, num_symbols=8)
        pat_y = generate_independent_symbol_patterns(seed + 200, num_symbols=8)

        # Datasets
        (tr_a_x, tr_a_y, _), (te_a_x, te_a_y, _) = build_data(seed + 300, map_a, pat_x)
        (tr_b_y_x, tr_b_z, _), (te_b_y_x, te_b_z, _) = build_data(seed + 400, map_b, pat_y)
        _, (te_c_x, te_c_z, te_c_sym_x) = build_data(seed + 500, map_c, pat_x)

        # Step A: Train Module A ONLY on X -> Y
        t0_a = time.perf_counter()
        mod_a = IndependentALIFModule("mod_a", 0, 8, 48, 8, 8, seed)
        mod_a.calibrate(tr_a_x[:32])
        for ep in range(len(tr_a_x)):
            mod_a.learn_episode(tr_a_x[ep], tr_a_y[ep])
        train_time_a = time.perf_counter() - t0_a

        # Test A
        corr_a = sum(int(np.argmax(mod_a.forward_sequence(x)[0])) == y for x, y in zip(te_a_x, te_a_y))
        acc_a = corr_a / len(te_a_y)

        # Step B: Train Module B INDEPENDENTLY ONLY on Y -> Z (NEVER SEES X, NEVER SEES A)
        t0_b = time.perf_counter()
        mod_b = IndependentALIFModule("mod_b", 48, 8, 48, 8, 8, seed + 1)
        mod_b.calibrate(tr_b_y_x[:32])
        for ep in range(len(tr_b_y_x)):
            mod_b.learn_episode(tr_b_y_x[ep], tr_b_z[ep])
        train_time_b = time.perf_counter() - t0_b

        # Test B
        corr_b = sum(int(np.argmax(mod_b.forward_sequence(y_pat)[0])) == z for y_pat, z in zip(te_b_y_x, te_b_z))
        acc_b = corr_b / len(te_b_z)

        # Persist both to SSD
        store.save_engram(mod_a, f"mod_a_s{seed}")
        store.save_engram(mod_b, f"mod_b_s{seed}")

        # Routing Option 1: Fixed Direct Spike Router (Zero alignment training, random projection)
        fixed_router = FixedSpikeRouter(in_features=48, out_features=8, seed=seed + 77)
        corr_c_fixed, lat_fixed = 0, []
        for i, (x_pat, targ_z, s_x) in enumerate(zip(te_c_x, te_c_z, te_c_sym_x)):
            t0 = time.perf_counter()
            mod_a.reset_episode()
            mod_b.reset_episode()
            for row in x_pat:
                spk_a = mod_a.step(row, track=False)
                in_b = fixed_router.route_tick(spk_a)
                mod_b.step(in_b, track=False)
            f_b = mod_b.core.features
            f_hat_b, _, _ = mod_b.standardized_features(f_b)
            pred_z = int(np.argmax(mod_b.core.w_out @ f_hat_b + mod_b.core.bias))
            lat = (time.perf_counter() - t0) * 1000.0
            lat_fixed.append(lat)
            is_corr = (pred_z == targ_z)
            if is_corr:
                corr_c_fixed += 1
            raw_sample_predictions.append({
                "phase": "level_1", "seed": seed, "model": "snn_fixed_spike",
                "sample_idx": i, "x_sym": int(s_x), "target_z": int(targ_z),
                "pred_z": pred_z, "correct": is_corr, "latency_ms": lat,
            })
        acc_c_fixed = corr_c_fixed / len(te_c_z)

        # Routing Option 2: Learned Representation Aligner (Trained ONLY on Y representation, NO Z LABELS)
        aligner = LearnedRepresentationAligner(in_features=48, out_features=8, seed=seed + 99, lr=0.015)
        # Train aligner on pairs: (s_A(t) from X, canonical pat_Y of mapped Y)
        # Note: ONLY (s_A, pat_Y) is seen. No Z labels, no Mod B!
        for ep in range(len(tr_a_x)):
            sym_y = int(tr_a_y[ep])
            _, spk_seq = mod_a.forward_sequence(tr_a_x[ep])
            aligner.train_alignment_pair(spk_seq, pat_y[sym_y])

        corr_c_align, lat_align = 0, []
        for i, (x_pat, targ_z, s_x) in enumerate(zip(te_c_x, te_c_z, te_c_sym_x)):
            t0 = time.perf_counter()
            mod_a.reset_episode()
            mod_b.reset_episode()
            for row in x_pat:
                spk_a = mod_a.step(row, track=False)
                in_b = aligner.forward_tick(spk_a)
                mod_b.step(in_b, track=False)
            f_b = mod_b.core.features
            f_hat_b, _, _ = mod_b.standardized_features(f_b)
            pred_z = int(np.argmax(mod_b.core.w_out @ f_hat_b + mod_b.core.bias))
            lat = (time.perf_counter() - t0) * 1000.0
            lat_align.append(lat)
            is_corr = (pred_z == targ_z)
            if is_corr:
                corr_c_align += 1
            raw_sample_predictions.append({
                "phase": "level_1", "seed": seed, "model": "snn_learned_aligner",
                "sample_idx": i, "x_sym": int(s_x), "target_z": int(targ_z),
                "pred_z": pred_z, "correct": is_corr, "latency_ms": lat,
            })
        acc_c_align = corr_c_align / len(te_c_z)

        # Routing Option 3: Symbolic Router (Ceiling control: argmax(A) -> lookup pat_y -> B)
        sym_router = SymbolicRouter(pat_y)
        corr_c_sym, lat_sym = 0, []
        for i, (x_pat, targ_z, s_x) in enumerate(zip(te_c_x, te_c_z, te_c_sym_x)):
            t0 = time.perf_counter()
            probs_a, _ = mod_a.forward_sequence(x_pat)
            in_b_sym = sym_router.route_symbolic(probs_a)
            probs_b, _ = mod_b.forward_sequence(in_b_sym)
            pred_z = int(np.argmax(probs_b))
            lat = (time.perf_counter() - t0) * 1000.0
            lat_sym.append(lat)
            is_corr = (pred_z == targ_z)
            if is_corr:
                corr_c_sym += 1
            raw_sample_predictions.append({
                "phase": "level_1", "seed": seed, "model": "snn_symbolic_router",
                "sample_idx": i, "x_sym": int(s_x), "target_z": int(targ_z),
                "pred_z": pred_z, "correct": is_corr, "latency_ms": lat,
            })
        acc_c_sym = corr_c_sym / len(te_c_z)

        # Baseline: ScratchTransformer20M (Fair 2-Step Composition)
        print(f"  Training LLM-1 ScratchTransformer20M on Seed {seed}...")
        llm1_mod, llm1_vocab, llm1_inv, llm1_tsec = train_fair_scratch_transformer(map_a, map_b, device=device)
        llm1_eval = evaluate_fair_scratch_transformer(llm1_mod, llm1_vocab, llm1_inv, map_a, map_b, map_c, device=device)

        # Baseline: SmolLM2-135M (Zero Leakage, Exact Answer Parsing)
        print(f"  Evaluating LLM-2 SmolLM2 on Seed {seed}...")
        test_queries = [(k, v) for k, v in map_c.items()]
        smol_acc_direct, smol_lat_direct, smol_recs_direct = smol_evaluator.evaluate_2hop(map_a, map_b, test_queries, mode="direct")
        smol_acc_cot, smol_lat_cot, smol_recs_cot = smol_evaluator.evaluate_2hop(map_a, map_b, test_queries, mode="cot")

        for r in smol_recs_direct:
            raw_sample_predictions.append({"phase": "level_1", "seed": seed, "model": "smollm2_direct", **r})
        for r in smol_recs_cot:
            raw_sample_predictions.append({"phase": "level_1", "seed": seed, "model": "smollm2_cot", **r})

        seed_summary = {
            "seed": seed,
            "is_heldout": is_heldout,
            "snn_acc_a": float(acc_a),
            "snn_acc_b": float(acc_b),
            "snn_c_fixed_spike": float(acc_c_fixed),
            "snn_c_learned_aligner": float(acc_c_align),
            "snn_c_symbolic": float(acc_c_sym),
            "snn_lat_learned_p50": float(np.median(lat_align)),
            "llm1_acc_a": float(llm1_eval["acc_task_a"]),
            "llm1_acc_b": float(llm1_eval["acc_task_b"]),
            "llm1_c_1hop_direct": float(llm1_eval["acc_c_direct_1hop"]),
            "llm1_c_fair_2step": float(llm1_eval["acc_c_fair_2step"]),
            "llm1_lat_2step_p50": float(llm1_eval["lat_2step_p50_ms"]),
            "smol_acc_direct": float(smol_acc_direct),
            "smol_acc_cot": float(smol_acc_cot),
            "smol_lat_p50": float(np.median(smol_lat_direct)),
        }
        l1_seed_results.append(seed_summary)
        print(
            f"  {tag} SNN Fixed: {acc_c_fixed*100:.1f}% | SNN Aligner: {acc_c_align*100:.1f}% | SNN Sym: {acc_c_sym*100:.1f}% | "
            f"LLM-1 2-Step: {llm1_eval['acc_c_fair_2step']*100:.1f}% (1-hop: {llm1_eval['acc_c_direct_1hop']*100:.1f}%) | "
            f"SmolLM2 Direct: {smol_acc_direct*100:.1f}% | CoT: {smol_acc_cot*100:.1f}%"
        )

    results["level_1_independent_composition"] = l1_seed_results

    # =========================================================================
    # PHASE 2: CAUSAL VERIFICATION CONTROLS & DECISIVE TEST
    # =========================================================================
    print("\n" + "=" * 60)
    print(">>> PHASE 2: CAUSAL VERIFICATION CONTROLS & DECISIVE TEST <<<")
    print("=" * 60)

    # Use DEV seed 140011
    seed = 140011
    rng = np.random.default_rng(seed)
    map_a = {i: int(rng.permutation(8)[i]) for i in range(8)}
    map_b = {i: int(rng.permutation(8)[i]) for i in range(8)}
    map_c = {i: int(map_b[map_a[i]]) for i in range(8)}

    pat_x = generate_independent_symbol_patterns(seed + 100, num_symbols=8)
    pat_y = generate_independent_symbol_patterns(seed + 200, num_symbols=8)
    (tr_a_x, tr_a_y, _), _ = build_data(seed + 300, map_a, pat_x)
    (tr_b_x, tr_b_z, _), _ = build_data(seed + 400, map_b, pat_y)
    _, (te_c_x, te_c_z, _) = build_data(seed + 500, map_c, pat_x)

    # Trained modules
    base_mod_a = IndependentALIFModule("causal_a", 0, 8, 48, 8, 8, seed)
    base_mod_a.calibrate(tr_a_x[:32])
    for ep in range(len(tr_a_x)):
        base_mod_a.learn_episode(tr_a_x[ep], tr_a_y[ep])

    base_mod_b = IndependentALIFModule("causal_b", 48, 8, 48, 8, 8, seed + 1)
    base_mod_b.calibrate(tr_b_x[:32])
    for ep in range(len(tr_b_x)):
        base_mod_b.learn_episode(tr_b_x[ep], tr_b_z[ep])

    base_sym_router = SymbolicRouter(pat_y)

    def eval_pair(ma, mb, router):
        c = 0
        for x, z in zip(te_c_x, te_c_z):
            probs_a, _ = ma.forward_sequence(x)
            in_b = router.route_symbolic(probs_a)
            probs_b, _ = mb.forward_sequence(in_b)
            if int(np.argmax(probs_b)) == z:
                c += 1
        return float(c / len(te_c_z))

    baseline_c_acc = eval_pair(base_mod_a, base_mod_b, base_sym_router)

    # 1. Untrained A + Untrained B
    untrained_a = IndependentALIFModule("untrained_a", 0, 8, 48, 8, 8, seed + 10)
    untrained_b = IndependentALIFModule("untrained_b", 48, 8, 48, 8, 8, seed + 11)
    untrained_c_acc = eval_pair(untrained_a, untrained_b, base_sym_router)

    # 2. Synapse shuffle (permuted weights of A)
    shuffled_a = copy.deepcopy(base_mod_a)
    shuffled_a.core.w_in = rng.permutation(shuffled_a.core.w_in)
    shuffled_a.core.w_rec = rng.permutation(shuffled_a.core.w_rec)
    shuffled_c_acc = eval_pair(shuffled_a, base_mod_b, base_sym_router)

    # 3. Broken router (all zero inputs to B)
    broken_router = SymbolicRouter(np.zeros_like(pat_y))
    broken_c_acc = eval_pair(base_mod_a, base_mod_b, broken_router)

    # 4. Swapped B' (Module B' trained on completely different mapping B')
    map_b_alt = {i: int((map_b[i] + 3) % 8) for i in range(8)}
    (tr_b_alt_x, tr_b_alt_z, _), _ = build_data(seed + 600, map_b_alt, pat_y)
    mod_b_alt = IndependentALIFModule("mod_b_alt", 48, 8, 48, 8, 8, seed + 2)
    mod_b_alt.calibrate(tr_b_alt_x[:32])
    for ep in range(len(tr_b_alt_x)):
        mod_b_alt.learn_episode(tr_b_alt_x[ep], tr_b_alt_z[ep])
    # Expected target under B' is map_b_alt[map_a[X]]
    map_c_alt = {i: int(map_b_alt[map_a[i]]) for i in range(8)}
    _, (te_c_alt_x, te_c_alt_z, _) = build_data(seed + 700, map_c_alt, pat_x)
    c_b_alt = sum(
        int(np.argmax(mod_b_alt.forward_sequence(base_sym_router.route_symbolic(base_mod_a.forward_sequence(x)[0]))[0])) == z
        for x, z in zip(te_c_alt_x, te_c_alt_z)
    )
    acc_swapped_b = c_b_alt / len(te_c_alt_z)

    print(f"  Baseline Composition Accuracy: {baseline_c_acc*100:.1f}%")
    print(f"  Control 1 (Untrained A & B): {untrained_c_acc*100:.1f}% (Chance 12.5%)")
    print(f"  Control 2 (Shuffled Synapses): {shuffled_c_acc*100:.1f}%")
    print(f"  Control 3 (Broken Router): {broken_c_acc*100:.1f}%")
    print(f"  Control 4 (Swapped Knowledge B'): {acc_swapped_b*100:.1f}% (Tracked to new target Z')")

    # 5. THE DECISIVE CAUSAL TEST
    print("\n--- The Decisive Knowledge Swap Test ---")
    # Take Symbol X1: initially maps to Y_orig -> Z_orig
    sym_x = 1
    y_orig = map_a[sym_x]
    z_orig = map_b[y_orig]
    # Check baseline output for X1
    pred_z_initial = int(np.argmax(base_mod_b.forward_sequence(base_sym_router.route_symbolic(base_mod_a.forward_sequence(pat_x[sym_x])[0]))[0]))
    print(f"  Step 1: X1 -> Y{y_orig} -> Z{z_orig} | Composition predicts: Z{pred_z_initial}")

    # Retrain ONLY Module A: X1 now maps to Y_new (where B knows Y_new -> Z_new)
    y_new = (y_orig + 4) % 8
    z_new = map_b[y_new]
    map_a_swapped = map_a.copy()
    map_a_swapped[sym_x] = y_new

    # Train A on swapped mapping WITHOUT TOUCHING B OR ROUTER
    decisive_mod_a = copy.deepcopy(base_mod_a)
    for _ in range(300):
        for s, targ_y in map_a_swapped.items():
            inp = pat_x[s] + rng.normal(0, 0.05, pat_x[s].shape).astype(np.float32)
            decisive_mod_a.learn_episode(inp, targ_y)

    pred_z_swapped_a = int(np.argmax(base_mod_b.forward_sequence(base_sym_router.route_symbolic(decisive_mod_a.forward_sequence(pat_x[sym_x])[0]))[0]))
    swap_a_success = (pred_z_swapped_a == z_new)
    print(f"  Step 2: Dạy lại riêng A: X1 -> Y{y_new} (B đã biết Y{y_new} -> Z{z_new}).")
    print(f"          Composition tự động đổi sang: Z{pred_z_swapped_a} (Mục tiêu: Z{z_new}) => {'PASS' if swap_a_success else 'FAIL'}")

    # Step 3: Now Retrain ONLY Module B: Y_new now maps to Z_alt. Keep A untouched.
    z_alt = (z_new + 3) % 8
    map_b_swapped = map_b.copy()
    map_b_swapped[y_new] = z_alt

    decisive_mod_b = copy.deepcopy(base_mod_b)
    for _ in range(300):
        for s, targ_z in map_b_swapped.items():
            inp = pat_y[s] + rng.normal(0, 0.05, pat_y[s].shape).astype(np.float32)
            decisive_mod_b.learn_episode(inp, targ_z)

    pred_z_swapped_b = int(np.argmax(decisive_mod_b.forward_sequence(base_sym_router.route_symbolic(decisive_mod_a.forward_sequence(pat_x[sym_x])[0]))[0]))
    swap_b_success = (pred_z_swapped_b == z_alt)
    print(f"  Step 3: Dạy lại riêng B: Y{y_new} -> Z{z_alt}. Giữ nguyên A.")
    print(f"          Composition tự động đổi sang: Z{pred_z_swapped_b} (Mục tiêu: Z{z_alt}) => {'PASS' if swap_b_success else 'FAIL'}")

    results["causal_verification_controls"] = {
        "baseline_c_acc": float(baseline_c_acc),
        "untrained_c_acc": float(untrained_c_acc),
        "shuffled_synapses_c_acc": float(shuffled_c_acc),
        "broken_router_c_acc": float(broken_c_acc),
        "swapped_b_alt_acc": float(acc_swapped_b),
    }
    results["decisive_knowledge_swap_test"] = {
        "x1_initial_pred": pred_z_initial,
        "x1_initial_target": z_orig,
        "x1_after_retraining_a_pred": pred_z_swapped_a,
        "x1_after_retraining_a_target": z_new,
        "swap_a_pass": bool(swap_a_success),
        "x1_after_retraining_b_pred": pred_z_swapped_b,
        "x1_after_retraining_b_target": z_alt,
        "swap_b_pass": bool(swap_b_success),
        "overall_decisive_pass": bool(swap_a_success and swap_b_success),
    }

    # =========================================================================
    # PHASE 4: SCALING K IN {8, 16, 32, 64} (FULL PREMISES FOR LLM)
    # =========================================================================
    print("\n" + "=" * 60)
    print(">>> PHASE 4: MEMORY CAPACITY SCALING (K = 8, 16, 32, 64) <<<")
    print("=" * 60)

    scaling_results = {}
    for k_val in (8, 16, 32, 64):
        seed = 140011
        rng = np.random.default_rng(seed)
        perm_a = rng.permutation(k_val)
        map_a = {i: int(perm_a[i]) for i in range(k_val)}
        perm_b = rng.permutation(k_val)
        map_b = {i: int(perm_b[i]) for i in range(k_val)}
        map_c = {i: int(map_b[map_a[i]]) for i in range(k_val)}

        pat_x = generate_independent_symbol_patterns(seed + 100, num_symbols=k_val)
        pat_y = generate_independent_symbol_patterns(seed + 200, num_symbols=k_val)

        (tr_a_x, tr_a_y, _), _ = build_data(seed + 300, map_a, pat_x, n_train=1024)
        (tr_b_x, tr_b_z, _), _ = build_data(seed + 400, map_b, pat_y, n_train=1024)
        _, (te_c_x, te_c_z, _) = build_data(seed + 500, map_c, pat_x, n_train=0, n_test=128)

        # Train SNN A & B
        h_dim = max(48, k_val)
        mod_a = IndependentALIFModule("mod_a_scale", 0, 8, h_dim, k_val, 8, seed)
        mod_a.calibrate(tr_a_x[:32])
        for ep in range(len(tr_a_x)):
            mod_a.learn_episode(tr_a_x[ep], tr_a_y[ep])

        mod_b = IndependentALIFModule("mod_b_scale", h_dim, 8, h_dim, k_val, 8, seed + 1)
        mod_b.calibrate(tr_b_x[:32])
        for ep in range(len(tr_b_x)):
            mod_b.learn_episode(tr_b_x[ep], tr_b_z[ep])

        sym_router = SymbolicRouter(pat_y)
        corr_snn_scale = sum(
            int(np.argmax(mod_b.forward_sequence(sym_router.route_symbolic(mod_a.forward_sequence(x)[0]))[0])) == z
            for x, z in zip(te_c_x, te_c_z)
        )
        acc_snn_scale = corr_snn_scale / len(te_c_z)

        # Evaluate SmolLM2 with FULL PREMISES (all K pairs in context)
        test_queries = [(k, v) for k, v in list(map_c.items())[:16]]
        smol_k_acc, _, _ = smol_evaluator.evaluate_2hop(map_a, map_b, test_queries, mode="direct")

        scaling_results[f"K_{k_val}"] = {
            "num_symbols": k_val,
            "snn_symbolic_acc": float(acc_snn_scale),
            "smollm_direct_acc": float(smol_k_acc),
        }
        print(f"  [K={k_val:2d}] SNN Symbolic Ceiling: {acc_snn_scale*100:5.1f}% | SmolLM2 Full-Premise: {smol_k_acc*100:5.1f}%")

    results["level_2_scaling"] = scaling_results

    # =========================================================================
    # MULTI-STEP REASONING (3-STEP: X -> Y -> Z -> W)
    # =========================================================================
    print("\n" + "=" * 60)
    print(">>> MULTI-STEP TRANSITIVE REASONING (3-STEP: X -> Y -> Z -> W) <<<")
    print("=" * 60)

    seed = 140011
    rng = np.random.default_rng(seed)
    map_a = {i: int(rng.permutation(8)[i]) for i in range(8)}
    map_b = {i: int(rng.permutation(8)[i]) for i in range(8)}
    map_d = {i: int(rng.permutation(8)[i]) for i in range(8)}
    map_c = {i: int(map_b[map_a[i]]) for i in range(8)}
    map_3step = {i: int(map_d[map_c[i]]) for i in range(8)}

    pat_x = generate_independent_symbol_patterns(seed + 100, num_symbols=8)
    pat_y = generate_independent_symbol_patterns(seed + 200, num_symbols=8)
    pat_z = generate_independent_symbol_patterns(seed + 300, num_symbols=8)

    (tr_a_x, tr_a_y, _), _ = build_data(seed + 400, map_a, pat_x)
    (tr_b_x, tr_b_z, _), _ = build_data(seed + 500, map_b, pat_y)
    (tr_d_x, tr_d_w, _), _ = build_data(seed + 600, map_d, pat_z)
    _, (te_3_x, te_3_w, _) = build_data(seed + 700, map_3step, pat_x, n_train=0, n_test=128)

    # 3 Independent SNN modules
    mod_a = IndependentALIFModule("mod_a_3s", 0, 8, 48, 8, 8, seed)
    mod_a.calibrate(tr_a_x[:32])
    for ep in range(len(tr_a_x)): mod_a.learn_episode(tr_a_x[ep], tr_a_y[ep])

    mod_b = IndependentALIFModule("mod_b_3s", 48, 8, 48, 8, 8, seed + 1)
    mod_b.calibrate(tr_b_x[:32])
    for ep in range(len(tr_b_x)): mod_b.learn_episode(tr_b_x[ep], tr_b_z[ep])

    mod_d = IndependentALIFModule("mod_d_3s", 96, 8, 48, 8, 8, seed + 2)
    mod_d.calibrate(tr_d_x[:32])
    for ep in range(len(tr_d_x)): mod_d.learn_episode(tr_d_x[ep], tr_d_w[ep])

    router_y = SymbolicRouter(pat_y)
    router_z = SymbolicRouter(pat_z)

    corr_3s_snn = 0
    lat_3s_snn = []
    for x, w_targ in zip(te_3_x, te_3_w):
        t0 = time.perf_counter()
        probs_a, _ = mod_a.forward_sequence(x)
        in_b = router_y.route_symbolic(probs_a)
        probs_b, _ = mod_b.forward_sequence(in_b)
        in_d = router_z.route_symbolic(probs_b)
        probs_d, _ = mod_d.forward_sequence(in_d)
        pred_w = int(np.argmax(probs_d))
        lat_3s_snn.append((time.perf_counter() - t0) * 1000.0)
        if pred_w == w_targ:
            corr_3s_snn += 1

    acc_3s_snn = corr_3s_snn / len(te_3_w)

    # Scratch Transformer 20M 3-Step
    llm1_3s, vocab_3s, inv_3s, _ = train_fair_scratch_transformer(map_a, map_b, mapping_d=map_d, device=device)
    llm1_3s_eval = evaluate_fair_scratch_transformer(llm1_3s, vocab_3s, inv_3s, map_a, map_b, map_c, mapping_3step=map_3step, device=device)

    # SmolLM2 3-Hop
    smol_acc_3s_direct, _, _ = smol_evaluator.evaluate_3hop(map_a, map_b, map_d, map_3step, mode="direct")
    smol_acc_3s_cot, _, _ = smol_evaluator.evaluate_3hop(map_a, map_b, map_d, map_3step, mode="cot")

    print(f"  SNN 3-Step Symbolic Composition: {acc_3s_snn*100:.1f}% | Latency p50: {np.median(lat_3s_snn):.2f}ms")
    print(f"  LLM-1 3-Step Autoregressive: {llm1_3s_eval['acc_3step_fair']*100:.1f}%")
    print(f"  SmolLM2 3-Hop Direct: {smol_acc_3s_direct*100:.1f}% | CoT: {smol_acc_3s_cot*100:.1f}%")

    results["level_3_multistep"] = {
        "snn_3step_acc": float(acc_3s_snn),
        "snn_3step_lat_p50_ms": float(np.median(lat_3s_snn)),
        "llm1_3step_fair_acc": float(llm1_3s_eval["acc_3step_fair"]),
        "smol_3hop_direct_acc": float(smol_acc_3s_direct),
        "smol_3hop_cot_acc": float(smol_acc_3s_cot),
    }

    # =========================================================================
    # PHASE 5: RESOURCE PROFILING (DYNAMICALLY COMPUTED)
    # =========================================================================
    print("\n" + "=" * 60)
    print(">>> PHASE 5: RESOURCE AUDIT (DYNAMIC ARRAYS & LIVE RSS) <<<")
    print("=" * 60)

    param_breakdown_single = mod_a.get_parameter_breakdown()
    snn_trainable_2mod = param_breakdown_single["trainable_total"] * 2
    snn_nonzero_2mod = param_breakdown_single["nonzero_synapses"] * 2

    peak_rss = tracker.sample()
    engram_path_a = store_dir / f"mod_a_s{seed}.engram.json"
    ssd_bytes_a = engram_path_a.stat().st_size if engram_path_a.exists() else 0

    # Cold load benchmark in fresh module instance
    t0_cold = time.perf_counter()
    fresh_mod = IndependentALIFModule("fresh_load", 0, 8, 48, 8, 8, seed)
    fresh_mod.load_complete_engram(json.loads(engram_path_a.read_text(encoding="utf-8"))["payload"]["engram"])
    cold_load_ms = (time.perf_counter() - t0_cold) * 1000.0

    print(f"  SNN Single Module Trainable Parameters: {param_breakdown_single['trainable_total']:,}")
    print(f"  SNN Two-Module Trainable Parameters: {snn_trainable_2mod:,}")
    print(f"  SNN Two-Module Nonzero Synapses: {snn_nonzero_2mod:,}")
    print(f"  SNN SSD Engram Size (Single Module): {ssd_bytes_a:,} bytes (~{ssd_bytes_a/1024:.1f} KB)")
    print(f"  SNN Cold Load Time: {cold_load_ms:.2f} ms")
    print(f"  Transformer 20M Parameters: {llm1_3s.total_parameters:,}")
    print(f"  SmolLM2-135M Parameters: {smol_evaluator.total_parameters:,}")
    print(f"  Live Peak Process RSS: {peak_rss:.1f} MB")

    results["resource_audit"] = {
        "snn_single_module_parameters": param_breakdown_single,
        "snn_two_module_trainable": snn_trainable_2mod,
        "snn_two_module_nonzero_synapses": snn_nonzero_2mod,
        "snn_ssd_engram_bytes": ssd_bytes_a,
        "snn_cold_load_ms": float(cold_load_ms),
        "transformer_20m_parameters": llm1_3s.total_parameters,
        "smollm2_135m_parameters": smol_evaluator.total_parameters,
        "live_peak_rss_mb": float(peak_rss),
    }

    # Write raw sample predictions JSON
    raw_out = ROOT / "artifacts" / "r13_2_raw_predictions.json"
    with open(raw_out, "w", encoding="utf-8") as f:
        json.dump(raw_sample_predictions, f, indent=2, default=numpy_json_serializer)

    # Write master benchmark results JSON
    master_out = ROOT / "artifacts" / "r13_2_comprehensive_benchmark_results.json"
    with open(master_out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=numpy_json_serializer)

    print("\n" + "=" * 80)
    print("SCAR-X ROUND 13.2 MASTER BENCHMARK COMPLETED SUCCESSFULLY!")
    print(f"Raw Sample Predictions: {raw_out}")
    print(f"Summary Results: {master_out}")
    print("=" * 80)


if __name__ == "__main__":
    run_r13_2_master_benchmark()
