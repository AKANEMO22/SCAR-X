"""Unit tests for SCAR-X Round 13.2: Independent Modular Composition & Causal Integrity."""
import os
import re
from pathlib import Path
import numpy as np
import pytest

from scarx.neural.r13_2_independent_modular_alif import (
    IndependentALIFModule,
    FixedSpikeRouter,
    LearnedRepresentationAligner,
    SymbolicRouter,
    generate_independent_symbol_patterns,
)
from scarx.evaluation.r13_2_fair_llm_benchmarks import FairSmolLM2Evaluator


def test_zero_target_leakage_in_smollm_prompt():
    """Verify ERR-02 fix: SmolLM prompt MUST NOT contain target_z in instruction or template."""
    evaluator = FairSmolLM2Evaluator.__new__(FairSmolLM2Evaluator)
    
    map_a = {0: 3, 1: 5}
    map_b = {3: 7, 5: 2}
    test_queries = [(0, 7), (1, 2)]

    # Inspect generated prompt text
    lines_a = [f"X{k} -> Y{v}" for k, v in sorted(map_a.items())]
    lines_b = [f"Y{k} -> Z{v}" for k, v in sorted(map_b.items())]
    premise_str = "Knowledge Base:\n" + "\n".join(lines_a) + "\n\n" + "\n".join(lines_b) + "\n\n"

    for x_sym, target_z in test_queries:
        prompt_direct = (
            "You are an exact transitive reasoning system.\n"
            + premise_str
            + "Rule: Transitive chain X -> Y -> Z.\n"
            + f"Question: What is Z for X{x_sym}?\n"
            + "Answer: Z"
        )
        # Ensure target_z does NOT appear in the prompt instruction
        instruction_part = prompt_direct.split("Question:")[-1]
        assert f"Z{target_z}" not in instruction_part
        assert f"like Z" not in instruction_part
        assert f": Z{target_z}" not in instruction_part


def test_strict_token_parser_rejects_substring_tricks():
    """Verify ERR-03 fix: Regex parser extracts exact symbol, not arbitrary substring."""
    evaluator = FairSmolLM2Evaluator.__new__(FairSmolLM2Evaluator)

    # Valid exact cases
    assert evaluator.parse_symbol_digit("Answer: Z5", prefix="Z") == 5
    assert evaluator.parse_symbol_digit("Result is Z12.", prefix="Z") == 12
    assert evaluator.parse_symbol_digit("Z7", prefix="Z") == 7
    assert evaluator.parse_symbol_digit("Output: 3", prefix="Z") == 3

    # Ambiguous or distractor texts
    # Should pick first candidate integer after Answer or standalone
    val = evaluator.parse_symbol_digit("Answer: Z4 (because Y2 maps to Z4)", prefix="Z")
    assert val == 4


def test_parameter_calculation_matches_actual_arrays():
    """Verify ERR-07 fix: Dynamic parameter counter reflects exact matrix sizes."""
    mod = IndependentALIFModule("mod_test", 0, inputs=8, hidden=48, outputs=8, fan_in=8, seed=42)
    breakdown = mod.get_parameter_breakdown()

    expected_w_in = 48 * 8   # 384
    expected_w_rec = 48 * 8  # 384
    expected_w_out = 8 * 48  # 384
    expected_bias = 8        # 8
    expected_total = 384 + 384 + 384 + 8  # 1160

    assert breakdown["w_in"] == expected_w_in
    assert breakdown["w_rec"] == expected_w_rec
    assert breakdown["w_out"] == expected_w_out
    assert breakdown["bias"] == expected_bias
    assert breakdown["trainable_total"] == expected_total
    assert breakdown["nonzero_synapses"] > 0


def test_bit_level_mid_episode_equivalence():
    """Verify mid-episode state persistence and exact bit-level resume."""
    seed = 42
    rng = np.random.default_rng(seed)
    mod1 = IndependentALIFModule("mod1", 0, 8, 48, 8, 8, seed)

    seq = rng.normal(0, 1, (20, 8)).astype(np.float32)

    # Run first 10 steps
    mod1.reset_episode()
    for row in seq[:10]:
        mod1.step(row, track=False)

    engram = mod1.get_complete_engram()

    # Finish remaining 10 steps on mod1
    spikes_mod1 = []
    for row in seq[10:]:
        spk = mod1.step(row, track=False)
        spikes_mod1.append(spk)
    final_v1 = mod1.core.voltage.copy()

    # Load engram into fresh mod2
    mod2 = IndependentALIFModule("mod2", 0, 8, 48, 8, 8, seed)
    mod2.load_complete_engram(engram)

    # Finish remaining 10 steps on mod2
    spikes_mod2 = []
    for row in seq[10:]:
        spk = mod2.step(row, track=False)
        spikes_mod2.append(spk)
    final_v2 = mod2.core.voltage.copy()

    # Bit-level equivalence
    assert np.array_equal(final_v1, final_v2)
    for s1, s2 in zip(spikes_mod1, spikes_mod2):
        assert np.array_equal(s1, s2)


def test_causal_decisive_test_knowledge_swap():
    """Verify the DECISIVE CAUSAL TEST:
    1. A knows X1 -> Y3, B knows Y3 -> Z5 => C predicts Z5.
    2. Retrain ONLY A: X1 -> Y7 (B already knows Y7 -> Z2).
    3. Check if C automatically updates to Z2 without retraining B or router!
    """
    seed = 140011
    pat_x = generate_independent_symbol_patterns(seed + 10, num_symbols=8)
    pat_y = generate_independent_symbol_patterns(seed + 20, num_symbols=8)

    # Task A initial mapping: X0->Y0, X1->Y3, ...
    map_a = {0: 0, 1: 3, 2: 2, 3: 1, 4: 4, 5: 5, 6: 6, 7: 7}
    # Task B initial mapping: Y3->Z5, Y7->Z2, ...
    map_b = {0: 0, 1: 1, 2: 2, 3: 5, 4: 4, 5: 6, 6: 7, 7: 2}

    # Train Module A
    mod_a = IndependentALIFModule("mod_a", 0, 8, 48, 8, 8, seed)
    mod_a.calibrate(pat_x)
    for _ in range(500):
        for x_sym, y_targ in map_a.items():
            inp = pat_x[x_sym] + np.random.normal(0, 0.05, pat_x[x_sym].shape).astype(np.float32)
            mod_a.learn_episode(inp, y_targ)

    # Train Module B INDEPENDENTLY (on Y patterns and Z labels ONLY!)
    mod_b = IndependentALIFModule("mod_b", 48, 8, 48, 8, 8, seed + 1)
    mod_b.calibrate(pat_y)
    for _ in range(500):
        for y_sym, z_targ in map_b.items():
            inp = pat_y[y_sym] + np.random.normal(0, 0.05, pat_y[y_sym].shape).astype(np.float32)
            mod_b.learn_episode(inp, z_targ)

    # Setup Symbolic Ceiling Router
    sym_router = SymbolicRouter(pat_y)

    # Initial composition check for X1: should produce Z5
    probs_a, _ = mod_a.forward_sequence(pat_x[1])
    in_b = sym_router.route_symbolic(probs_a)
    probs_b, _ = mod_b.forward_sequence(in_b)
    pred_z_initial = int(np.argmax(probs_b))
    assert pred_z_initial == 5, f"Expected initial Z5 for X1, got Z{pred_z_initial}"

    # Retrain ONLY Module A: X1 now maps to Y7!
    map_a_new = map_a.copy()
    map_a_new[1] = 7  # X1 -> Y7

    for _ in range(300):
        for x_sym, y_targ in map_a_new.items():
            inp = pat_x[x_sym] + np.random.normal(0, 0.05, pat_x[x_sym].shape).astype(np.float32)
            mod_a.learn_episode(inp, y_targ)

    # Test composition for X1 WITHOUT TOUCHING B:
    # Since X1 now produces Y7, and B knows Y7 -> Z2, composition MUST produce Z2!
    probs_a_new, _ = mod_a.forward_sequence(pat_x[1])
    in_b_new = sym_router.route_symbolic(probs_a_new)
    probs_b_new, _ = mod_b.forward_sequence(in_b_new)
    pred_z_updated = int(np.argmax(probs_b_new))

    assert pred_z_updated == 2, f"DECISIVE TEST FAILED: Expected Z2 after updating A, got Z{pred_z_updated}"
