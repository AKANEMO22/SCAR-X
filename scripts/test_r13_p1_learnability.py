"""SCAR-X Round 13: Phase 1 Learnability Benchmark for Task A and Task B."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from scarx.neural.r13_modular_alif import ModularALIFModule


def generate_orthogonal_symbol_patterns(seed: int, num_symbols: int = 8, seq_len: int = 20, dim: int = 8) -> np.ndarray:
    """Generate distinct temporal burst signatures for each symbol."""
    rng = np.random.default_rng(seed)
    patterns = np.zeros((num_symbols, seq_len, dim), dtype=np.float32)
    for s in range(num_symbols):
        # Distinct multi-channel burst
        ch1 = s % dim
        ch2 = (s + 3) % dim
        patterns[s, 3:8, ch1] = 1.0
        patterns[s, 5:10, ch2] = 0.8
        # Query / context trigger at ticks 15-18
        patterns[s, 15:18, :] += 0.2
    return patterns


def build_symbolic_task_data(
    seed: int,
    mapping: dict,
    symbol_patterns: np.ndarray,
    n_train: int = 1024,
    n_test: int = 256,
    noise_std: float = 0.05
):
    rng = np.random.default_rng(seed)
    num_symbols = len(mapping)

    def sample_set(n):
        xs, ys = [], []
        for _ in range(n):
            sym_in = int(rng.integers(0, num_symbols))
            sym_out = int(mapping[sym_in])
            pat = symbol_patterns[sym_in].copy()
            pat += rng.normal(0, noise_std, pat.shape).astype(np.float32)
            xs.append(pat)
            ys.append(sym_out)
        return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.int64)

    train_x, train_y = sample_set(n_train)
    test_x, test_y = sample_set(n_test)
    return (train_x, train_y), (test_x, test_y)


def evaluate_module(module: ModularALIFModule, test_x: np.ndarray, test_y: np.ndarray) -> float:
    correct = 0
    for x, y in zip(test_x, test_y):
        probs, _ = module.forward_sequence(x)
        pred = int(np.argmax(probs))
        if pred == int(y):
            correct += 1
    return correct / len(test_y)


def run_learnability_test():
    seed = 140011
    rng = np.random.default_rng(seed)

    # 1. Random bijective mappings (8-way, chance = 12.5%)
    perm_a = rng.permutation(8)
    mapping_a = {i: int(perm_a[i]) for i in range(8)}

    perm_b = rng.permutation(8)
    mapping_b = {i: int(perm_b[i]) for i in range(8)}

    print(f"Mapping A (X -> Y): {mapping_a}")
    print(f"Mapping B (Y -> Z): {mapping_b}")

    # Generate patterns
    patterns_x = generate_orthogonal_symbol_patterns(seed + 100)
    patterns_y = generate_orthogonal_symbol_patterns(seed + 200)

    (train_a_x, train_a_y), (test_a_x, test_a_y) = build_symbolic_task_data(
        seed + 300, mapping_a, patterns_x, n_train=1024, n_test=256
    )
    (train_b_x, train_b_y), (test_b_x, test_b_y) = build_symbolic_task_data(
        seed + 400, mapping_b, patterns_y, n_train=1024, n_test=256
    )

    # Train Module A
    mod_a = ModularALIFModule("module_A", start_neuron_id=0, inputs=8, hidden=48, outputs=8, seed=seed)
    mod_a.calibrate(train_a_x[:32])

    print("\nTraining Module A on X -> Y...")
    for ep in range(1024):
        mod_a.learn_episode(train_a_x[ep], train_a_y[ep])

    acc_a = evaluate_module(mod_a, test_a_x, test_a_y)
    print(f"Module A Test Accuracy (Chance 12.5%): {acc_a * 100:.2f}%")

    # Train Module B
    mod_b = ModularALIFModule("module_B", start_neuron_id=48, inputs=8, hidden=48, outputs=8, seed=seed + 1)
    mod_b.calibrate(train_b_x[:32])

    print("\nTraining Module B on Y -> Z...")
    for ep in range(1024):
        mod_b.learn_episode(train_b_x[ep], train_b_y[ep])

    acc_b = evaluate_module(mod_b, test_b_x, test_b_y)
    print(f"Module B Test Accuracy (Chance 12.5%): {acc_b * 100:.2f}%")

    assert acc_a >= 0.80, f"Module A failed target (80%), got {acc_a*100:.2f}%"
    assert acc_b >= 0.80, f"Module B failed target (80%), got {acc_b*100:.2f}%"
    print("\n>>> PHASE 1 LEARNABILITY GATE: PASS (Both A and B >= 80%) <<<")


if __name__ == "__main__":
    run_learnability_test()
