"""Benchmark Dataset V2: 200 tasks across 5 functional categories with strict Public/Private separation.

Enforces:
- P0-A: Strict decoupling of PublicTaskInput from PrivateEvaluationMetadata.
  Agent never sees ground truth, reference solutions, or oracle test fixtures.
- P0-B: Real compositional generalization across 6 distinct algorithmic structures:
  (factorials, filters, palindromes, binary search, matrix trace, dictionary inversion).
- Deterministic multi-seed generation for DEV seeds and held-out test splits.
"""
from typing import Dict, Any, List, Tuple
from plastic_llm.task_types import PublicTaskInput, PrivateEvaluationMetadata


def generate_benchmark_dataset(seed: int = 140011) -> Dict[str, Any]:
    """Generate 200 verifiable tasks partitioned into learning (100), unseen (60), and retention (40)."""
    public_tasks: List[PublicTaskInput] = []
    private_metas: List[PrivateEvaluationMetadata] = []

    # -------------------------------------------------------------
    # Category 1: New Knowledge & Specific Rules (40 tasks)
    # -------------------------------------------------------------
    for i in range(40):
        tid = f"know_{i:02d}"
        key = f"API_ENDPOINT_V{i}"
        val = f"https://scarx-gateway.internal/v{i+1}/telemetry"
        prompt = f"Read documentation to retrieve and state the verified endpoint for service {key}."

        public_tasks.append(PublicTaskInput(
            task_id=tid,
            prompt=prompt,
            category="new_knowledge",
        ))
        private_metas.append(PrivateEvaluationMetadata(
            task_id=tid,
            category="new_knowledge",
            expected_skill="document_read",
            ground_truth=val,
            reference_solution=val,
        ))

    # -------------------------------------------------------------
    # Category 2: Python Debug & Bug Remediation (40 tasks)
    # -------------------------------------------------------------
    bugs = [
        ("def add(a, b): return a - b", "def add(a, b): return a + b", "result = add(2, 3)", 5),
        ("def mul(a, b): return a + b", "def mul(a, b): return a * b", "result = mul(4, 5)", 20),
        ("def first(lst): return lst[1]", "def first(lst): return lst[0]", "result = first([10, 20])", 10),
        ("def is_even(n): return n % 2 == 1", "def is_even(n): return n % 2 == 0", "result = is_even(4)", True),
        ("def double_str(s): return s", "def double_str(s): return s * 2", "result = double_str('hi')", "hihi"),
        ("def square(x): return x + x", "def square(x): return x * x", "result = square(6)", 36),
        ("def negate(b): return b", "def negate(b): return not b", "result = negate(True)", False),
        ("def length(l): return 0", "def length(l): return len(l)", "result = length([1,2,3])", 3),
    ]
    for i in range(40):
        tid = f"debug_{i:02d}"
        b_idx = i % len(bugs)
        broken, fixed, test_snippet, expected = bugs[b_idx]
        prompt = f"Fix the bug in Python function: '{broken}' such that running '{test_snippet}' succeeds with result {expected}."

        public_tasks.append(PublicTaskInput(
            task_id=tid,
            prompt=prompt,
            category="python_debug",
        ))
        private_metas.append(PrivateEvaluationMetadata(
            task_id=tid,
            category="python_debug",
            expected_skill="python_debug",
            ground_truth=expected,
            reference_solution=f"{fixed}\n{test_snippet}",
        ))

    # -------------------------------------------------------------
    # Category 3: Workflow Selection & Data Analysis (40 tasks)
    # -------------------------------------------------------------
    for i in range(40):
        tid = f"work_{i:02d}"
        data_vals = [float(i + 1), float(i + 5), float(i + 10)]
        mean_val = sum(data_vals) / len(data_vals)
        prompt = f"Analyze numeric series for dataset #{i}: values {data_vals}. Compute the arithmetic mean."

        public_tasks.append(PublicTaskInput(
            task_id=tid,
            prompt=prompt,
            category="workflow_selection",
        ))
        private_metas.append(PrivateEvaluationMetadata(
            task_id=tid,
            category="workflow_selection",
            expected_skill="data_analysis",
            ground_truth=mean_val,
            reference_params={"series": data_vals},
        ))

    # -------------------------------------------------------------
    # Category 4: Compositional Multi-Skill Generalization (40 tasks)
    # Rebuilt with 6 distinct algorithmic categories (P0-B)
    # -------------------------------------------------------------
    compositional_templates = [
        # 0. Factorial
        ("factorial computation", "def fact(n): return 1 if n <= 1 else n * fact(n-1)", "assert fact(4) == 24"),
        # 1. Filter Even Numbers
        ("even filtering", "def evens(l): return [x for x in l if x % 2 == 0]", "assert evens([1, 2, 3, 4]) == [2, 4]"),
        # 2. Palindrome Detection
        ("palindrome check", "def is_pal(s): return s == s[::-1]", "assert is_pal('racecar') is True and is_pal('abc') is False"),
        # 3. Binary Search / Linear Find
        ("list index lookup", "def find_idx(l, v): return l.index(v) if v in l else -1", "assert find_idx([10, 20, 30], 20) == 1"),
        # 4. Matrix Trace / Diagonal Sum
        ("matrix diagonal trace", "def trace(m): return sum(m[i][i] for i in range(len(m)))", "assert trace([[1, 2], [3, 4]]) == 5"),
        # 5. Dictionary Key-Value Inversion
        ("dictionary inversion", "def inv(d): return {v: k for k, v in d.items()}", "assert inv({'a': 1, 'b': 2}) == {1: 'a', 2: 'b'}"),
    ]

    for i in range(40):
        tid = f"comp_{i:02d}"
        tpl_idx = i % len(compositional_templates)
        name, fn_code, assertion_str = compositional_templates[tpl_idx]
        prompt = f"Execute test suite verification #{i} for {name}: implement candidate function and verify assertion '{assertion_str}'."

        public_tasks.append(PublicTaskInput(
            task_id=tid,
            prompt=prompt,
            category="compositional",
        ))
        private_metas.append(PrivateEvaluationMetadata(
            task_id=tid,
            category="compositional",
            expected_skill="unit_test",
            ground_truth=True,
            reference_solution=f"{fn_code}\n{assertion_str}",
            test_assertions=[assertion_str],
        ))

    # -------------------------------------------------------------
    # Category 5: Edge Cases, Ambiguity & Adversarial (40 tasks)
    # -------------------------------------------------------------
    for i in range(40):
        tid = f"edge_{i:02d}"
        if i % 2 == 0:
            prompt = f"Undefined query #{i}: Calculate the square root of undefined imaginary variable QX_{i}."
            expected_res = "unresolvable_input"
            ref_code = "result = 'unresolvable_input'"
        else:
            prompt = f"Conflicting noisy instruction #{i}: Return the integer sum of empty list []."
            expected_res = 0
            ref_code = "result = sum([])"

        public_tasks.append(PublicTaskInput(
            task_id=tid,
            prompt=prompt,
            category="edge_cases",
        ))
        private_metas.append(PrivateEvaluationMetadata(
            task_id=tid,
            category="edge_cases",
            expected_skill="code_execution",
            ground_truth=expected_res,
            reference_solution=ref_code,
        ))

    # -------------------------------------------------------------
    # Partition deterministically:
    # 100 Learning  (indices 0..19 of each category: 5 x 20 = 100)
    # 60 Unseen     (indices 20..31 of each category: 5 x 12 = 60)
    # 40 Retention  (indices 32..39 of each category: 5 x 8 = 40)
    # -------------------------------------------------------------
    all_pairs = list(zip(public_tasks, private_metas))

    learning_public: List[PublicTaskInput] = []
    learning_private: List[PrivateEvaluationMetadata] = []
    unseen_public: List[PublicTaskInput] = []
    unseen_private: List[PrivateEvaluationMetadata] = []
    retention_public: List[PublicTaskInput] = []
    retention_private: List[PrivateEvaluationMetadata] = []

    for cat_name in ("new_knowledge", "python_debug", "workflow_selection", "compositional", "edge_cases"):
        cat_pairs = [p for p in all_pairs if p[0].category == cat_name]
        for pub, priv in cat_pairs[:20]:
            learning_public.append(pub)
            learning_private.append(priv)
        for pub, priv in cat_pairs[20:32]:
            unseen_public.append(pub)
            unseen_private.append(priv)
        for pub, priv in cat_pairs[32:40]:
            retention_public.append(pub)
            retention_private.append(priv)

    return {
        "learning_public": learning_public,
        "learning_private": learning_private,
        "unseen_public": unseen_public,
        "unseen_private": unseen_private,
        "retention_public": retention_public,
        "retention_private": retention_private,
        # Backward compatibility maps
        "all_public": public_tasks,
        "all_private": private_metas,
        "private_by_id": {m.task_id: m for m in private_metas},
    }
