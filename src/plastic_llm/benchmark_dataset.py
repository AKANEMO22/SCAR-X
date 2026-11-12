"""Benchmark Dataset: 200 tasks across 5 functional categories for Continual Learning evaluation.

Categories:
1. new_knowledge: 40 tasks testing retrieval and application of novel facts/APIs.
2. python_debug: 40 tasks testing syntax, runtime exceptions, and bug remediation.
3. workflow_selection: 40 tasks testing correct tool call decisions.
4. compositional: 40 tasks testing multi-skill combination on unseen tasks.
5. edge_cases: 40 tasks testing noisy, ambiguous, or unanswerable edge conditions.
"""
from typing import Dict, Any, List, Tuple
import re


def generate_benchmark_dataset(seed: int = 140011) -> Dict[str, List[Dict[str, Any]]]:
    """Generate 200 verifiable tasks split into learning (100), unseen (60), and retention (40)."""
    all_tasks = []

    # Category 1: New Knowledge & Specific Rules (40 tasks)
    for i in range(40):
        key = f"API_ENDPOINT_V{i}"
        val = f"https://scarx-gateway.internal/v{i+1}/telemetry"
        all_tasks.append({
            "task_id": f"know_{i:02d}",
            "category": "new_knowledge",
            "prompt": f"Read documentation to retrieve and state the verified endpoint for service {key}.",
            "expected_skill": "document_read",
            "knowledge_key": key,
            "knowledge_val": val,
            "ground_truth": val,
        })

    # Category 2: Python Debug & Execution (40 tasks)
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
        b_idx = i % len(bugs)
        broken, fixed, test_snippet, expected = bugs[b_idx]
        all_tasks.append({
            "task_id": f"debug_{i:02d}",
            "category": "python_debug",
            "prompt": f"Fix the bug in Python function: '{broken}' such that running '{test_snippet}' succeeds with result {expected}.",
            "expected_skill": "python_debug",
            "broken_code": broken,
            "fix_patch": f"{fixed}\n{test_snippet}",
            "ground_truth": expected,
        })

    # Category 3: Workflow Selection (40 tasks)
    for i in range(40):
        data_vals = [i + 1, i + 5, i + 10]
        mean_val = sum(data_vals) / len(data_vals)
        all_tasks.append({
            "task_id": f"work_{i:02d}",
            "category": "workflow_selection",
            "prompt": f"Analyze numeric series for dataset #{i}: values {data_vals}. Compute the arithmetic mean.",
            "expected_skill": "data_analysis",
            "params": {"rows": [{"value": v} for v in data_vals], "key": "value"},
            "ground_truth": mean_val,
        })

    # Category 4: Compositional Multi-Skill Generalization (40 tasks)
    for i in range(40):
        all_tasks.append({
            "task_id": f"comp_{i:02d}",
            "category": "compositional",
            "prompt": f"Execute test suite verification #{i}: check if list reverse algorithm holds.",
            "expected_skill": "unit_test",
            "params": {
                "function_code": "def rev(lst): return list(reversed(lst))",
                "test_code": "assert rev([1, 2]) == [2, 1]",
            },
            "ground_truth": True,
        })

    # Category 5: Edge Cases, Noisy & Adversarial Inputs (40 tasks)
    for i in range(40):
        if i % 2 == 0:
            all_tasks.append({
                "task_id": f"edge_{i:02d}",
                "category": "edge_cases",
                "prompt": f"Undefined query #{i}: Calculate the square root of undefined imaginary variable QX_{i}.",
                "expected_skill": "code_execution",
                "params": {"code": "result = 'unresolvable_input'"},
                "ground_truth": "unresolvable_input",
            })
        else:
            all_tasks.append({
                "task_id": f"edge_{i:02d}",
                "category": "edge_cases",
                "prompt": f"Conflicting noisy instruction #{i}: Return the integer sum of empty list [].",
                "expected_skill": "code_execution",
                "params": {"code": "result = sum([])"},
                "ground_truth": 0,
            })

    # Split 200 tasks deterministically:
    # 100 Learning (indices 0..19 of each category: 5 x 20 = 100)
    # 60 Unseen Generalization (indices 20..31 of each category: 5 x 12 = 60)
    # 40 Retention / Retest (indices 32..39 of each category: 5 x 8 = 40)
    learning_tasks = []
    unseen_tasks = []
    retention_tasks = []

    for cat_name in ("new_knowledge", "python_debug", "workflow_selection", "compositional", "edge_cases"):
        cat_tasks = [t for t in all_tasks if t["category"] == cat_name]
        learning_tasks.extend(cat_tasks[:20])    # 20 * 5 = 100
        unseen_tasks.extend(cat_tasks[20:32])   # 12 * 5 = 60
        retention_tasks.extend(cat_tasks[32:40]) # 8 * 5 = 40

    return {
        "learning": learning_tasks,
        "unseen": unseen_tasks,
        "retention": retention_tasks,
        "all": all_tasks,
    }


def validate_task_execution(task: Dict[str, Any], agent_output: Dict[str, Any]) -> Tuple[bool, float, str]:
    """Strict ground-truth validator that evaluates execution outputs, not arbitrary text."""
    selected_skill = agent_output.get("selected_skill", "")
    exec_res = agent_output.get("exec_result", {})
    expected_skill = task.get("expected_skill", "")
    ground_truth = task.get("ground_truth", None)

    # 1. Skill selection check
    if selected_skill != expected_skill:
        return False, -1.0, f"Incorrect skill selected: '{selected_skill}' != expected '{expected_skill}'"

    # 2. Execution success check
    if not exec_res.get("success", False):
        return False, -1.0, f"Tool execution failed: {exec_res.get('error', 'Unknown error')}"

    # 3. Domain-specific value verification
    actual_res = exec_res.get("result", exec_res.get("stdout", exec_res.get("output")))

    cat = task.get("category")
    if cat == "new_knowledge":
        # Check if ground truth endpoint or knowledge is present
        raw_text = agent_output.get("raw_output", "") + str(actual_res) + " " + " ".join(agent_output.get("retrieved_memories", []))
        if str(ground_truth) in raw_text:
            return True, 1.0, "Verified knowledge recall"
        return False, -0.5, f"Missing expected value: '{ground_truth}'"

    elif cat == "python_debug":
        if actual_res == ground_truth or str(ground_truth) in str(actual_res):
            return True, 1.0, "Bug patch verified"
        return False, -1.0, f"Bug fix output '{actual_res}' != ground truth '{ground_truth}'"

    elif cat == "workflow_selection":
        actual_mean = exec_res.get("mean")
        if actual_mean is not None and abs(actual_mean - float(ground_truth)) < 1e-3:
            return True, 1.0, "Data analysis verified"
        return False, -1.0, f"Calculated mean '{actual_mean}' != expected '{ground_truth}'"

    elif cat == "compositional":
        if actual_res is True or exec_res.get("result") is True:
            return True, 1.0, "Unit test suite passed"
        return False, -1.0, f"Unit test failed: {actual_res}"

    elif cat == "edge_cases":
        if str(ground_truth) in str(actual_res) or actual_res == ground_truth:
            return True, 1.0, "Edge case correctly resolved"
        return False, -1.0, f"Edge case failed: {actual_res}"

    return True, 1.0, "Success"
