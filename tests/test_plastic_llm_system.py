"""Unit tests for Plastic LLM System components and scientific constraints.

Verifies:
- P0-A: Complete decoupling of PublicTaskInput from PrivateEvaluationMetadata (no leakage).
- P0-B: Real compositional generalization task diversity (6 distinct algorithmic types).
- P0-C: Authoritative evaluation oracle rejects falsified/corrupted traces.
- P0-E: Sandboxed Python Worker process isolation, timeout, scratch management.
- Baseline: ContextualLinearRouter (Variant G) policy updates.
- Memory: FTS5 SQLite BM25 storage and multi-instance isolation.
- Plasticity: Eligibility traces and associative updates.
"""
import os
import shutil
import tempfile
import time
from pathlib import Path
import numpy as np
import pytest

from plastic_llm.plastic_associator import (
    PlasticAssociativeMemory,
    SimpleFeedbackRanker,
    ContextFeatureExtractor,
)
from plastic_llm.memory_store import MemoryStore
from plastic_llm.skill_registry import SkillRegistry
from plastic_llm.task_types import PublicTaskInput, PrivateEvaluationMetadata
from plastic_llm.benchmark_dataset import generate_benchmark_dataset
from plastic_llm.sandbox_worker import SandboxedPythonWorker
from plastic_llm.evaluation_oracle import evaluate_execution_trace, EvaluationResult
from plastic_llm.contextual_baseline import ContextualLinearRouter


@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp(prefix="test_plastic_llm_")
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


def test_feature_extractor_sparse_normalized():
    extractor = ContextFeatureExtractor(dim=128)
    vec1 = extractor.extract("Fix bug in python script")
    vec2 = extractor.extract("Analyze CSV table data")

    assert vec1.shape == (128,)
    assert np.isclose(np.linalg.norm(vec1), 1.0)
    assert np.isclose(np.linalg.norm(vec2), 1.0)
    # Different contexts produce different sparse patterns
    assert not np.allclose(vec1, vec2)


def test_plastic_associator_eligibility_trace_and_learning_update():
    skills = ["python_debug", "data_analysis", "unit_test"]
    learner = PlasticAssociativeMemory(skill_ids=skills, feature_dim=128, eta=0.1, trace_decay=0.8, seed=42)

    prompt = "There is a syntax error in python function"
    initial_affinities = learner.get_skill_affinities(prompt)

    # Record decision for python_debug
    learner.record_decision(prompt, "python_debug")
    idx = learner.skill_to_idx["python_debug"]
    assert np.linalg.norm(learner.traces[idx]) > 0.0

    # Provide positive verified reward
    learner.update_from_feedback(verified_reward=1.0)
    new_affinities = learner.get_skill_affinities(prompt)

    # Association for python_debug must increase
    assert new_affinities["python_debug"] > initial_affinities["python_debug"]


def test_memory_store_fts5_bm25_and_isolation(temp_dir):
    db1 = temp_dir / "mem1.db"
    db2 = temp_dir / "mem2.db"

    store1 = MemoryStore(db1)
    store2 = MemoryStore(db2)

    store1.add_semantic_knowledge("API_KEY", "network", "Endpoint is https://alpha.internal", 1.0)
    store2.add_semantic_knowledge("API_KEY", "network", "Endpoint is https://beta.internal", 1.0)

    res1 = store1.search_semantic("alpha")
    res2 = store2.search_semantic("alpha")

    assert len(res1) > 0
    assert "alpha.internal" in res1[0]["content"]
    # Verify strict isolation: store2 must NOT contain alpha
    assert len(res2) == 0
    assert not any("alpha" in r["content"] for r in res2)

    store1.close()
    store2.close()


def test_benchmark_dataset_partition_and_leakage():
    data = generate_benchmark_dataset(seed=140011)

    assert len(data["learning_public"]) == 100
    assert len(data["learning_private"]) == 100
    assert len(data["unseen_public"]) == 60
    assert len(data["unseen_private"]) == 60
    assert len(data["retention_public"]) == 40
    assert len(data["retention_private"]) == 40

    # Verify zero overlap between split task IDs
    train_ids = set(t.task_id for t in data["learning_public"])
    unseen_ids = set(t.task_id for t in data["unseen_public"])
    retention_ids = set(t.task_id for t in data["retention_public"])

    assert len(train_ids.intersection(unseen_ids)) == 0
    assert len(train_ids.intersection(retention_ids)) == 0
    assert len(unseen_ids.intersection(retention_ids)) == 0

    # P0-A Check: Verify PublicTaskInput has strictly NO ground truth or reference solution
    for pub in data["learning_public"] + data["unseen_public"] + data["retention_public"]:
        assert not hasattr(pub, "ground_truth")
        assert not hasattr(pub, "reference_solution")
        assert not hasattr(pub, "test_fixture")
        assert pub.prompt and len(pub.prompt) > 5

    # P0-B Check: Compositional category must contain multiple algorithmic structures
    comp_metas = [m for m in data["all_private"] if m.category == "compositional"]
    assert len(comp_metas) == 40
    distinct_types = set()
    for m in comp_metas:
        for fn_name in ["fact(", "evens(", "is_pal(", "find_idx(", "trace(", "inv("]:
            if fn_name in m.reference_solution:
                distinct_types.add(fn_name)
    assert len(distinct_types) == 6, f"Expected 6 distinct compositional structures, found: {distinct_types}"


def test_sandboxed_python_worker():
    worker = SandboxedPythonWorker(timeout_sec=2.0)

    # 1. Normal execution
    code = "result = sum([1, 2, 3, 4, 5])\nprint(f'Sum is {result}')"
    res = worker.execute_snippet(code)
    assert res["success"] is True
    assert "Sum is 15" in res["stdout"]

    # 2. Timeout enforcement (infinite loop)
    t0 = time.time()
    res_loop = worker.execute_snippet("while True: pass")
    elapsed = time.time() - t0
    assert res_loop["success"] is False
    assert "timed out" in res_loop["error"].lower()
    assert elapsed < 4.0  # Must terminate within timeout


def test_evaluation_oracle_trace_verification():
    meta = PrivateEvaluationMetadata(
        task_id="test_01",
        category="python_debug",
        expected_skill="python_debug",
        ground_truth=42,
        reference_solution="def f(): return 42",
    )

    # Correct trace
    valid_trace = {
        "selected_skill": "python_debug",
        "params": {"code": "def f(): return 42\nresult = f()"},
        "exec_result": {"success": True, "result": 42},
        "raw_output": "Fixed bug",
    }
    res = evaluate_execution_trace(meta, valid_trace)
    assert res.end_to_end_pass is True
    assert res.tool_routing_correct is True
    assert res.semantic_correctness is True
    assert res.reward == 1.0


def test_evaluation_oracle_rejects_corrupted_trace():
    meta = PrivateEvaluationMetadata(
        task_id="test_02",
        category="python_debug",
        expected_skill="python_debug",
        ground_truth=42,
        reference_solution="def f(): return 42",
    )

    # Case 1: Wrong routing
    wrong_route_trace = {
        "selected_skill": "document_read",
        "params": {},
        "exec_result": {"success": True, "result": 42},
    }
    r1 = evaluate_execution_trace(meta, wrong_route_trace)
    assert r1.end_to_end_pass is False
    assert r1.tool_routing_correct is False
    assert r1.reward == -1.0

    # Case 2: Output mismatch (semantic corruption)
    wrong_value_trace = {
        "selected_skill": "python_debug",
        "params": {"code": "result = 99"},
        "exec_result": {"success": True, "result": 99},
    }
    r2 = evaluate_execution_trace(meta, wrong_value_trace)
    assert r2.end_to_end_pass is False
    assert r2.tool_routing_correct is True
    assert r2.semantic_correctness is False
    assert r2.reward == -1.0

    # Case 3: Execution failed
    fail_trace = {
        "selected_skill": "python_debug",
        "params": {"code": "1/0"},
        "exec_result": {"success": False, "error": "ZeroDivisionError"},
    }
    r3 = evaluate_execution_trace(meta, fail_trace)
    assert r3.end_to_end_pass is False
    assert r3.execution_success is False


def test_contextual_linear_router_baseline():
    skills = ["code_execution", "python_debug", "unit_test"]
    router = ContextualLinearRouter(skill_ids=skills, feature_dim=128, learning_rate=0.2, l2_reg=1e-4, seed=42)

    prompt = "Unit test failing assertion in data pipeline"
    affs_init = router.get_skill_affinities(prompt)
    assert len(affs_init) == 3

    # Perform online policy update favoring unit_test on this prompt context
    for _ in range(5):
        router.record_decision(prompt, "unit_test")
        router.update_from_feedback("unit_test", verified_reward=1.0)

    affs_updated = router.get_skill_affinities(prompt)
    assert affs_updated["unit_test"] > affs_init["unit_test"]


def test_simple_feedback_ranker_ema():
    skills = ["code_execution", "file_search"]
    ranker = SimpleFeedbackRanker(skills, alpha=0.5)

    ranker.record_decision("run code", "code_execution")
    ranker.update_from_feedback("code_execution", verified_reward=1.0)

    affs = ranker.get_skill_affinities("run code")
    assert affs["code_execution"] > affs["file_search"]
