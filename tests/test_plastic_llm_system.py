"""Unit tests for Plastic LLM System components and scientific constraints."""
import os
import shutil
import tempfile
from pathlib import Path
import numpy as np
import pytest

from plastic_llm.plastic_associator import PlasticAssociativeMemory, SimpleFeedbackRanker, ContextFeatureExtractor
from plastic_llm.memory_store import MemoryStore
from plastic_llm.skill_registry import SkillRegistry
from plastic_llm.benchmark_dataset import generate_benchmark_dataset, validate_task_execution


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
    # Different contexts must produce different sparse patterns
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

    assert len(data["learning"]) == 100
    assert len(data["unseen"]) == 60
    assert len(data["retention"]) == 40
    assert len(data["all"]) == 200

    # Verify zero overlap between learning and unseen task IDs
    train_ids = set(t["task_id"] for t in data["learning"])
    unseen_ids = set(t["task_id"] for t in data["unseen"])
    retention_ids = set(t["task_id"] for t in data["retention"])

    assert len(train_ids.intersection(unseen_ids)) == 0
    assert len(train_ids.intersection(retention_ids)) == 0
    assert len(unseen_ids.intersection(retention_ids)) == 0


def test_simple_feedback_ranker_ema():
    skills = ["code_execution", "file_search"]
    ranker = SimpleFeedbackRanker(skills, alpha=0.5)

    ranker.record_decision("run code", "code_execution")
    ranker.update_from_feedback("code_execution", verified_reward=1.0)

    affs = ranker.get_skill_affinities("run code")
    assert affs["code_execution"] > affs["file_search"]
