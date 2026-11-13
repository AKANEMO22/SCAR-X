"""Engineering checks, not evidence of lifelong/representation performance."""

import io
import json
import zipfile
import numpy as np
import pytest
from scarx.hybrid.config import HybridConfig
from scarx.hybrid.learner import LocalALIF, features
from scarx.hybrid.store import Store
from scarx.hybrid.engine import HybridEngine
from scarx.hybrid.public_web import public_url


class Monitor:
    def sample(self):
        return {"rss_bytes": 30_000_000, "limit_bytes": 4_000_000_000}

    def check(self):
        return self.sample()


class NoModel:
    def generate(self, *a, **kw):
        raise AssertionError("No external model calls in engineering fixture")


@pytest.fixture
def engine(tmp_path):
    store = Store(tmp_path)
    value = HybridEngine(store, NoModel(), Monitor())
    yield value
    store.close()


def test_budget_amendment():
    assert HybridConfig().ram_limit == 4_000_000_000
    with pytest.raises(ValueError):
        HybridConfig(ram_limit=4_000_000_001)


def test_checkpoint_exact_cold_next_update():
    model = LocalALIF()
    seq = features("visible red car then blue bicycle", (1.0, 0.0))
    evidence = {"kind": "machine_verifier", "id": "fixture-output-not-real-user"}
    model.learn(seq, 2, evidence=evidence)
    payload = model.dumps({"goals": {"x": "y"}, "pending_question": "fixture"})
    other, state = LocalALIF.loads(payload)
    assert state["pending_question"] == "fixture"
    for name, array in model.arrays().items():
        np.testing.assert_array_equal(array, other.arrays()[name])
    assert model.rng.bit_generator.state == other.rng.bit_generator.state
    first = model.learn(seq, 3, evidence=evidence)
    second = other.learn(seq, 3, evidence=evidence)
    assert first == second
    assert model.dumps(state) == other.dumps(state)


def test_local_hidden_ablation():
    learned = LocalALIF(seed=11)
    readout = LocalALIF(seed=11, mode="readout_only")
    frozen = LocalALIF(seed=11, mode="frozen")
    seq = features("pick a source to inspect", (1.0,))
    w0 = learned.core.w_in.copy()
    for model in (learned, readout, frozen):
        model.learn(seq, 1, evidence={"kind": "human_feedback", "id": "fixture"})
    assert not np.array_equal(learned.core.w_in, w0)
    np.testing.assert_array_equal(readout.core.w_in, w0)
    np.testing.assert_array_equal(frozen.core.w_in, w0)
    assert (
        learned.hidden_updates == 1
        and readout.hidden_updates == frozen.hidden_updates == 0
    )
    assert frozen.updates == 0


@pytest.mark.parametrize(
    "source",
    [
        None,
        {},
        {"kind": "model_confidence", "id": "x"},
        {"kind": "verified", "id": "x"},
    ],
)
def test_generated_claims_cannot_authorize_learning(source):
    model = LocalALIF()
    before = model.dumps()
    with pytest.raises(ValueError):
        model.learn(features("unverified output"), 1, evidence=source)
    assert model.dumps() == before


def test_snapshot_rejects_duplicate_member():
    payload = LocalALIF().dumps()
    stream = io.BytesIO(payload)
    with zipfile.ZipFile(stream, "a") as archive:
        with pytest.warns(UserWarning):
            archive.writestr("core.w_in.npy", b"bad")
    with pytest.raises(ValueError):
        LocalALIF.loads(stream.getvalue())


def test_snapshot_rejects_float64_and_bad_topology():
    payload = LocalALIF().dumps()
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        data = {k: archive[k] for k in archive.files}
    data["core.w_in"] = data["core.w_in"].astype(np.float64)
    stream = io.BytesIO()
    np.savez(stream, **data)
    with pytest.raises(ValueError):
        LocalALIF.loads(stream.getvalue())
    data["core.w_in"] = data["core.w_in"].astype(np.float32)
    data["core.indices"][0, 0] = 999
    stream = io.BytesIO()
    np.savez(stream, **data)
    with pytest.raises(ValueError):
        LocalALIF.loads(stream.getvalue())


def test_replay_is_bounded_and_state_fully_counted():
    model = LocalALIF(mode="readout_only")
    for i in range(75):
        model.learn(
            features("experience " + str(i)),
            i % 6,
            evidence={"kind": "machine_verifier", "id": str(i)},
        )
    assert model.replay_seen == 75 and len(model.replay) == 64
    assert model.memory_bytes() >= sum(a.nbytes for a in model.arrays().values())
    other, _ = LocalALIF.loads(model.dumps())
    assert other.replay == model.replay


def test_store_fts_temporal_split_and_verification(engine):
    first = engine.observe(
        "red car before bridge", source="fixture", stream="a", start=0, end=3
    )
    engine.observe(
        "blue car after bridge", source="fixture", stream="a", start=8, end=10
    )
    engine.observe(
        "red car in another video", source="fixture", stream="b", start=1, end=2
    )
    rows = engine.store.search("car", stream="a", before=4)
    assert [row["id"] for row in rows] == [first["id"]]
    with pytest.raises(ValueError):
        engine.store.observe("wrong", source="checkbox", status="verified")


def test_private_goal_cannot_search_web(engine):
    goal = engine.goal("Tên người trong video riêng tư là gì?", origin="user")
    assert "search" not in engine.allowed(goal)
    with pytest.raises(PermissionError):
        engine.web.search("model-invented-public-id")


def test_absent_question_is_queued_and_skip_is_not_a_label(engine):
    goal = engine.goal("Điều gì chưa rõ trong fixture này?", origin="user")
    goal["reviewed"] = True
    engine.model.decide = lambda sequence, allowed: ("ask", {})
    assert engine.step() == "ask"
    question = engine.state["question"]
    assert question["status"] == "queued_absent"
    assert engine.status()["question"] is None
    engine.presence()
    assert engine.status()["question"]["status"] == "pending"
    before = engine.model.updates
    result = engine.feedback(question["id"], text="bỏ qua")
    assert result["skipped"] and engine.model.updates == before
    assert engine.state["goals"][goal["id"]]["status"] == "deferred"


def test_save_rollback_preserves_audit_and_pending(engine):
    first = engine.goal("Original gap?", origin="user")
    first["reviewed"] = True
    engine.model.decide = lambda s, a: ("ask", {})
    engine.step()
    snapshot = engine.save()
    question_id = engine.state["question"]["id"]
    old_events = engine.store.events()
    engine.feedback(question_id, text="A human statement", helpful=True)
    engine.goal("Another gap?", origin="user")
    engine.load(snapshot)
    assert engine.state["question"]["id"] == question_id
    assert len(engine.state["goals"]) == 1 and first["id"] in engine.state["goals"]
    assert len(engine.store.events()) > len(old_events)
    assert engine.store.search("human statement")  # Immutable observation retained.


def test_generic_feedback_does_not_label_all_memories(engine):
    item = engine.observe("known event", source="fixture")
    engine.store.put(
        "answer",
        "a",
        {"question": "which event?", "candidate_ids": [item["id"]], "stream": None},
    )
    before = engine.model.updates
    result = engine.feedback("a", helpful=False)
    assert result["learning"] == [] and engine.model.updates == before


def test_bad_feedback_batch_cannot_partially_update(engine):
    item = engine.observe("known event", source="fixture")
    engine.store.put(
        "answer",
        "a",
        {"question": "which event?", "candidate_ids": [item["id"]], "stream": None},
    )
    before = engine.model.dumps(engine.state)
    with pytest.raises(ValueError):
        engine.feedback("a", positive_ids=[item["id"], "foreign"])
    assert engine.model.dumps(engine.state) == before


def test_database_failure_restores_entire_local_state(engine, monkeypatch):
    before = engine.model.dumps(engine.state)

    def fail(*args, **kwargs):
        raise OSError("fixture disk failure")

    monkeypatch.setattr(engine.store, "save_snapshot", fail)
    result = engine._guarded_learn(
        features("independent source"),
        1,
        "relevance",
        1.0,
        {"kind": "machine_verifier", "id": "fixture"},
    )
    assert result["persistence_failed"] and result["rolled_back"]
    assert engine.model.dumps(engine.state) == before


def test_automatic_checkpoints_bounded_manual_pin_retained(engine):
    pinned = engine.save(pinned=True)
    for _ in range(36):
        engine.save()
    assert engine.store.snapshot(pinned)
    count = engine.store.db.execute("SELECT count(*) FROM checkpoints").fetchone()[0]
    assert count == 33


def test_public_url_blocks_private_addresses(monkeypatch):
    import socket

    monkeypatch.setattr(
        socket, "getaddrinfo", lambda *a, **k: [(2, 1, 6, "", ("127.0.0.1", 443))]
    )
    with pytest.raises(ValueError):
        public_url("https://example.com/data")
    for url in (
        "file:///private",
        "http://example.com",
        "https://name:password@example.com",
    ):
        with pytest.raises(ValueError):
            public_url(url)
