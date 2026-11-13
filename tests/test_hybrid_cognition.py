"""Lifecycle/race/transaction fixtures; not evidence of model intelligence."""

import copy
import json
import threading
import time
from types import SimpleNamespace
import numpy as np
import pytest
from scarx.hybrid.engine import HybridEngine
from scarx.hybrid.config import HybridConfig
from scarx.hybrid.learner import LocalALIF, features, ACTIONS
from scarx.hybrid.store import Store


class Monitor:
    def sample(self):
        return {"rss_bytes": 30_000_000, "conservative_ram_bytes": 30_000_000}

    def check(self):
        return self.sample()


class Backend:
    def __init__(self):
        self.cancel_idle = threading.Event()
        self.calls = []
        self.sleep_calls = 0
        self.vlm = None

    def generate(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return {
            "text": json.dumps(
                {
                    "question": "Có nguồn nào cho kết luận này?",
                    "reason": "Cần nguồn kiểm tra.",
                    "public_topic_id": None,
                }
            )
        }

    def hibernate(self, reason, permitted=None):
        self.sleep_calls += 1
        return True

    def wake(self):
        return None


@pytest.fixture
def engine(tmp_path):
    store = Store(tmp_path)
    config = HybridConfig(consolidation_interval_seconds=1, sleep_replay_budget=2)
    e = HybridEngine(store, Backend(), Monitor(), config)
    yield e
    store.close()


def teach(e):
    return e._guarded_learn(
        features("a previously verified action"),
        ACTIONS.index("verify"),
        "action",
        1.0,
        {"kind": "machine_verifier", "id": "fixture:12+3=15"},
    )


def test_heartbeat_does_not_reset_real_activity_quiet_timer(engine):
    engine.state["cognition"]["last_activity"] = time.time() - 45
    engine.record_agent_answer(
        "a", "What is the source of that claim?", "I guess this is true."
    )
    before = engine.state["cognition"]["last_activity"]
    engine.presence(activity=False)
    assert engine.present() and engine.state["cognition"]["last_activity"] == before
    assert engine.review_blocker() is None
    engine.presence(activity=True)
    assert engine.review_blocker() == "waiting_for_quiet"


def test_parallel_foreground_leases_prevent_early_background_update(engine):
    engine.begin_foreground()
    engine.begin_foreground()
    before = {k: v.copy() for k, v in engine.model.arrays().items()}
    engine.end_foreground()
    assert engine.foreground.is_set() and engine.cognition.requests == 1
    with pytest.raises(InterruptedError):
        teach(engine)
    for k, v in before.items():
        np.testing.assert_array_equal(v, engine.model.arrays()[k])
    engine.end_foreground()
    assert not engine.foreground.is_set() and not engine.backend.cancel_idle.is_set()
    assert engine.cognition.requests == 0


def test_replay_budget_and_rng_survive_cold_restart_without_self_growth(engine):
    assert teach(engine)["accepted"]
    count = len(engine.model.replay)
    seen = engine.model.replay_seen
    for i in range(2):
        assert engine.cognition.consolidate_once(time.time() + i * 2)["accepted"]
    assert engine.cognition.consolidate_once(time.time() + 8) is None
    assert len(engine.model.replay) == count and engine.model.replay_seen == seen
    cold = HybridEngine(engine.store, Backend(), Monitor(), engine.config)
    assert cold.state["cognition"]["replay_used"] == 2
    assert cold.cognition.consolidate_once(time.time() + 20) is None
    assert cold.model.rng.bit_generator.state == engine.model.rng.bit_generator.state


def test_replay_db_failure_restores_sampling_rng_and_controller(engine, monkeypatch):
    teach(engine)
    before = engine.model.dumps(engine.state)

    def fail(*args, **kwargs):
        raise OSError("controlled checkpoint write failure")

    monkeypatch.setattr(engine.store, "save_snapshot", fail)
    result = engine.cognition.consolidate_once()
    assert result["persistence_failed"] and not result["accepted"]
    assert engine.model.dumps(engine.state) == before


def test_pre_cognition_migration_keeps_every_neural_array_rng_and_review(engine):
    legacy = copy.deepcopy(engine.state)
    legacy.pop("cognition")
    identifier = engine.store.save_snapshot(engine.model.dumps(legacy))
    before = engine.model.dumps()
    engine.load(identifier)
    assert engine.state["cognition"]["format"] == "scarx-cognition-v1"
    assert engine.model.dumps() == before


def test_social_and_compaction_helpers_are_stored_but_never_teacher_labels(engine):
    engine.record_agent_answer("social", "chào bạn", "Chào bạn!")
    engine.record_agent_answer(
        "summary",
        "Create a context checkpoint. TURNS TO SUMMARIZE: x ## Historical Task Snapshot",
        "an auxiliary summary",
    )
    assert engine.store.get("answer", "social") and engine.store.get(
        "answer", "summary"
    )
    assert not engine.state["review"]["pending"] and engine.model.updates == 0
    engine.save()
    cold = HybridEngine(engine.store, Backend(), Monitor(), engine.config)
    assert not cold.state["review"]["pending"]
    engine.record_agent_answer("real", "chào bạn, 17+25 bằng bao nhiêu?", "41")
    assert engine.state["review"]["pending"] == ["real"]


def test_unrelated_model_topic_removed_and_defer_becomes_clarification(
    engine, monkeypatch
):
    topic = next(
        t for t in engine.store.list("public_topic") if t["topic"] == "toán học"
    )
    g = engine.goal(
        "What is the capital of Vietnam?", public_topic_id=topic["id"], origin="vlm"
    )
    assert g["public_topic_id"] is None
    assert engine.step() == "review"
    monkeypatch.setattr(
        engine.model,
        "decide",
        lambda seq, allowed: ("defer", dict.fromkeys(ACTIONS, 1 / 6)),
    )
    assert engine.step() == "ask"
    q = engine.state["question"]
    assert q["origin"] == "runtime_progress_guard"
    assert q["status"] == "queued_absent"
    engine.presence(activity=False)
    assert engine.state["question"]["status"] == "pending"
    before = engine.model.updates
    assert engine.feedback(q["id"], text="bỏ qua")["skipped"]
    assert engine.model.updates == before and not engine.state["question"]


def test_local_verifier_gets_priority_without_fake_model_answer(engine, monkeypatch):
    g = engine.goal("12+3", origin="engineering_acceptance")
    assert engine.step() == "review"
    monkeypatch.setattr(
        engine.model,
        "decide",
        lambda seq, allowed: ("defer", dict.fromkeys(ACTIONS, 1 / 6)),
    )
    assert engine.step() == "verify"
    assert g["status"] == "resolved_verified"
    assert engine.model.updates == 1 and engine.model.hidden_updates == 1
    assert engine.model.replay[0]["evidence"]["kind"] == "machine_verifier"
    assert not engine.backend.calls


def test_capture_and_foreground_prevent_sleep_and_pause_does_not_block_chat(engine):
    engine.cognition.command("sleep")
    engine.capture_active = lambda: True
    assert engine.cognition.choose_work(time.time()) == "observing"
    engine.capture_active = lambda: False
    engine.state["learning_enabled"] = False
    engine.begin_foreground()
    assert engine.cognition.choose_work(time.time()) == "serving"
    engine.end_foreground()
    assert engine.cognition.choose_work(time.time()) == "paused"


def test_network_lookup_does_not_hold_learner_lock(engine, monkeypatch):
    topic = next(
        t for t in engine.store.list("public_topic") if t["topic"] == "toán học"
    )
    g = engine.goal("toán học", public_topic_id=topic["id"], origin="user")
    g["reviewed"] = True
    monkeypatch.setattr(
        engine.model,
        "decide",
        lambda seq, allowed: ("search", dict.fromkeys(ACTIONS, 1 / 6)),
    )
    entered = threading.Event()
    release = threading.Event()

    def blocked(*args):
        entered.set()
        release.wait(2)
        return []

    monkeypatch.setattr(engine.web, "search", blocked)
    thread = threading.Thread(target=engine.step)
    thread.start()
    try:
        assert entered.wait(1)
        start = time.monotonic()
        engine.begin_foreground()
        assert time.monotonic() - start < 0.25
        engine.end_foreground()
    finally:
        release.set()
        thread.join(3)
    assert not thread.is_alive()


def test_explicit_awake_can_load_immediately_even_with_learning_paused(engine):
    engine.state["learning_enabled"] = False
    engine.cognition.command("awake")
    assert engine.cognition.choose_work(time.time()) == "waking"


def test_failed_foreground_event_does_not_leak_lease(engine, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("audit write failure")

    monkeypatch.setattr(engine.store, "event", fail)
    with pytest.raises(OSError):
        engine.begin_foreground()
    assert engine.cognition.requests == 0
    assert not engine.foreground.is_set()
    assert not engine.cognition.request_pending.is_set()
    assert not engine.backend.cancel_idle.is_set()


def test_hibernation_rechecks_permission_under_backend_lock(engine):
    from scarx.hybrid.backends import NativeBackends

    backend = NativeBackends(engine.store.directory, engine.store, engine.monitor)
    with pytest.raises(InterruptedError):
        backend.hibernate("stale scheduled rest", permitted=lambda: False)


def test_changed_assets_are_reverified_before_resume(engine, tmp_path, monkeypatch):
    from scarx.hybrid.backends import NativeBackends
    from scarx.hybrid import qualification

    backend = NativeBackends(tmp_path, engine.store, engine.monitor)
    backend.vlm_command = ["llama-server.exe"]
    backend.asset_signatures = ("old",)
    monkeypatch.setattr(backend, "_asset_signatures", lambda: ("changed",))

    def reject(_):
        raise ValueError("native integrity mismatch")

    monkeypatch.setattr(qualification, "verify_native_storage", reject)
    with pytest.raises(ValueError, match="integrity mismatch"):
        backend.ensure_vlm()
    assert backend.vlm is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("cycles", -1),
        ("replay_used", True),
        ("next_replay", float("nan")),
        ("intent", "dream"),
    ],
)
def test_invalid_cognitive_state_rejected(field, value):
    from scarx.hybrid.cognition import initial_cognition, validate_cognition

    state = initial_cognition()
    state[field] = value
    with pytest.raises(ValueError):
        validate_cognition(state)
