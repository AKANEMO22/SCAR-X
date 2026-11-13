"""Real controller/learner fixtures; these tests do not prove general self-improvement."""

import json
import time
from pathlib import Path
import numpy as np
import pytest
from scarx.hybrid.engine import HybridEngine
from scarx.hybrid.store import Store
from scarx.hybrid.learner import features, LocalALIF
from scarx.hybrid.review import parse_review, review_format, upgrade_controller


class Monitor:
    def sample(self):
        return {"rss_bytes": 30_000_000, "conservative_ram_bytes": 30_000_000}

    def check(self):
        return self.sample()


class Reviewer:
    def __init__(self, proposal=None, error=None):
        self.proposal = proposal or {
            "question": "Có bằng chứng độc lập cho kết luận này không?",
            "reason": "Cần kiểm lại nguồn.",
            "public_topic_id": None,
        }
        self.error = error
        self.calls = []

    def generate(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        if self.error:
            raise self.error
        return {"text": json.dumps(self.proposal, ensure_ascii=False)}


@pytest.fixture
def engine(tmp_path):
    store = Store(tmp_path)
    value = HybridEngine(store, Reviewer(), Monitor())
    yield value
    store.close()


def record(
    engine, question="Ứng dụng này làm gì?", answer="Tôi đoán ứng dụng quản lý video."
):
    engine.record_agent_answer(
        "a", question, answer, source="actual-controller-fixture"
    )
    engine.state["last_presence"] = time.time() - 45


def test_completed_turn_is_queued_and_title_helper_is_excluded(engine):
    record(engine)
    engine.record_agent_answer("title", "title task", '{"title":"A short title"}')
    assert engine.state["review"]["pending"] == ["a"]
    assert engine.model.updates == 0


def test_review_can_run_after_quiet_while_user_still_present(engine):
    record(engine)
    assert engine.present() and engine.review_blocker() is None
    result = engine.review_once()
    assert result["status"] == "needs_verification" and not result["weights_updated"]
    assert (
        engine.state["review"]["completed"] == 1
        and not engine.state["review"]["pending"]
    )
    assert engine.model.updates == 0 and not engine.model.replay
    messages, options = engine.backend.calls[0]
    assert "actual" not in options  # Actual source record, no invented training target.
    assert "Tôi đoán ứng dụng" in messages[1]["content"]
    assert options["response_format"]["type"] == "json_schema" and options["background"]
    assert engine.review_blocker() == "review_cooldown"


@pytest.mark.parametrize("reason", ["foreground", "paused", "quiet", "memory"])
def test_review_respects_priority_pause_quiet_and_resource_budget(engine, reason):
    record(engine)
    if reason == "foreground":
        engine.foreground.set()
    elif reason == "paused":
        engine.state["learning_enabled"] = False
    elif reason == "quiet":
        engine.state["last_presence"] = time.time()
    else:
        engine.monitor.sample = lambda: {
            "rss_bytes": 30_000_000,
            "conservative_ram_bytes": engine.config.soft_ram_limit + 1,
        }
    assert engine.review_once() is None
    assert not engine.backend.calls and engine.state["review"]["pending"] == ["a"]
    assert engine.model.updates == 0


def test_preempted_review_retains_work_for_later_without_updates(engine):
    record(engine)
    engine.backend = Reviewer(error=InterruptedError("foreground"))
    assert engine.review_once() is None
    assert (
        engine.state["review"]["pending"] == ["a"]
        and engine.state["review"]["completed"] == 0
    )
    assert engine.model.updates == 0


def test_machine_checked_numeric_error_can_train_only_local_policy(engine):
    record(engine, "2+2", "5")
    result = engine.review_once()
    assert (
        result["status"] == "verified_error" and result["verification"]["expected"] == 4
    )
    assert result["weights_updated"] and engine.model.updates == 1
    assert result["learning"]["evidence"]["kind"] == "machine_verifier"
    assert "frozen VLM unchanged" in result["learning_scope"]
    assert not engine.backend.calls  # Arithmetic was checked independently of the VLM.


def test_correct_numeric_answer_is_not_self_training(engine):
    record(engine, "2+2", "4")
    assert engine.review_once()["status"] == "verified_correct"
    assert engine.model.updates == 0 and not engine.model.replay


def test_replay_runs_without_an_open_goal_and_does_not_grow_training_set(engine):
    engine.model.learn(
        features("past verified experience"),
        1,
        head="relevance",
        evidence={"kind": "human_feedback", "id": "fixture-human-feedback"},
    )
    old_count = len(engine.model.replay)
    old_seen = engine.model.replay_seen
    old_updates = engine.model.updates
    old_probes = json.loads(json.dumps(engine.state["probes"]))
    result = engine.review_once()
    assert result["accepted"] and engine.model.updates == old_updates + 1
    assert (
        len(engine.model.replay) == old_count and engine.model.replay_seen == old_seen
    )
    assert engine.state["review"]["replay_updates"] == 1 and not engine.state["goals"]
    assert engine.state["probes"] == old_probes


def test_review_checkpoint_restart_and_rollback_are_complete(engine):
    record(engine)
    before = engine.save(pinned=True)
    engine.review_once()
    completed = engine.save()
    arrays = {k: v.copy() for k, v in engine.model.arrays().items()}
    cold = HybridEngine(engine.store, Reviewer(), Monitor())
    assert cold.state["review"] == engine.state["review"]
    for name, value in arrays.items():
        np.testing.assert_array_equal(value, cold.model.arrays()[name])
    cold.load(before)
    assert (
        cold.state["review"]["pending"] == ["a"]
        and not cold.state["review"]["completed"]
    )
    cold.load(completed)
    assert cold.state["review"]["completed"] == 1


def test_pre_review_checkpoint_additive_migration_preserves_neural_rng(engine):
    legacy = dict(engine.state)
    legacy.pop("review")
    payload = engine.model.dumps(legacy)
    identifier = engine.store.save_snapshot(payload)
    before = engine.model.dumps()
    engine.load(identifier)
    assert (
        engine.state["review"]["format"] == "scarx-review-v1"
        and engine.model.dumps() == before
    )
    bad = dict(engine.state, review={"format": "invented"})
    with pytest.raises(ValueError):
        upgrade_controller(bad)


def test_invalid_review_is_visible_and_never_a_label(engine):
    record(engine)
    engine.backend = Reviewer(proposal={"question": "Missing field"})
    result = engine.review_once()
    assert result["status"] == "review_failed" and not result["weights_updated"]
    assert engine.state["review"]["rejected"] == 1 and engine.model.updates == 0


def test_review_schema_does_not_authorize_an_unapproved_public_topic():
    schema = review_format([])["json_schema"]["schema"]
    assert schema["properties"]["public_topic_id"] == {"type": "null"}
    with pytest.raises(ValueError):
        parse_review(
            '{"question":"q?","reason":"r","public_topic_id":"private-data"}', []
        )


def test_structured_review_schema_reaches_native_request(engine, monkeypatch):
    import scarx.hybrid.backends as module
    from scarx.hybrid.backends import NativeBackends

    backend = NativeBackends(engine.store.directory, engine.store, engine.monitor)
    backend.ensure_vlm = lambda: None
    captured = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self, *args):
            return json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "content": '{"question":"","reason":"","public_topic_id":null}'
                            }
                        }
                    ]
                }
            ).encode()

    def request(req, **kwargs):
        captured.append(json.loads(req.data))
        return Response()

    monkeypatch.setattr(module, "urlopen", request)
    schema = review_format([])
    backend.generate(
        [{"role": "user", "content": "actual bounded engineering input"}],
        response_format=schema,
    )
    assert captured[0]["response_format"] == schema
    assert captured[0]["chat_template_kwargs"]["enable_thinking"] is False
