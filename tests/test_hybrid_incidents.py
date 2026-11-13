"""Observation/credit boundaries; synthetic fixtures are not neural-skill proof."""

import copy
import hashlib
import importlib.util
from pathlib import Path
import numpy as np
import pytest
from scarx.hybrid.engine import HybridEngine
from scarx.hybrid.incidents import IncidentLedger
from scarx.hybrid.store import Store
from scarx.hybrid.learner import ACTIONS
from types import SimpleNamespace

PATH = (
    Path(__file__).resolve().parents[1]
    / "integrations/hermes/hybrid_plugin/learning_hooks.py"
)
SPEC = importlib.util.spec_from_file_location("scarx_experience_hooks", PATH)
hooks = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(hooks)


@pytest.fixture
def engine(tmp_path):
    store = Store(tmp_path)
    monitor = SimpleNamespace(
        sample=lambda: {"rss_bytes": 1_000_000}, check=lambda: {"rss_bytes": 1_000_000}
    )
    e = HybridEngine(store, SimpleNamespace(), monitor)
    yield e
    store.close()


def payload(i=0, tool="read_file", scope="turn"):
    return hooks.tool_observation(
        tool,
        {"path": "private/path"},
        {"error": "not found"},
        hooks.digest(scope),
        "call" + str(i),
        "error",
        "FileNotFoundError",
        12,
    )


def test_three_real_observation_shapes_create_one_goal_without_weight_update(engine):
    before = engine.model.dumps()
    for i in range(3):
        result = engine.observe_tool_experience(payload(i))
    incident = result["incident"]
    assert incident and incident["root_cause"] == "unresolved"
    assert engine.model.dumps() == before and engine.model.updates == 0
    assert (
        len(engine.state["goals"]) == 1
        and incident["observation_is_training_label"] is False
    )
    assert engine.observe_tool_experience(payload(2))["duplicate"]
    for i in range(3, 8):
        engine.observe_tool_experience(payload(i))
    assert (
        len(engine.incidents.summary()["recent"]) == 1
        and len(engine.state["goals"]) == 1
    )
    assert "private/path" not in str(engine.incidents.summary())


def test_pollers_and_new_turns_do_not_share_a_false_loop(engine):
    for i in range(8):
        engine.observe_tool_experience(payload(i, tool="process_manage"))
    for i in range(2):
        engine.observe_tool_experience(payload(i, scope="new-turn"))
    assert not engine.incidents.summary()["recent"]


def test_ledger_cold_persistence_and_audit_survives_neural_rollback(engine):
    old = engine.save(pinned=True)
    for i in range(3):
        result = engine.observe_tool_experience(payload(i))
    cold = IncidentLedger(engine.store)
    assert cold.summary()["recent"][0]["id"] == result["incident"]["id"]
    engine.load(old)
    assert not engine.state["goals"] and cold.summary()["recent"]
    assert engine.model.updates == 0


def test_turn_storage_and_incidents_remain_bounded(engine):
    for s in range(70):
        for i in range(3):
            engine.observe_tool_experience(payload(i, scope=str(s)))
    assert len(engine.store.get("tool_experience", "bounded_turns")) == 64
    assert (
        engine.store.db.execute(
            "select count(*) from objects where kind='tool_incident'"
        ).fetchone()[0]
        == 32
    )


def test_reported_verified_flag_and_unknown_fields_cannot_create_learning(engine):
    p = payload()
    p["verified"] = True
    with pytest.raises(ValueError):
        engine.observe_tool_experience(p)
    assert engine.model.updates == 0


def test_hook_only_tracks_exact_scarx_route_and_hides_raw_inputs():
    sent = []
    observer = hooks.ToolExperienceObserver(sent.append)
    args = {"command": "secret command with token private123"}
    result = {"data": {"web": []}}
    common = {"session_id": "s", "turn_id": "t"}
    observer.pre_api(model="other", base_url="http://127.0.0.1:8775/v1", **common)
    observer.post_tool(
        tool_name="web_search", args=args, result=result, tool_call_id="one", **common
    )
    assert not sent
    observer.pre_api(
        model="scarx-hybrid", base_url="http://127.0.0.1:8775/v1", **common
    )
    observer.post_tool(
        tool_name="web_search", args=args, result=result, tool_call_id="one", **common
    )
    assert len(sent) == 1 and sent[0]["empty_result"] and "private123" not in str(sent)
    observer.pre_api(model="scarx-hybrid", base_url="https://remote.test/v1", **common)
    observer.post_tool(
        tool_name="web_search", args=args, result=result, tool_call_id="two", **common
    )
    assert len(sent) == 1


def test_changing_model_route_releases_per_turn_dedup_state():
    observer = hooks.ToolExperienceObserver(lambda payload: None)
    for turn in range(100):
        scope = dict(session_id="same-session", turn_id=str(turn))
        observer.pre_api(
            model="scarx-hybrid", base_url="http://127.0.0.1:8775/v1", **scope
        )
        observer.post_tool(
            tool_name="read_file", args={"path": "missing"},
            result={"not_found": True}, tool_call_id=str(turn), **scope
        )
        observer.pre_api(
            model="other-model", base_url="http://127.0.0.1:8775/v1", **scope
        )
    assert not observer.routes
    assert not observer.contexts
    assert not observer.previous
    assert not observer.seen_calls
    assert not observer.probed


def test_only_one_probe_per_turn_across_new_signatures_and_api_requests():
    attempts = []
    observer = hooks.ToolExperienceObserver(
        lambda payload: None, probe=lambda *args: attempts.append(args)
    )
    for i in range(6):
        observer.pre_api(
            model="scarx-hybrid", base_url="http://127.0.0.1:8775/v1",
            session_id="s", turn_id="t",
            request={"tools": [{"function": {"name": "read_file"}}]},
        )
        observer.post_tool(
            tool_name="read_file", args={"path": f"wrong-{i // 3}/README.md"},
            result={"not_found": True}, tool_call_id=str(i),
            session_id="s", turn_id="t",
        )
    assert len(attempts) == 1
    assert len(observer.probed) == 1


def test_observer_failure_never_blocks_normal_tool_return():
    def fail(_):
        raise OSError("gateway offline")

    observer = hooks.ToolExperienceObserver(fail)
    observer.pre_api(
        model="scarx-hybrid",
        base_url="http://127.0.0.1:8775/v1",
        session_id="s",
        turn_id="t",
    )
    observer.post_tool(
        tool_name="read_file",
        result="actual result",
        tool_call_id="one",
        session_id="s",
        turn_id="t",
    )
    assert observer.failures == 1


def test_video_tool_uses_public_task_session_metadata(monkeypatch):
    p = PATH.with_name("__init__.py")
    spec = importlib.util.spec_from_file_location("scarx_video_public_session", p)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ctx = SimpleNamespace(
        register_hook=lambda *a, **kw: None, register_tool=lambda *a, **kw: None
    )
    module.register(ctx)
    module._tool_observer.pre_api(
        model="scarx-hybrid",
        base_url="http://127.0.0.1:8775/v1",
        session_id="actual-session",
        turn_id="turn",
        task_id="actual-task",
    )
    captured = []
    monkeypatch.setattr(
        module,
        "_backend",
        lambda: SimpleNamespace(
            forward=lambda path, body: captured.append(body) or {"ok": True}
        ),
    )
    module.handler({"action": "watch", "hwnd": 123}, task_id="actual-task")
    assert captured[-1]["scarx_session_id"] == "actual-session"


def test_semantic_timestamp_pid_and_cwd_changes_are_not_erased_from_result():
    common = ("read_file", {"path": "a"})
    first = hooks.tool_observation(
        *common, {"timestamp": 1, "pid": 11, "cwd": "a"}, "scope", "one"
    )
    for key, value in [("timestamp", 2), ("pid", 12), ("cwd", "b")]:
        source = {"timestamp": 1, "pid": 11, "cwd": "a"}
        source[key] = value
        changed = hooks.tool_observation(*common, source, "scope", "two")
        assert first["result_sha256"] != changed["result_sha256"]
    a = hooks.tool_observation(
        "terminal", {}, {"output": "ok", "exit_code": 0, "duration": 1}, "s", "a"
    )
    b = hooks.tool_observation(
        "terminal", {}, {"output": "ok", "exit_code": 0, "duration": 2}, "s", "b"
    )
    assert a["result_sha256"] == b["result_sha256"]


def ask_fixture(engine):
    for i in range(3):
        engine.observe_tool_experience(payload(i))
    engine.model.mode = "readout_only"
    assert engine.step() == "review"
    engine.model.decide = lambda seq, allowed: ("defer", dict.fromkeys(ACTIONS, 1 / 6))
    assert engine.step() == "ask"
    return engine.state["question"]


def test_negative_owner_feedback_penalizes_action_and_does_not_claim_resolved(engine):
    question = ask_fixture(engine)
    decision = engine.state["decisions"][question["decision_id"]]
    target = ACTIONS.index("ask")
    before = float(engine.model.forward(decision["sequence"])[0][target])
    result = engine.feedback(
        question["id"], text="This question was not useful.", helpful=False
    )
    assert result["learning"]["accepted"] and result["learning"]["reward"] == -1
    after = float(engine.model.forward(decision["sequence"])[0][target])
    assert after < before
    assert engine.state["goals"][question["goal_id"]]["status"] == "deferred"
    assert engine.incidents.summary()["recent"][0]["root_cause"] == "unresolved"
    assert not engine.incidents.summary()["recent"][0]["last_feedback"][
        "factual_cause_verified"
    ]


def test_skipped_incident_question_is_not_a_negative_example(engine):
    question = ask_fixture(engine)
    before = engine.model.dumps()
    result = engine.feedback(question["id"], text="bỏ qua", helpful=True)
    assert result["skipped"] and engine.model.dumps() == before
    assert engine.incidents.summary()["recent"][0]["status"] == "owner_skipped"


def test_unrated_reply_is_neither_negative_reward_nor_factual_verification(engine):
    question = ask_fixture(engine)
    before = engine.model.updates
    result = engine.feedback(question["id"], text="I will check the source later.")
    assert result["learning"] is None and engine.model.updates == before
    incident = engine.incidents.summary()["recent"][0]
    assert (
        incident["status"] == "owner_guidance_received_unrated"
        and incident["root_cause"] == "unresolved"
    )


def test_readonly_probe_uses_recorded_scope_and_never_exports_file_contents():
    calls = []
    ctx = SimpleNamespace(
        session_cwd=lambda: r"C:\Project",
        dispatch_tool=lambda name, args, **kw: calls.append((name, args, kw))
        or '{"content":"private project text","total_lines":1}',
    )
    probe = hooks.readonly_probe(
        ctx,
        "read_file",
        {"path": r"C:\Wrong\README.md"},
        '{"not_found":true}',
        {"file_tool": True},
        "actual-task",
    )
    assert probe["read_ok"] and probe["semantic_resolution_verified"] is False
    assert calls[0][1]["path"] == r"C:\Project\README.md"
    assert "private project text" not in str(probe)
    assert (
        hooks.readonly_probe(
            ctx,
            "read_file",
            {"path": "README.md"},
            '{"not_found":true}',
            {"file_tool": True},
            "a",
        )
        is None
    )
    assert (
        hooks.readonly_probe(
            ctx,
            "read_file",
            {"path": r"C:\Wrong\secret.txt"},
            '{"not_found":true}',
            {"file_tool": True},
            "a",
        )
        is None
    )
    assert (
        hooks.readonly_probe(
            ctx,
            "read_file",
            {"path": r"C:\Wrong\README.md"},
            '{"not_found":true}',
            {"file_tool": False},
            "a",
        )
        is None
    )


def test_optional_probe_supports_hypothesis_but_is_not_a_training_label(engine):
    ctx = SimpleNamespace(
        session_cwd=lambda: r"C:\Project",
        dispatch_tool=lambda *a, **kw: '{"content":"project description"}',
    )
    probe = hooks.readonly_probe(
        ctx,
        "read_file",
        {"path": r"C:\Wrong\README.md"},
        '{"not_found":true}',
        {"file_tool": True},
        "task",
    )
    for i in range(3):
        p = payload(i)
        if i == 2:
            p["probe"] = probe
        result = engine.observe_tool_experience(p)
    assert result["incident"]["status"] == "probe_supported_pending_owner"
    assert engine.model.updates == 0
    assert (
        engine.store.search("Diagnostic", stream="diagnostics")[0]["status"]
        == "observed"
    )


def test_duplicate_hook_call_ids_do_not_launch_an_extra_probe():
    probes = []
    sent = []
    observer = hooks.ToolExperienceObserver(sent.append, lambda *a: probes.append(True))
    observer.pre_api(
        model="scarx-hybrid",
        base_url="http://127.0.0.1:8775/v1",
        session_id="s",
        turn_id="t",
        request={"tools": [{"function": {"name": "read_file"}}]},
    )
    for _ in range(3):
        observer.post_tool(
            tool_name="read_file",
            args={"path": "x"},
            result="same",
            tool_call_id="same-call",
            session_id="s",
            turn_id="t",
        )
    assert len(sent) == 1 and not probes


def test_new_incidents_do_not_reuse_a_resolved_goal_for_same_tool(engine):
    for i in range(3):
        first = engine.observe_tool_experience(payload(i, scope="turn1"))
    old_id = engine.incidents.summary()["recent"][0]["goal_id"]
    engine.state["goals"][old_id]["status"] = "resolved_human"
    for i in range(3):
        second = engine.observe_tool_experience(payload(i, scope="turn2"))
    records = engine.incidents.summary()["recent"]
    ids = {r["goal_id"] for r in records}
    assert (
        len(ids) == 2
        and engine.state["goals"][next(x for x in ids if x != old_id)]["status"]
        == "open"
    )


@pytest.mark.parametrize(
    "helpful,text",
    [
        (False, "This clarification was not useful."),
        (True, "Confirmed useful."),
        (None, "Unrated guidance."),
        (None, "bỏ qua"),
    ],
)
def test_feedback_and_question_closure_are_atomic_on_checkpoint_failure(
    engine, monkeypatch, helpful, text
):
    question = ask_fixture(engine)
    before = engine.model.dumps(engine.state)
    original = engine.store.save_snapshot

    def fail(*a, **kw):
        raise OSError("controlled checkpoint fault")

    monkeypatch.setattr(engine.store, "save_snapshot", fail)
    with pytest.raises(OSError):
        engine.feedback(question["id"], text=text, helpful=helpful)
    assert engine.model.dumps(engine.state) == before
    assert engine.state["question"]["id"] == question["id"]
    assert "last_feedback" not in engine.incidents.summary()["recent"][0]
    monkeypatch.setattr(engine.store, "save_snapshot", original)
    engine.feedback(question["id"], text=text, helpful=helpful)
    assert engine.state["question"] is None
    assert engine.model.updates == (
        1 if helpful is not None and text != "bỏ qua" else 0
    )
