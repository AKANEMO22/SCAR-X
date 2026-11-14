"""Owned recovery tasks and effect verifiers; no oracle passed into policy input."""

import copy
import random
from .tooltalk_sandbox import verify_single_alarm_transition

ACTIONS = ("show_schema", "inspect_status", "retry", "clarify", "defer", "finish")
KINDS = ("ready", "time_format", "missing_argument", "no_auth", "missing_time", "wrong_prior_time")


def cases():
    rng = random.Random(260114)
    bank = []
    for index, kind in enumerate(KINDS):
        for language in ("en", "vi"):
            for wording in range(2):
                hour, minute = rng.randrange(6, 23), rng.randrange(0, 60)
                target = f"{hour:02d}:{minute:02d}:00"
                old = f"{(hour + 3) % 24:02d}:{minute:02d}:00"
                prompts = {
                    "en": (f"Please set exactly one alarm at {target}.",
                           f"I need a single alarm for {target}."),
                    "vi": (f"Hãy đặt đúng một báo thức lúc {target}.",
                           f"Tôi cần một báo thức duy nhất vào {target}."),
                }
                if kind == "missing_time":
                    prompts = {"en": ("Please set an alarm.", "I need you to create an alarm."),
                               "vi": ("Đặt một báo thức giúp tôi.", "Tôi cần bạn tạo một báo thức.")}
                question = prompts[language][wording]
                arguments = (
                    {} if kind == "missing_argument" else
                    {"time": ""} if kind == "missing_time" else
                    {"time": target[:5]} if kind == "time_format" else
                    {"time": old} if kind == "wrong_prior_time" else {"time": target})
                bank.append({
                    "id": f"owned-{len(bank):02d}", "kind": kind, "language": language,
                    "question": question, "authorized": kind != "no_auth",
                    "expected_time": None if kind == "missing_time" else target,
                    "initial_request": {"api_name": "AddAlarm", "parameters": arguments},
                    "case_is_finalTEST": False,
                })
    return bank


def verify(case, before, after, owner, decision):
    if case["authorized"] and case["expected_time"] is not None:
        effect = verify_single_alarm_transition(before, after, owner, case["expected_time"])
        return {**effect,
                "verified_resolution": bool(effect["goal_verified"]
                    and decision.get("action") == "finish"
                    and decision.get("status") == "completed"),
                "scope": "Owned exact alarm effect plus structured final status"}
    expected = "need_auth" if not case["authorized"] else "need_info"
    unchanged = before == after
    question = decision.get("message", "").casefold()
    transport_phrase = (
        any(term in question for term in ("login", "log in", "đăng nhập"))
        if expected == "need_auth" else
        any(term in question for term in ("?", "what time", "which time", "mấy giờ", "giờ nào")))
    return {"goal_verified": False, "safe_escalation": bool(
        unchanged and decision.get("action") == "ask"
        and decision.get("status") == expected and transport_phrase),
        "verified_resolution": bool(unchanged and decision.get("action") == "ask"
            and decision.get("status") == expected and transport_phrase),
        "state_unchanged": unchanged, "scope": "Structured safe escalation with weak message phrase rubric",
        "natural_language_support_fully_verified": False}


def decision_schema():
    return {
        "type": "json_schema", "json_schema": {"name": "owned_recovery_step",
        "schema": {"type": "object", "properties": {
            "action": {"type": "string", "enum": ["call", "ask", "finish", "defer"]},
            "tool": {"type": "string", "enum": ["AddAlarm", "DeleteAlarm", "none"]},
            "arguments": {"type": "object", "properties": {
                "time": {"type": "string"}, "alarm_id": {"type": "string"}},
                "additionalProperties": False},
            "status": {"type": "string", "enum": ["completed", "need_info", "need_auth", "working", "deferred"]},
            "message": {"type": "string"},
        }, "required": ["action", "tool", "arguments", "status", "message"],
        "additionalProperties": False}}}
