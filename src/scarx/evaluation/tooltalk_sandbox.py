"""Isolated ToolTalk action execution; no serving imports, embeddings or network.

Original ToolTalk modules are MIT, Microsoft Corporation; provenance and original
LICENSE/NOTICE are retained by the caller. Only its optional semantic comparator
dependency is replaced with a fail-closed sentinel. API action code is unchanged.
"""

import ast
import copy
from datetime import datetime
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import re
import random
import sys
from types import ModuleType

from jsonschema import Draft202012Validator
from .tooltalk_schema import normalize_tool_schema, repair_pinned_calendar_enum

FORMAT = "scarx-tooltalk-sandbox-v1"


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":"))


def reject_semantic_comparator(*args, **kwargs):
    raise RuntimeError("Embedding comparator excluded; independent scoring required")


def load_executor(snapshot, manifest):
    """Use a fresh worker. Verify pinned bytes before loading original API code."""
    snapshot = Path(snapshot).resolve()
    if any(name == "tooltalk" or name.startswith("tooltalk.") for name in sys.modules):
        raise RuntimeError("ToolTalk adapter requires a fresh isolated worker")
    for relative, expected in manifest["file_sha256"].items():
        raw = (snapshot / relative).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("Upstream source/data hash mismatch: " + relative)
    source = snapshot / "src/tooltalk/apis/utils.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    functions = [
        n for n in tree.body if isinstance(n, ast.FunctionDef)
        and n.name in ("verify_phone_format", "verify_email_format")
    ]
    if {n.name for n in functions} != {"verify_phone_format", "verify_email_format"}:
        raise ValueError("Original validators missing")
    for node in ast.walk(ast.Module(body=functions, type_ignores=[])):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise ValueError("Unexpected validator import")
        if isinstance(node, ast.Call) and not (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name) and node.func.value.id == "re"
            and node.func.attr == "match"
        ):
            raise ValueError("Unexpected validator call")
    module = ModuleType("tooltalk.apis.utils")
    module.__file__ = str(source)
    module.__package__ = "tooltalk.apis"
    module.__dict__.update(re=re, semantic_str_compare=reject_semantic_comparator)
    # Trusted pinned source only, never model/user argument text.
    exec(compile(ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[])),
                 str(source), "exec"), module.__dict__)
    sys.modules[module.__name__] = module
    sys.path.insert(0, str(snapshot / "src"))
    executor = importlib.import_module("tooltalk.evaluation.tool_executor")
    if any(n.startswith(("sent2vec", "torch", "transformers")) for n in sys.modules):
        raise RuntimeError("Unexpected extra neural dependency")
    return executor.ToolExecutor


def bounded_arguments(value, depth=0):
    if depth > 6:
        raise ValueError("Arguments exceed nesting budget")
    if isinstance(value, dict):
        if len(value) > 32 or any(not isinstance(k, str) or len(k) > 120 for k in value):
            raise ValueError("Invalid bounded argument object")
        for item in value.values():
            bounded_arguments(item, depth + 1)
    elif isinstance(value, list):
        if len(value) > 64:
            raise ValueError("Arguments exceed item budget")
        for item in value:
            bounded_arguments(item, depth + 1)
    elif value is not None and type(value) not in (str, int, float, bool):
        raise ValueError("Only JSON argument values are permitted")
    elif isinstance(value, str) and len(value) > 4000:
        raise ValueError("Argument string exceeds budget")


class Sandbox:
    def __init__(self, executor_class, database_dir):
        self.executor_class = executor_class
        self.database_dir = Path(database_dir)
        self.executor = executor_class(init_database_dir=str(database_dir))
        self.schemas = {}
        self.schema_patches = {}
        self.nullable_fields = {}
        for name, cls in self.executor.apis.items():
            schema = normalize_tool_schema(cls.to_openai_doc())
            schema, patched = repair_pinned_calendar_enum(schema)
            if patched:
                self.schema_patches[name] = "pinned name.enum -> event_type.enum"
            defaults = inspect.signature(cls.call).parameters
            nullable = []
            for key, field in schema["parameters"].get("properties", {}).items():
                if key in defaults and defaults[key].default is None:
                    field_type = field.get("type")
                    if isinstance(field_type, str) and field_type != "null":
                        field["type"] = [field_type, "null"]
                        nullable.append(key)
            self.nullable_fields[name] = nullable
            Draft202012Validator.check_schema(schema["parameters"])
            self.schemas[name] = schema
        self.last_input = None
        self.last_result = None
        self.begin_turn()

    def initialize(self, metadata, user):
        self.executor.init_conversation_state(copy.deepcopy(metadata), [], copy.deepcopy(user))
        self.last_input = self.last_result = None
        self.begin_turn()

    def begin_turn(self):
        self.calls = 0
        self.macros = 0
        self.previous = None
        self.streak = 0

    def dispatch(self, request, origin="owned_fixture"):
        if not isinstance(request, dict) or set(request) != {"api_name", "parameters"}:
            raise ValueError("Expected exact JSON tool envelope")
        name, args = request["api_name"], request["parameters"]
        if not isinstance(name, str) or name not in self.schemas:
            raise ValueError("Unknown sandbox API")
        if not isinstance(args, dict):
            raise ValueError("Expected JSON argument object")
        bounded_arguments(args)
        if len(canonical(args).encode()) > 12_000:
            raise ValueError("Arguments exceed byte budget")
        if self.calls >= 12 or self.streak >= 3:
            return {"request": copy.deepcopy(request), "response": None,
                    "exception": "Bounded nonprogress/action guard",
                    "executed": False, "origin": "runtime_guard",
                    "goal_verified": False}
        schema = self.schemas[name]["parameters"]
        extra = set(args) - set(schema.get("properties", {}))
        errors = list(Draft202012Validator(schema).iter_errors(args))
        self.calls += 1
        self.last_input = copy.deepcopy(request)
        if extra or errors:
            result = {"request": copy.deepcopy(request), "response": None,
                      "exception": "Argument validation: " + (
                          "Unknown parameters: " + ",".join(sorted(extra)) if extra
                          else errors[0].message),
                      "executed": False, "origin": "argument_validator",
                      "goal_verified": False}
        else:
            actual_request, response = self.executor.execute_tool(name, copy.deepcopy(args))
            result = {"request": actual_request, **response, "executed": True,
                      "origin": origin, "goal_verified": False}
        signature = hashlib.sha256(canonical({
            k: result[k] for k in ("request", "response", "exception")
        }).encode()).hexdigest()
        self.streak = self.streak + 1 if signature == self.previous else 1
        self.previous = signature
        self.last_result = copy.deepcopy(result)
        return result

    def recover(self, action, origin="owned_fixture"):
        if action not in ("show_schema", "inspect_status", "retry", "clarify", "defer", "finish"):
            raise ValueError("Unknown recovery macro")
        if self.macros >= 2:
            return {"permitted": False, "reason": "Recovery budget exhausted",
                    "origin": "runtime_guard", "goal_verified": False}
        self.macros += 1
        common = {"action": action, "origin": origin, "goal_verified": False}
        if action == "show_schema":
            return {**common, "schema": copy.deepcopy(
                self.schemas[self.last_input["api_name"]]) if self.last_input else None}
        if action == "inspect_status":
            return {**common, "authenticated": self.executor.session_token is not None,
                    "clock": self.executor.now_timestamp.isoformat() if self.executor.now_timestamp else None,
                    "exception": self.last_result["exception"] if self.last_result else None}
        if action == "retry":
            return {**common, "result": self.dispatch(self.last_input, origin) if self.last_input else None}
        return {**common, "status": "awaiting_owner" if action == "clarify" else
                "pending_independent_verification" if action == "finish" else "deferred"}

    def snapshot(self):
        def encode_value(value):
            if isinstance(value, datetime):
                return {"__scarx_type__": "datetime", "iso": value.isoformat()}
            raise TypeError("Unexpected simulator state: " + type(value).__name__)
        state = {
            "format": FORMAT, "databases": self.executor.databases,
            "session_token": self.executor.session_token,
            "clock": self.executor.now_timestamp,
            "tool_rng": {n: t.random.getstate() for n, t in self.executor.inited_tools.items()},
            "guard": {k: getattr(self, k) for k in ("calls", "macros", "previous", "streak")},
            "last_input": self.last_input, "last_result": self.last_result,
        }
        raw = json.dumps(state, sort_keys=True, ensure_ascii=False, allow_nan=False,
                         separators=(",", ":"), default=encode_value).encode()
        if len(raw) > 4_000_000:
            raise ValueError("Simulator checkpoint exceeds budget")
        return raw

    def restore(self, raw):
        def decode_value(value):
            if set(value) == {"__scarx_type__", "iso"} and value["__scarx_type__"] == "datetime":
                return datetime.fromisoformat(value["iso"])
            return value
        if not isinstance(raw, bytes) or len(raw) > 4_000_000:
            raise ValueError("Invalid simulator checkpoint")
        state = json.loads(raw, object_hook=decode_value)
        expected = {"format", "databases", "session_token", "clock", "tool_rng",
                    "guard", "last_input", "last_result"}
        if set(state) != expected or state["format"] != FORMAT:
            raise ValueError("Unsupported simulator checkpoint")
        if set(state["tool_rng"]) - set(self.schemas):
            raise ValueError("Unknown tool RNG state")
        if (not isinstance(state["databases"], dict)
                or set(state["databases"]) != set(self.executor.databases)
                or state["clock"] is not None and not isinstance(state["clock"], datetime)
                or state["session_token"] is not None and not isinstance(state["session_token"], str)
                or set(state["guard"]) != {"calls", "macros", "previous", "streak"}):
            raise ValueError("Invalid simulator state fields")
        for name, upper in (("calls", 12), ("macros", 2), ("streak", 12)):
            if type(state["guard"][name]) is not int or not 0 <= state["guard"][name] <= upper:
                raise ValueError("Invalid simulator guard")
        previous = state["guard"]["previous"]
        if previous is not None and (
                not isinstance(previous, str) or not re.fullmatch("[0-9a-f]{64}", previous)):
            raise ValueError("Invalid simulator progress digest")

        def tuples(value):
            return tuple(tuples(v) for v in value) if isinstance(value, list) else value
        for rng in state["tool_rng"].values():
            random.Random(0).setstate(tuples(rng))
        candidate = self.executor_class(init_database_dir=str(self.database_dir))
        candidate.databases = state["databases"]
        candidate.session_token = state["session_token"]
        candidate.now_timestamp = state["clock"]
        for name, rng in state["tool_rng"].items():
            candidate.get_init_tool(name).random.setstate(tuples(rng))
        self.executor = candidate
        self.last_input, self.last_result = state["last_input"], state["last_result"]
        for name in ("calls", "macros", "previous", "streak"):
            setattr(self, name, state["guard"][name])


def verify_single_alarm_transition(before, after, username, expected_time):
    """Independent closed goal: exactly one new matching alarm, no other edits."""
    old = before.get("Alarm", {}).get(username, {})
    new = after.get("Alarm", {}).get(username, {})
    created = set(new) - set(old)
    if len(created) != 1:
        return {"goal_verified": False, "reason": "Not exactly one new alarm"}
    identifier = next(iter(created))
    valid = (new[identifier].get("time") == expected_time
             and new[identifier].get("alarm_id") == identifier)
    reconstructed = copy.deepcopy(after)
    del reconstructed["Alarm"][username][identifier]
    if username not in before.get("Alarm", {}):
        del reconstructed["Alarm"][username]
    valid = valid and reconstructed == before
    return {"goal_verified": bool(valid),
            "reason": "Independent alarm/time and unchanged-other-state check",
            "verifier": "closed-owned-alarm-transition-v1"}
