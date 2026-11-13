"""Isolated audited method extraction, never upstream imports/embeddings/tools."""

import ast
from collections import deque
import copy
import hashlib
import json
import logging
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import uuid
import traceback
from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from scarx.evaluation.tooltalk_schema import normalize_tool_schema


def main():
    parent = ROOT / "runs/20261113T233305Z-tooltalk-intake-v1-08df3e"
    intake = json.loads((parent / "manifest.json").read_text(encoding="utf-8"))
    upstream = Path(intake["local_snapshot"])
    run = ROOT / "runs" / (
        time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-tooltalk-harness-p0-" + uuid.uuid4().hex[:6])
    run.mkdir()
    globals()["_CURRENT_RUN"] = run
    (run / "audit-script.py").write_bytes(Path(__file__).read_bytes())
    (run / "protocol.md").write_bytes((ROOT / "experiments/hybrid-tool-benchmark-harness-p0-v1.md").read_bytes())
    sources = {}

    def method(relative, clsname, name, globals_):
        raw = (upstream / relative).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == intake["file_sha256"][relative]
        sources[relative] = intake["file_sha256"][relative]
        tree = ast.parse(raw.decode("utf-8"))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == clsname)
        node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
        # Preserve exact extracted upstream source and attribution. Only remove
        # its descriptor decorator for direct invocation, not its function body.
        lines = raw.decode("utf-8").splitlines(keepends=True)
        (run / (clsname + "-" + name + ".py.txt")).write_text(
            "".join(lines[node.lineno-1:node.end_lineno]), encoding="utf-8")
        if any(isinstance(n, (ast.Import, ast.ImportFrom)) for n in ast.walk(node)):
            raise ValueError("Unexpected import inside isolated method")
        allowed_calls = {"dict", "list", "len", "deque", "items", "append", "popleft",
                         "extend", "debug", "dumps", "copy", "compare_api_calls", "is_action"}
        for n in ast.walk(node):
            if isinstance(n, ast.Call):
                call = n.func.id if isinstance(n.func, ast.Name) else (
                    n.func.attr if isinstance(n.func, ast.Attribute) else "")
                if call not in allowed_calls:
                    raise ValueError("Unexpected method call: " + call)
        node = copy.deepcopy(node)
        node.decorator_list = []
        module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
        namespace = {"__builtins__": {"dict": dict, "list": list, "len": len, "bool": bool}, **globals_}
        exec(compile(module, relative, "exec"), namespace)
        return namespace[name]

    schema_method = method("src/tooltalk/apis/api.py", "API", "to_openai_doc", {})
    schema = schema_method(SimpleNamespace(
        __name__="OwnedRead", description="Fixture only",
        parameters={"path": {"type": "string", "description": "Owned source", "required": True}}))
    repaired = normalize_tool_schema(schema)
    schema_result = {
        "original": schema, "repaired": repaired,
        "original_empty_args_accepted": Draft202012Validator(schema["parameters"]).is_valid({}),
        "repaired_empty_args_accepted": Draft202012Validator(repaired["parameters"]).is_valid({}),
        "repaired_valid_args_accepted": Draft202012Validator(repaired["parameters"]).is_valid({"path": "README.md"}),
    }
    score = method("src/tooltalk/evaluation/tool_executor.py", "ToolExecutor", "evaluate_predictions",
                   {"deque": deque, "json": json, "logger": logging.getLogger("isolated-tool-score")})
    oracle = SimpleNamespace(
        compare_api_calls=lambda a, b: a["request"] == b["request"] and a["exception"] == b["exception"],
        is_action=lambda name: name == "OwnedMutation",
    )
    correct = {"request": {"api_name": "OwnedRead", "parameters": {"path": "README.md"}},
               "response": {"value": 9237}, "exception": None}
    failed_read = {"request": {"api_name": "OwnedRead", "parameters": {"path": "missing.md"}},
                   "response": None, "exception": "FileNotFound"}
    mutation = {"request": {"api_name": "OwnedMutation", "parameters": {}},
                "response": {"changed": True}, "exception": None}
    failed_mutation = {**mutation, "response": None, "exception": "Denied"}
    fixture_predictions = {
        "correct_only": [correct],
        "correct_plus24_failed_identical_reads": [correct] + [failed_read] * 24,
        "correct_plus_successful_unintended_mutation": [correct, mutation],
        "correct_plus24_failed_mutations": [correct] + [failed_mutation] * 24,
        "no_correct_call": [failed_read],
    }
    results = {}
    for name, predictions in fixture_predictions.items():
        conversation = {"conversation": [
            {"role": "user", "text": "Owned fixture"},
            {"role": "assistant", "apis": [copy.deepcopy(correct)],
             "predictions": [dict(copy.deepcopy(p), role="api") for p in predictions]},
        ]}
        result = score(oracle, conversation)
        results[name] = result
    assert schema_result["original_empty_args_accepted"] is True
    assert schema_result["repaired_empty_args_accepted"] is False
    assert schema_result["repaired_valid_args_accepted"] is True
    assert results["correct_only"]["metrics"]["success"] is True
    assert results["correct_plus24_failed_identical_reads"]["metrics"]["success"] is True
    assert results["correct_plus_successful_unintended_mutation"]["metrics"]["success"] is False
    assert results["correct_plus24_failed_mutations"]["metrics"]["success"] is True
    assert results["no_correct_call"]["metrics"]["success"] is False
    report = {
        "upstream_revision": intake["revision"], "source_sha256": sources,
        "schema_validation": schema_result, "scorer_fixtures": results,
        "verdict": "DO_NOT_USE_PUBLISHER_SUCCESS_ALONE_FOR_SCARX_ERROR_RECOVERY",
        "scope": "Actual isolated upstream pure methods with fixture comparator; not API execution or a model score",
        "upstream_whole_modules_executed": False, "loaded_embedding_or_LLM": False,
        "live_model_modified": False,
    }
    for name in ("LICENSE", "NOTICE"):
        (run / name).write_bytes((upstream / name).read_bytes())
    (run / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (run / "audit-script.py").write_bytes(Path(__file__).read_bytes())
    (run / "protocol.md").write_bytes((ROOT / "experiments/hybrid-tool-benchmark-harness-p0-v1.md").read_bytes())
    print(json.dumps({"run_id": run.name,
        "schema": {k:v for k,v in schema_result.items() if k not in ("original", "repaired")},
        "successes": {k:v["metrics"]["success"] for k,v in results.items()}}))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        if "_CURRENT_RUN" in globals():
            (_CURRENT_RUN / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
