"""Behavior of the explicit local first-tool rule; no synthetic model answers."""

import copy
import pytest
from scarx.hybrid.tool_routing import workspace_tool_choice
from scarx.hybrid.tool_routing import (
    workspace_prelude,
    is_context_summary,
    workspace_finalization,
)
import json
import shlex
import subprocess
import sys
from pathlib import Path

TOOLS = [
    {"type": "function", "function": {"name": "terminal"}},
    {"type": "function", "function": {"name": "web_search"}},
]


@pytest.mark.parametrize(
    "query",
    [
        "đọc folder app z fold xem nó làm gì",
        "phân tích workspace đang làm gì",
        "Inspect the current project",
        "Read this folder",
        "Phân tích workspace, chỉ đọc và không tìm trên web",
        "Inspect the project. Do not search the web.",
    ],
)
def test_explicit_project_inspection_starts_with_local_tool_without_changing_history(
    query,
):
    messages = [
        {
            "role": "system",
            "content": "Host OS: Windows\nCurrent working directory: C:\\Project Folder",
        },
        {"role": "user", "content": query},
    ]
    before = copy.deepcopy(messages)
    choice, event = workspace_tool_choice(messages, TOOLS, "auto")
    assert (
        choice["function"]["name"] == "terminal"
        and event["cwd"] == r"C:\Project Folder"
    )
    assert event["selector"].startswith("runtime") and messages == before


@pytest.mark.parametrize(
    "query",
    [
        "Search the web and explain this project",
        "tim tren web ve folder",
        "What is the capital of Vietnam?",
        "chào bạn",
    ],
)
def test_explicit_web_and_non_workspace_turns_are_not_forced(query):
    messages = [
        {"role": "system", "content": "Current working directory: C:\\Repo"},
        {"role": "user", "content": query},
    ]
    assert workspace_tool_choice(messages, TOOLS, "auto") == ("auto", None)


def test_existing_tool_result_and_none_grace_call_never_reforced():
    messages = [
        {"role": "system", "content": "Current working directory: C:\\Repo"},
        {"role": "user", "content": "read the workspace"},
    ]
    assert workspace_tool_choice(messages, TOOLS, "none") == ("none", None)
    assert workspace_tool_choice(messages, [TOOLS[1]], None) == (None, None)
    assert workspace_tool_choice(messages[1:], TOOLS, None) == (None, None)
    messages += [
        {"role": "assistant", "tool_calls": [{"id": "a"}]},
        {"role": "tool", "content": "README.md"},
    ]
    assert workspace_tool_choice(messages, TOOLS, "auto") == ("auto", None)


def test_runtime_prelude_reads_real_files_once_and_preserves_files(tmp_path):
    (tmp_path / "README.md").write_text("Inventory port9237", encoding="utf-8")
    (tmp_path / ".env").write_text("PRIVATE_KEY=never_read", encoding="utf-8")
    folder = tmp_path / "decoded"
    folder.mkdir()
    (folder / "AndroidManifest.xml").write_text(
        '<manifest package="example.jade"/>', encoding="utf-8"
    )
    messages = [
        {"role": "system", "content": "Current working directory: " + str(tmp_path)},
        {"role": "user", "content": "Read this workspace"},
    ]
    request = workspace_prelude(messages, TOOLS)
    args = json.loads(request["arguments"])
    command = shlex.split(args["command"])
    command[0] = sys.executable
    result = json.loads(
        subprocess.check_output(command, cwd=tmp_path, text=True, encoding="utf-8")
    )
    assert result["cwd"] == str(tmp_path)
    assert result["files"][0]["text"] == "Inventory port9237"
    assert {Path(row["path"]).name for row in result["files"]} == {
        "README.md",
        "AndroidManifest.xml",
    }
    assert (tmp_path / ".env").read_text() == "PRIVATE_KEY=never_read"
    messages += [
        {"role": "assistant", "tool_calls": [{"id": "runtime-read"}]},
        {"role": "tool", "content": json.dumps(result)},
    ]
    assert workspace_prelude(messages, TOOLS) is None


def test_purpose_questions_finish_after_four_results_but_editing_is_not_capped():
    base = [
        {"role": "system", "content": r"Current working directory: C:\Project"},
        {"role": "user", "content": "đọc folder này xem nó làm gì"},
    ]
    three = base + [{"role": "tool", "content": "actual read"}] * 3
    assert workspace_finalization(three, TOOLS) == (None, None)
    four = three + [{"role": "tool", "content": "actual read"}]
    choice, event = workspace_finalization(four, TOOLS)
    assert choice == "none" and event["tool_results"] == 4
    four[1] = {
        "role": "user",
        "content": "đọc folder này xem nó làm gì, không sửa file",
    }
    assert workspace_finalization(four, TOOLS)[0] == "none"
    four.append(
        {
            "role": "user",
            "content": "You just executed tool calls but returned an empty response. Please process the tool results above and continue with the task.",
        }
    )
    assert workspace_finalization(four, TOOLS)[0] == "none"
    assert workspace_prelude(four, TOOLS) is None
    four.pop()
    assert workspace_finalization(four, TOOLS, "required") == ("required", None)
    four[1] = {
        "role": "user",
        "content": "Read this folder and fix the bug; explain its purpose",
    }
    assert workspace_finalization(four, TOOLS) == (None, None)


def test_last_runtime_cwd_wins_over_quoted_project_prose():
    messages = [
        {
            "role": "system",
            "content": "Current working directory: C:\\QuotedExample\nProject docs above\nCurrent working directory: C:\\ActualProject",
        },
        {"role": "user", "content": "read this workspace"},
    ]
    _, event = workspace_tool_choice(messages, TOOLS)
    assert event["cwd"] == r"C:\ActualProject"


def test_only_identified_auxiliary_summary_gets_larger_output_budget():
    summary = "You are a summarization agent creating a context checkpoint.\n## Goal\nTURNS TO SUMMARIZE: prior records"
    assert is_context_summary([{"role": "user", "content": summary}])
    assert not is_context_summary([{"role": "user", "content": summary}], TOOLS)
    assert not is_context_summary(
        [{"role": "user", "content": "Please summarize this folder"}]
    )


def test_large_directory_overview_stays_under_hermes_minimum_inline_budget(tmp_path):
    for i in range(45):
        (tmp_path / ("long_document_" + str(i) + "x" * 90 + ".md")).write_text(
            '"quoted source" ' * 500, encoding="utf-8"
        )
    request = workspace_prelude(
        [
            {
                "role": "system",
                "content": "Current working directory: " + str(tmp_path),
            },
            {"role": "user", "content": "read this workspace"},
        ],
        TOOLS,
    )
    command = shlex.split(json.loads(request["arguments"])["command"])
    command[0] = sys.executable
    raw = subprocess.check_output(command, cwd=tmp_path, text=True, encoding="utf-8")
    data = json.loads(raw)
    assert data["entry_count"] == 45 and data["payload_truncated"]
    assert len(json.dumps({"output": raw}, ensure_ascii=False)) < 8000


def test_read_only_audit_allows_only_project_and_its_own_spillover(tmp_path):
    import importlib.util
    from types import SimpleNamespace

    source = (
        Path(__file__).resolve().parents[1]
        / "integrations/hermes/compat_tests/read_only_audit_plugin.py"
    )
    plugin = tmp_path / "plugin.py"
    plugin.write_bytes(source.read_bytes())
    root = tmp_path / "project"
    root.mkdir()
    home = tmp_path / "home"
    home.mkdir()
    (tmp_path / "scope.json").write_text(
        json.dumps(
            {"cwd": str(root), "home": str(home), "terminal_command": "owned-read"}
        ),
        encoding="utf-8",
    )
    spec = importlib.util.spec_from_file_location("audit_read_policy", plugin)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    callbacks = {}
    module.register(
        SimpleNamespace(register_hook=lambda name, fn: callbacks.update({name: fn}))
    )
    before = callbacks["pre_tool_call"]
    assert (
        before("read_file", {"path": str(home / "cache/spillover/result.txt")}) is None
    )
    assert before("read_file", {"path": str(root / "README.md")}) is None
    assert (
        before("read_file", {"path": str(tmp_path / "other-home/secret.txt")})["action"]
        == "block"
    )
    assert before("terminal", {"command": "python mutate.py"})["action"] == "block"
