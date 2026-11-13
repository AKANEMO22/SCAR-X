"""Actual file-tool observations in an owned temp home; no model or user data."""

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import importlib.util
from types import SimpleNamespace


def main(output):
    if output.exists():
        raise ValueError("Refuse overwrite")
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "local/hermes-reliability-source"))
    home = Path(tempfile.mkdtemp(prefix="scarx-information-p0-"))
    os.environ["HERMES_HOME"] = str(home)
    os.environ["TERMINAL_CWD"] = str(home)
    os.chdir(home)
    from hermes_yaml import safe_dump
    (home / "config.yaml").write_text(safe_dump({
        "plugins": {"enabled": []},
        "terminal": {"backend": "local"},
        "tools": {"tool_search": {"enabled": "off"}},
    }), encoding="utf-8")
    from model_tools import handle_function_call
    spec = importlib.util.spec_from_file_location(
        "information_hooks", root / "integrations/hermes/hybrid_plugin/learning_hooks.py"
    )
    hooks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hooks)

    report = {"kind": "actual Hermes read_file, manually requested fixtures",
              "cases": [], "errors": []}
    try:
        for i in range(3):
            project = home / str(i) / "project"
            project.mkdir(parents=True)
            alternative = project / "README.md"
            if i == 0:
                alternative.write_text("Owned source. Current port9237.\n", encoding="utf-8")
            elif i == 2:
                alternative.mkdir()
            requested = project / "wrong" / "README.md"
            outputs = [
                handle_function_call("read_file", {"path": str(requested)},
                    task_id=f"information-{i}", session_id=f"information-{i}",
                    turn_id="turn-one", tool_call_id=f"missing-{j}")
                for j in range(3)
            ]
            probe_outputs = []

            def dispatch(tool, args, **kwargs):
                value = handle_function_call(
                    tool, args, session_id=f"information-{i}", turn_id="turn-one",
                    tool_call_id="probe", **kwargs)
                probe_outputs.append(value)
                return value

            # Only fixture project scope is provided manually. The actual file
            # dispatcher and published readonly_probe are executed unchanged.
            probe = hooks.readonly_probe(
                SimpleNamespace(session_cwd=lambda: str(project), dispatch_tool=dispatch),
                "read_file", {"path": str(requested)}, outputs[-1],
                {"file_tool": True}, f"information-{i}",
            )
            report["cases"].append({
                "id": f"case-{i}", "requested_path": str(requested),
                "requested_exists": requested.exists(),
                "alternative_path": str(alternative),
                "alternative_exists": alternative.exists(),
                "alternative_is_file": alternative.is_file(),
                "alternative_is_directory": alternative.is_dir(),
                "missing_outputs": outputs, "probe": probe,
                "probe_outputs": probe_outputs,
            })
        source = home / "bounded.txt"
        source.write_text("".join(f"Owned observation line{i}.\n" for i in range(20)),
                          encoding="utf-8")
        report["bounded_read"] = {
            "source_lines": 20,
            "partial": handle_function_call("read_file",
                {"path": str(source), "offset": 1, "limit": 3},
                task_id="bounded", tool_call_id="partial"),
            "full": handle_function_call("read_file",
                {"path": str(source), "offset": 1, "limit": 100},
                task_id="bounded", tool_call_id="full"),
        }
    except Exception as exc:
        report["errors"].append(type(exc).__name__ + ": " + str(exc))
    with output.open("x", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps({"cases": len(report["cases"]), "errors": report["errors"]}))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("output", type=Path)
    main(p.parse_args().output.resolve())
