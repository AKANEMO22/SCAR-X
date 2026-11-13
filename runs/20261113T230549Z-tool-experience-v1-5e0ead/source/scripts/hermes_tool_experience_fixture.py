"""Real Hermes dispatcher + published hooks; manual read-only fault injection."""

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time


def main(out, probe_mode=False):
    if out.exists():
        raise ValueError("Refuse overwrite fixture result")
    root = Path(__file__).resolve().parents[1]
    source = root / "local/hermes-reliability-source"
    sys.path.insert(0, str(source))
    home = Path(tempfile.mkdtemp(prefix="scarx-tool-experience-"))
    project = home / "project"
    project.mkdir()
    if probe_mode:
        (project / "README.md").write_text(
            "Owned engineering project. Port9237.\n", encoding="utf-8"
        )
    os.environ["HERMES_HOME"] = str(home)
    os.environ["TERMINAL_CWD"] = str(project)
    os.chdir(project)
    from hermes_yaml import safe_dump

    (home / "config.yaml").write_text(
        safe_dump(
            {
                "plugins": {"enabled": ["tool-experience-fixture"]},
                "terminal": {"backend": "local"},
                "tools": {"tool_search": {"enabled": "off"}},
            }
        ),
        encoding="utf-8",
    )
    plugin = home / "plugins/tool-experience-fixture"
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text(
        "name: tool-experience-fixture\nversion: 1.0.0\ndescription: Read-only engineering tool observations\n",
        encoding="utf-8",
    )
    (plugin / "learning_hooks.py").write_bytes(
        (root / "integrations/hermes/hybrid_plugin/learning_hooks.py").read_bytes()
    )
    code = """import importlib.util,json,os
from pathlib import Path
from urllib.request import Request,urlopen
def send(payload):
 base='http://127.0.0.1:8775';extra={'X-SCARX-Audit':os.environ['SCARX_AUDIT_NONCE'],'X-SCARX-Backend':'hermes'}
 with urlopen(Request(base+'/api/status?after=2147483647',headers=extra),timeout=3) as r:token=json.load(r)['token']
 with urlopen(Request(base+'/api/tool-experience',data=json.dumps(payload).encode(),headers={**extra,'Content-Type':'application/json','X-SCARX-Token':token}),timeout=3) as r:return json.load(r)
def register(ctx):
 spec=importlib.util.spec_from_file_location('experience_fixture_hooks',Path(__file__).with_name('learning_hooks.py'))
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 if os.environ.get('SCARX_PROBE_FIXTURE')=='1':
  from types import SimpleNamespace
  ctx=SimpleNamespace(register_hook=ctx.register_hook,session_cwd=lambda:os.environ['SCARX_PROBE_PROJECT'],dispatch_tool=ctx.dispatch_tool)
 module.register_observer(ctx,send)
"""
    (plugin / "__init__.py").write_text(code, encoding="utf-8")
    started = time.monotonic()
    report = {
        "source": "real Hermes model_tools.handle_function_call and public hooks; manual fault injection, not a model-generated tool trajectory",
        "home": str(home),
        "results": [],
        "errors": [],
    }
    if probe_mode:
        os.environ["SCARX_PROBE_FIXTURE"] = "1"
        os.environ["SCARX_PROBE_PROJECT"] = str(project)
    try:
        from model_tools import handle_function_call
        from hermes_cli.plugins import discover_plugins
        from hermes_cli.lifecycle import invoke_hook

        discover_plugins(force=True)
        common = {
            "session_id": "experience-engineering",
            "turn_id": "turn-one",
            "base_url": "http://127.0.0.1:8775/v1",
        }
        if probe_mode:
            common["request"] = {"tools": [{"function": {"name": "read_file"}}]}
        missing = (
            project / "wrong/README.md"
            if probe_mode
            else project / "intentionally-missing.txt"
        )
        invoke_hook("pre_api_request", model="other-model", **common)
        ignored = handle_function_call(
            "read_file",
            {"path": str(missing)},
            task_id="experience-engineering",
            session_id=common["session_id"],
            turn_id=common["turn_id"],
            tool_call_id="ignored-other-model",
        )
        report["other_model_result"] = ignored
        invoke_hook("pre_api_request", model="scarx-hybrid", **common)
        for i in range(3):
            result = handle_function_call(
                "read_file",
                {"path": str(missing)},
                task_id="experience-engineering",
                session_id=common["session_id"],
                turn_id=common["turn_id"],
                tool_call_id="actual-read-" + str(i),
            )
            report["results"].append(result)
        report["actual_file_absent"] = not missing.exists()
        report["probe_mode"] = probe_mode
        report["owner_readme_exists"] = (project / "README.md").exists()
    except Exception as exc:
        report["errors"].append(type(exc).__name__ + ": " + str(exc))
    report["seconds"] = time.monotonic() - started
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "results": len(report["results"]),
                "errors": report["errors"],
                "seconds": report["seconds"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    p.add_argument("--probe-mode", action="store_true")
    a = p.parse_args()
    main(a.out.resolve(), a.probe_mode)
