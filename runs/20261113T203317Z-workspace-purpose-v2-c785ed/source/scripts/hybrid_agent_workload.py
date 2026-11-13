"""Real Hermes agent/compaction in a private home; run under its PM test Python."""

import argparse
import json
import os
from pathlib import Path
import sys
import time
import uuid
import shutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    job = json.loads(args.job.read_text(encoding="utf-8"))
    if args.out.exists():
        raise ValueError("Refuse overwrite actual agent result")
    source = Path(job["hermes_source"])
    sys.path.insert(0, str(source))
    home = Path(job["home"])
    home.mkdir(parents=True, exist_ok=True)
    os.environ["HERMES_HOME"] = str(home)
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["TERMINAL_CWD"] = job["cwd"]
    from agent.runtime_cwd import set_session_cwd

    set_session_cwd(job["cwd"])
    from hermes_yaml import safe_dump

    nonce = os.environ.get("SCARX_AUDIT_NONCE")
    provider = {
        "name": "SCAR-X isolated engineering",
        "api": "http://127.0.0.1:8775/v1",
        "api_key": "local",
        "context_length": 16384,
        "default_model": "scarx-hybrid",
        "discover_models": False,
        "extra_headers": {"X-SCARX-Audit": nonce} if nonce else {},
    }
    config = {
        "model": {
            "default": "scarx-hybrid",
            "provider": "custom:scarx-local",
            "context_length": 16384,
        },
        "providers": {"scarx-local": provider},
        "model_overrides": {
            "custom:scarx-local": {
                "scarx-hybrid": {"context_window": 16384, "supports_tools": True}
            }
        },
        "terminal": {"backend": "local"},
        "tools": {"tool_search": {"enabled": "on"}},
        "web": {"search_backend": "searxng"},
        "compression": {"enabled": True},
        "plugins": {"enabled": ["audit-read-only"]},
    }
    # Load this stdlib-only helper directly; importing scarx.__init__ would
    # needlessly require NumPy in Hermes' separately managed Python environment.
    import importlib.util

    routing_path = (
        Path(__file__).resolve().parents[1] / "src/scarx/hybrid/tool_routing.py"
    )
    spec = importlib.util.spec_from_file_location("scarx_audit_routing", routing_path)
    routing = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(routing)
    workspace_prelude = routing.workspace_prelude

    request = workspace_prelude(
        [
            {"role": "system", "content": "Current working directory: " + job["cwd"]},
            {"role": "user", "content": "Read this workspace"},
        ],
        [{"function": {"name": "terminal"}}],
    )
    plugin = home / "plugins/audit-read-only"
    plugin.mkdir(parents=True, exist_ok=True)
    (plugin / "plugin.yaml").write_text(
        "name: audit-read-only\nversion: 1.0.0\ndescription: Isolated read-only engineering scope\n",
        encoding="utf-8",
    )
    shutil.copyfile(
        Path(__file__).resolve().parents[1]
        / "integrations/hermes/compat_tests/read_only_audit_plugin.py",
        plugin / "__init__.py",
    )
    (plugin / "scope.json").write_text(
        json.dumps(
            {
                "cwd": job["cwd"],
                "terminal_command": json.loads(request["arguments"])["command"],
            }
        ),
        encoding="utf-8",
    )
    (home / "config.yaml").write_text(safe_dump(config), encoding="utf-8")
    from run_agent import AIAgent

    started = time.monotonic()
    report = {
        "job": job["id"],
        "started": time.time(),
        "operation": job.get("operation", "workspace"),
        "source": "real Hermes AIAgent; no substitute answers",
    }
    stream = []

    def on_delta(text):
        if text and not stream:
            report["first_visible_delta_seconds"] = time.monotonic() - started
        if text:
            stream.append(text)

    try:
        agent = AIAgent(
            model="scarx-hybrid",
            provider="custom:scarx-local",
            requested_provider="custom:scarx-local",
            base_url="http://127.0.0.1:8775/v1",
            api_key="local",
            cwd=job["cwd"],
            enabled_toolsets=["terminal", "file", "web"],
            skip_memory=True,
            skip_background_review=True,
            quiet_mode=True,
            max_tokens=512,
            max_iterations=10,
            run_budget_seconds=75,
            session_id="engineering-" + uuid.uuid4().hex,
            stream_delta_callback=on_delta,
            request_overrides=(
                {"extra_headers": {"X-SCARX-Audit": nonce}} if nonce else None
            ),
        )
        report["initialization_seconds"] = time.monotonic() - started
        report["eager_tools"] = [t["function"]["name"] for t in agent.tools]
        report["guard"] = {
            "hard_stop": agent._tool_guardrails.config.hard_stop_enabled,
            "repeat_limit": agent._tool_guardrails.config.no_progress_block_after,
            "web_limit": agent._tool_guardrails.config.loop_caps.max_web_searches,
        }
        if report["operation"] == "compression":
            agent.run_conversation("Reply exactly READY. Do not use tools.")
            history = [
                {
                    "role": "user",
                    "content": "Project receipt: code is JADE-472. The port is initially8123.",
                },
                {"role": "assistant", "content": "Recorded JADE-472 and port8123."},
            ]
            for i in range(25):
                history += [
                    {
                        "role": "user",
                        "content": (
                            f"Unrelated archived batch {i}, contains no configuration update. "
                            * 45
                        ),
                    },
                    {"role": "assistant", "content": f"Archived batch {i}."},
                ]
            history += [
                {
                    "role": "user",
                    "content": "Update only port: 9237. Keep original project receipt code.",
                },
                {
                    "role": "assistant",
                    "content": "Port updated9237; receipt code retained.",
                },
            ]
            report["before_messages"] = len(history)
            report["before_characters"] = sum(len(m["content"]) for m in history)
            compacted, system = agent._compress_context(
                history, None, approx_tokens=20000, force=True
            )
            report["after_messages"] = len(compacted)
            report["after_characters"] = sum(
                len(str(m.get("content", ""))) for m in compacted
            )
            report["compacted_history"] = compacted
            report["history_actually_reduced"] = (
                report["after_characters"] < report["before_characters"]
            )
            report["compression_completed_without_64K_error"] = True
            response = agent.run_conversation(
                'Return JSON only with the original receipt code and current port: {"code":"...","port":number}.',
                conversation_history=compacted,
                system_message=system,
            )
            report["answer"] = response.get("final_response", "")
            report["facts_retained_in_answer"] = (
                "JADE-472" in report["answer"] and "9237" in report["answer"]
            )
            report["transcript"] = response.get("messages", [])
        else:
            response = agent.run_conversation(job["prompt"])
            report["answer"] = response.get("final_response", "")
            report["transcript"] = response.get("messages", [])
            tools = [
                tc.get("function", {}).get("name")
                for m in report["transcript"]
                for tc in (m.get("tool_calls") or [])
            ]
            report["tool_calls"] = tools
            report["local_read_used"] = bool(
                {"terminal", "read_file", "search_files"} & set(tools)
            )
            report["web_used"] = "web_search" in tools
            report["expected_terms_present"] = {
                s: s.casefold() in report["answer"].casefold()
                for s in job.get("expected_terms", [])
            }
        report["completed"] = True
        report["answer_producer"] = (
            "runtime_controlled_halt"
            if report.get("answer", "").startswith("Trạng thái từ runtime SCAR-X:")
            else "VLM_or_Hermes"
        )
        prompt = getattr(agent, "_cached_system_prompt", None) or ""
        report["workspace_prompt_lines"] = [
            s for s in prompt.splitlines() if "Current working directory:" in s
        ]
    except Exception as exc:
        report["completed"] = False
        report["error"] = type(exc).__name__ + ": " + str(exc)
    report["seconds"] = time.monotonic() - started
    args.out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in ("transcript", "compacted_history")
            },
            ensure_ascii=True,
        ),
        flush=True,
    )
    return 0 if report["completed"] else 1


if __name__ == "__main__":
    sys.exit(main())
