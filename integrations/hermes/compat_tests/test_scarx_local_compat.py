"""Local installation regression tests; run via Hermes scripts/run_tests.sh.

No inference is mocked into a quality result. These exercise real configuration,
agent construction, disclosure and feasibility; end-to-end model tests are separate.
"""

import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import pytest
import hermes_yaml as yaml


@pytest.fixture
def agent(tmp_path):
    home = Path(os.environ["HERMES_HOME"])
    config = {
        "model": {
            "default": "scarx-hybrid",
            "provider": "custom:scarx-local",
            "context_length": 16384,
        },
        "providers": {
            "scarx-local": {
                "name": "SCAR-X Local Test",
                "api": "http://127.0.0.1:8775/v1",
                "api_key": "local",
                "context_length": 16384,
                "default_model": "scarx-hybrid",
                "discover_models": False,
            }
        },
        "model_overrides": {
            "custom:scarx-local": {
                "scarx-hybrid": {"context_window": 16384, "supports_tools": True}
            }
        },
        "terminal": {"backend": "local"},
        "compression": {"enabled": True},
        "tools": {"tool_search": {"enabled": "on"}},
    }
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    from run_agent import AIAgent

    value = AIAgent(
        model="scarx-hybrid",
        provider="custom:scarx-local",
        requested_provider="custom:scarx-local",
        base_url="http://127.0.0.1:8775/v1",
        api_key="local",
        cwd=str(tmp_path),
        enabled_toolsets=["terminal", "file", "web"],
        skip_memory=True,
        quiet_mode=True,
        max_tokens=512,
        max_iterations=8,
    )
    yield value


def test_local_agent_exposes_filesystem_and_enforces_repeat_halt(agent):
    names = {t["function"]["name"] for t in agent.tools}
    assert {"terminal", "read_file", "search_files"} <= names
    guard = agent._tool_guardrails
    assert guard.config.hard_stop_enabled and guard.config.no_progress_block_after == 3
    for _ in range(3):
        assert guard.before_call("web_search", {"query": "unchanged"}).action == "allow"
        guard.after_call(
            "web_search",
            {"query": "unchanged"},
            json.dumps({"success": True, "data": {"web": []}}),
        )
    decision = guard.before_call("web_search", {"query": "unchanged"})
    assert decision.should_halt and decision.code == "idempotent_no_progress_block"


def test_same_measured_main_route_passes_real_compression_feasibility(agent):
    from agent.conversation_compression import check_compression_model_feasibility

    assert agent._config_context_length == 16384
    check_compression_model_feasibility(agent)
    assert 0 < agent.context_compressor.threshold_tokens < 16384


def test_small_local_summary_failure_uses_existing_redacted_recovery(agent):
    compressor = agent.context_compressor
    turns = [
        {"role": "user", "content": "Project code JADE-472. Port9237."},
        {"role": "assistant", "content": "Recorded."},
    ]
    result = compressor._on_summary_failure(
        RuntimeError("summary was truncated"), turns, None, ""
    )
    assert isinstance(result, str) and "deterministic" in result.lower()
    assert "JADE-472" in result and "9237" in result
    assert "SCAR-X local summary unavailable" in result


@pytest.mark.parametrize(
    "model,url",
    [
        ("scarx-hybrid", "http://example.test/v1"),
        ("another-model", "http://127.0.0.1:8775/v1"),
        ("scarx-hybrid", "http://127.0.0.1:9000/v1"),
    ],
)
def test_other_small_compression_routes_keep_the_generic_floor(agent, model, url):
    from agent.conversation_compression import check_compression_model_feasibility

    client = SimpleNamespace(base_url=url, api_key="local")
    with (
        patch(
            "agent.auxiliary_client.get_text_auxiliary_client",
            return_value=(client, model),
        ),
        patch("agent.model_metadata.get_model_context_length", return_value=16384),
    ):
        with pytest.raises(ValueError, match="64,000"):
            check_compression_model_feasibility(agent)
