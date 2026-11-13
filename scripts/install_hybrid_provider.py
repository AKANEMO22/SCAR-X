"""Normal Hermes provider; one scoped floor exception, no prompt/tool stripping."""

from datetime import datetime, timezone
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import yaml

ROOT = Path(__file__).resolve().parents[1]


def main():
    home = Path(os.environ["LOCALAPPDATA"]) / "hermes"
    source = home / "hermes-agent/agent/agent_init.py"
    original = source.read_bytes()
    text = original.decode("utf-8")
    old = "    if _ctx and _ctx < MINIMUM_CONTEXT_LENGTH and not _allow_lmstudio_explicit_below_floor:\n"
    new = """    # SCAR-X uses its measured local working buffer; never advertise a fake64K.
    from urllib.parse import urlsplit as _scarx_urlsplit
    _scarx_url = _scarx_urlsplit(str(agent.base_url or ""))
    _allow_scarx_measured = (
        str(agent.model) == "scarx-hybrid" and _scarx_url.scheme == "http"
        and _scarx_url.hostname in ("127.0.0.1", "localhost")
        and _scarx_url.port == 8775 and _scarx_url.path.rstrip("/") == "/v1"
        and _ctx == 16384 and agent._config_context_length == 16384
    )
    if _ctx and _ctx < MINIMUM_CONTEXT_LENGTH and not (_allow_lmstudio_explicit_below_floor or _allow_scarx_measured):
"""
    if new in text:
        updated = original
    elif text.count(old) == 1:
        updated = text.replace(old, new, 1).encode("utf-8")
    else:
        raise ValueError(
            "Hermes minimum-context source changed; refusing an unreviewed patch"
        )
    guard_old = """        agent._tool_guardrails = ToolCallGuardrailController(
            ToolCallGuardrailConfig.from_mapping(
                _agent_cfg.get("tool_loop_guardrails", {}), platform=platform,
            )
        )"""
    guard_new = """        _loop_cfg = _agent_cfg.get("tool_loop_guardrails", {})
        # Local SCAR-X needs an enforced boundary, not an endless warn-only loop.
        from urllib.parse import urlsplit as _loop_urlsplit
        _loop_url = _loop_urlsplit(str(agent.base_url or ""))
        if (str(agent.model) == "scarx-hybrid" and _loop_url.scheme == "http"
                and _loop_url.hostname in ("127.0.0.1", "localhost")
                and _loop_url.port == 8775 and _loop_url.path.rstrip("/") == "/v1"):
            _loop_cfg = dict(_loop_cfg) if isinstance(_loop_cfg, dict) else {}
            _loop_cfg.update(hard_stop_enabled=True, non_interactive_hard_stop_enabled=True)
            _loop_cfg["hard_stop_after"] = dict(_loop_cfg.get("hard_stop_after") or {},
                exact_failure=3, same_tool_failure=5, idempotent_no_progress=3)
            _loop_cfg["loop_caps"] = dict(_loop_cfg.get("loop_caps") or {}, max_web_searches=6)
            agent.max_iterations = min(agent.max_iterations, 24)
        agent._tool_guardrails = ToolCallGuardrailController(
            ToolCallGuardrailConfig.from_mapping(_loop_cfg, platform=platform)
        )"""
    init_text = updated.decode("utf-8")
    guard_previous = guard_new.replace(
        "            agent.max_iterations = min(agent.max_iterations, 24)\n", ""
    )
    if guard_previous in init_text:
        init_text = init_text.replace(guard_previous, guard_new, 1)
    elif guard_new not in init_text:
        if init_text.count(guard_old) != 1:
            raise ValueError("Hermes tool-loop source changed; review before patch")
        init_text = init_text.replace(guard_old, guard_new, 1)
    updated = init_text.encode("utf-8")
    compile(updated, str(source), "exec")
    compression_source = home / "hermes-agent/agent/conversation_compression.py"
    compression_original = compression_source.read_bytes()
    compression_text = compression_original.decode("utf-8")
    compression_old = (
        "        if aux_context and aux_context < MINIMUM_CONTEXT_LENGTH:\n"
    )
    compression_new = """        # The same measured local16K route can summarize its bounded input;
        # keep the generic64K rejection for all other models/endpoints.
        from urllib.parse import urlsplit as _compression_urlsplit
        _compression_url = _compression_urlsplit(aux_base_url)
        _measured_local_summary = (
            str(aux_model) == "scarx-hybrid" and aux_context == 16384
            and getattr(agent, "_config_context_length", None) == 16384
            and _aux_inherits_main_route(agent, aux_model, aux_base_url)
            and _compression_url.scheme == "http"
            and _compression_url.hostname in ("127.0.0.1", "localhost")
            and _compression_url.port == 8775
            and _compression_url.path.rstrip("/") == "/v1"
        )
        if aux_context and aux_context < MINIMUM_CONTEXT_LENGTH and not _measured_local_summary:
"""
    if compression_new in compression_text:
        compression_updated = compression_original
    elif compression_text.count(compression_old) == 1:
        compression_updated = compression_text.replace(
            compression_old, compression_new, 1
        ).encode("utf-8")
    else:
        raise ValueError(
            "Hermes compression feasibility source changed; review before patch"
        )
    compile(compression_updated, str(compression_source), "exec")
    compressor_source = home / "hermes-agent/agent/context_compressor.py"
    compressor_original = compressor_source.read_bytes()
    compressor_text = compressor_original.decode("utf-8")
    fallback_old = """        verdict = classify_api_error(e)
        kind = _classify_summary_failure(e, verdict)"""
    fallback_new = """        # A small measured local summarizer may truncate; use Hermes' existing
        # redacted extractive fallback in the same transaction instead of retrying
        # until its main16K prompt overflows. Other routes keep their normal policy.
        from urllib.parse import urlsplit as _fallback_urlsplit
        _fallback_url = _fallback_urlsplit(str(self.base_url or ""))
        if (self.model == "scarx-hybrid" and self.context_length == 16384
                and _fallback_url.scheme == "http"
                and _fallback_url.hostname in ("127.0.0.1", "localhost")
                and _fallback_url.port == 8775 and _fallback_url.path.rstrip("/") == "/v1"):
            logger.warning("SCAR-X summary failed; using bounded deterministic recovery: %s", _short_error_text(e))
            return self._build_static_fallback_summary(
                turns_to_summarize, reason="SCAR-X local summary unavailable: " + _short_error_text(e))
        verdict = classify_api_error(e)
        kind = _classify_summary_failure(e, verdict)"""
    if fallback_new not in compressor_text:
        if compressor_text.count(fallback_old) != 1:
            raise ValueError(
                "Hermes summary recovery source changed; review before patch"
            )
        compressor_text = compressor_text.replace(fallback_old, fallback_new, 1)
    compressor_updated = compressor_text.encode("utf-8")
    compile(compressor_updated, str(compressor_source), "exec")
    # Use Hermes' existing progressive tool disclosure. All enabled tools stay
    # callable via tool_search/describe/call; only eager schema loading changes.
    search_source = home / "hermes-agent/tools/tool_search.py"
    search_original = search_source.read_bytes()
    search_text = search_original.decode("utf-8")
    search_old = """        tools_cfg = (getattr(_cfg_mod, loader_name)() or {}).get("tools")
        tools_cfg = tools_cfg if isinstance(tools_cfg, dict) else {}
        return ToolSearchConfig.from_raw(tools_cfg.get("tool_search"))"""
    search_new = """        _full_config = getattr(_cfg_mod, loader_name)() or {}
        tools_cfg = _full_config.get("tools")
        tools_cfg = tools_cfg if isinstance(tools_cfg, dict) else {}
        _raw = tools_cfg.get("tool_search")
        # SCAR-X local profile: same tools, lazy schema disclosure, no64K fiction.
        _model = _full_config.get("model") or {}
        if _model.get("default") == "scarx-hybrid" and _model.get("provider") == "custom:scarx-local":
            from tools.registry import registry as _scarx_registry
            _raw = dict(_raw) if isinstance(_raw, dict) else {}
            _raw.update(enabled="on", listing="off", listing_max_tokens=200,
                        defer=[entry.name for entry in _scarx_registry.get_all_entries()
                               if entry.name not in ("web_search", "web_extract", "read_file", "search_files", "terminal", "scarx_video")])
        return ToolSearchConfig.from_raw(_raw)"""
    search_previous = search_new.replace(
        "defer=[entry.name for entry in _scarx_registry.get_all_entries()\n"
        '                               if entry.name not in ("web_search", "web_extract", "read_file", "search_files", "terminal", "scarx_video")]',
        "defer=[entry.name for entry in _scarx_registry.get_all_entries()]",
    )
    if search_new in search_text:
        search_updated = search_original
    elif (
        search_new.replace(
            '("web_search", "web_extract", "read_file", "search_files", "terminal", "scarx_video")',
            '("web_search", "web_extract")',
        )
        in search_text
    ):
        search_updated = search_text.replace(
            search_new.replace(
                '("web_search", "web_extract", "read_file", "search_files", "terminal", "scarx_video")',
                '("web_search", "web_extract")',
            ),
            search_new,
            1,
        ).encode("utf-8")
    elif search_text.count(search_previous) == 1:
        search_updated = search_text.replace(search_previous, search_new, 1).encode(
            "utf-8"
        )
    elif search_text.count(search_old) == 1:
        search_updated = search_text.replace(search_old, search_new, 1).encode("utf-8")
    else:
        raise ValueError(
            "Hermes tool-disclosure source changed; inspect before patching"
        )
    compile(search_updated, str(search_source), "exec")
    path = home / "config.yaml"
    raw = path.read_bytes()
    config_text = raw.decode("utf-8-sig")
    current = yaml.safe_load(config_text)
    expected = copy.deepcopy(current)
    expected.setdefault("providers", {})["scarx-local"] = {
        "name": "SCAR-X Hybrid",
        "api": "http://127.0.0.1:8775/v1",
        "api_key": "local",
        "transport": "chat_completions",
        "default_model": "scarx-hybrid",
        "discover_models": False,
        "session_affinity_header": "X-SCARX-Session",
        "context_length": 16384,
        "models": {
            "scarx-hybrid": {
                "supports_vision": True,
                "supports_reasoning": False,
                "tool_calling": True,
            }
        },
        "capabilities": {
            "supports_vision": True,
            "supports_reasoning": False,
            "tool_calling": True,
        },
    }
    expected.setdefault("model", {}).update(
        {
            "default": "scarx-hybrid",
            "provider": "custom:scarx-local",
            "context_length": 16384,
            "supports_vision": True,
        }
    )
    for owner in ("scarx-local", "custom:scarx-local"):
        expected.setdefault("model_overrides", {}).setdefault(owner, {})[
            "scarx-hybrid"
        ] = {
            "supports_tools": True,
            "supports_vision": True,
            "supports_reasoning": False,
            "context_window": 16384,
            "model_family": "qwen",
        }
    # Preserve unrelated providers, credentials, plugin list and other settings.
    document = yaml.compose(config_text)
    pairs = document.value
    replacements = []
    for target in ("providers", "model", "model_overrides"):
        index = next(
            (i for i, (key, _) in enumerate(pairs) if key.value == target), None
        )
        block = yaml.safe_dump(
            {target: expected[target]}, sort_keys=False, allow_unicode=True
        )
        if index is None:
            replacements.append((len(config_text), len(config_text), "\n" + block))
        else:
            start = config_text.rfind("\n", 0, pairs[index][0].start_mark.index) + 1
            end = (
                config_text.rfind("\n", 0, pairs[index + 1][0].start_mark.index) + 1
                if index + 1 < len(pairs)
                else len(config_text)
            )
            replacements.append((start, end, block + "\n"))
    for start, end, block in sorted(replacements, reverse=True):
        config_text = config_text[:start] + block + config_text[end:]
    if yaml.safe_load(config_text) != expected:
        raise ValueError("Unrelated config was modified")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backup = ROOT / "local/hybrid-v1/hermes-provider-backups" / stamp
    backup.mkdir(parents=True)
    (backup / "config.yaml").write_bytes(raw)
    (backup / "agent_init.py").write_bytes(original)
    (backup / "tool_search.py").write_bytes(search_original)
    (backup / "conversation_compression.py").write_bytes(compression_original)
    (backup / "context_compressor.py").write_bytes(compressor_original)
    receipt = {
        "backup": str(backup),
        "model": "scarx-hybrid",
        "api": "http://127.0.0.1:8775/v1",
        "buffer": 16384,
        "normal_agent_features_retained": True,
        "original_source_sha256": hashlib.sha256(original).hexdigest(),
        "patched_source_sha256": hashlib.sha256(updated).hexdigest(),
        "tool_disclosure": "local file/terminal/video + web eager for SCAR-X; every other enabled tool remains callable",
        "loop_hard_stop": {
            "no_progress": 3,
            "web_search_cap": 6,
            "scope": "SCAR-X loopback8775 only",
        },
        "compression_measured_16K": True,
        "compression_before": hashlib.sha256(compression_original).hexdigest(),
        "compression_after": hashlib.sha256(compression_updated).hexdigest(),
        "summary_recovery_before": hashlib.sha256(compressor_original).hexdigest(),
        "summary_recovery_after": hashlib.sha256(compressor_updated).hexdigest(),
        "tool_search_before": hashlib.sha256(search_original).hexdigest(),
        "tool_search_after": hashlib.sha256(search_updated).hexdigest(),
    }
    for target, before, after in (
        (source, original, updated),
        (search_source, search_original, search_updated),
        (compression_source, compression_original, compression_updated),
        (compressor_source, compressor_original, compressor_updated),
        (path, raw, config_text.encode()),
    ):
        if target.read_bytes() != before:
            raise RuntimeError("Concurrent Hermes modification")
        if before == after:
            continue
        fd, temp = tempfile.mkstemp(dir=target.parent, suffix=".scarx-provider.tmp")
        with os.fdopen(fd, "wb") as handle:
            handle.write(after)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, target)
    (backup / "receipt.json").write_text(
        json.dumps(receipt, indent=2), encoding="utf-8"
    )
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
