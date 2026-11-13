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
    compile(updated, str(source), "exec")
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
                        defer=[entry.name for entry in _scarx_registry.get_all_entries()])
        return ToolSearchConfig.from_raw(_raw)"""
    if search_new in search_text:
        search_updated = search_original
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
    receipt = {
        "backup": str(backup),
        "model": "scarx-hybrid",
        "api": "http://127.0.0.1:8775/v1",
        "buffer": 16384,
        "normal_agent_features_retained": True,
        "original_source_sha256": hashlib.sha256(original).hexdigest(),
        "patched_source_sha256": hashlib.sha256(updated).hexdigest(),
        "tool_disclosure": "existing Hermes bridge, all enabled tool names remain discoverable",
        "tool_search_before": hashlib.sha256(search_original).hexdigest(),
        "tool_search_after": hashlib.sha256(search_updated).hexdigest(),
    }
    for target, before, after in (
        (source, original, updated),
        (search_source, search_original, search_updated),
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
