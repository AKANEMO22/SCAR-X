"""Reversible disk-plugin install. Never patch Hermes core or select a cloud model."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import yaml

ROOT = Path(__file__).resolve().parents[1]


def main():
    home = Path(os.environ["LOCALAPPDATA"]) / "hermes"
    source = ROOT / "integrations/hermes/hybrid_plugin"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backup = ROOT / "local/hybrid-v1/hermes-backups" / stamp
    backup.mkdir(parents=True)
    mapping = {
        home
        / "desktop-plugins/scarx-hybrid/plugin.js": (source / "plugin.js").read_bytes(),
        home
        / "plugins/scarx-hybrid/__init__.py": (source / "__init__.py").read_bytes(),
        home
        / "plugins/scarx-hybrid/learning_hooks.py": (
            source / "learning_hooks.py"
        ).read_bytes(),
        home
        / "plugins/scarx-hybrid/plugin.yaml": (source / "plugin.yaml").read_bytes(),
        home
        / "plugins/scarx-hybrid/dashboard/plugin_api.py": (
            source / "dashboard/plugin_api.py"
        ).read_bytes(),
        home
        / "plugins/scarx-hybrid/dashboard/manifest.json": (
            source / "dashboard/manifest.json"
        ).read_bytes(),
    }
    runtime_python = ROOT / "local/hybrid-v1/runtime-env/Scripts/python.exe"
    config = {
        "gateway": "http://127.0.0.1:8775",
        "entrypoint": str(ROOT / "run_hybrid.py"),
        "python": str(runtime_python) if runtime_python.exists() else sys.executable,
        "data_dir": str(ROOT / "local/hybrid-v1"),
    }
    mapping[home / "plugins/scarx-hybrid/dashboard/settings.json"] = json.dumps(
        config
    ).encode()
    receipt = {
        "installed": stamp,
        "home": str(home),
        "route": "/scarx-hybrid",
        "core_patches": False,
        "files": {},
    }
    for target, raw in mapping.items():
        relative = target.relative_to(home)
        if not target.resolve().is_relative_to(home.resolve()):
            raise ValueError("Install path escapes Hermes home")
        original = target.read_bytes() if target.exists() else None
        if original == raw:
            continue
        if original is not None:
            saved = backup / relative
            saved.parent.mkdir(parents=True, exist_ok=True)
            saved.write_bytes(original)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        receipt["files"][relative.as_posix()] = {
            "before": hashlib.sha256(original).hexdigest() if original else None,
            "after": hashlib.sha256(raw).hexdigest(),
        }
    path = home / "config.yaml"
    original = path.read_bytes()
    text = original.decode("utf-8-sig")
    current = yaml.safe_load(text)
    plugins = dict(current.get("plugins") or {})
    enabled = plugins.get("enabled", [])
    if not isinstance(enabled, list):
        raise ValueError("Hermes plugin list is not a list")
    if "scarx-hybrid" not in enabled:
        plugins["enabled"] = enabled + ["scarx-hybrid"]
    if isinstance(plugins.get("disabled"), list):
        plugins["disabled"] = [p for p in plugins["disabled"] if p != "scarx-hybrid"]
    if plugins != (current.get("plugins") or {}):
        (backup / "config.yaml").write_bytes(original)
        document = yaml.compose(text)
        pairs = document.value
        index = next(
            (i for i, (key, _) in enumerate(pairs) if key.value == "plugins"), None
        )
        block = yaml.safe_dump(
            {"plugins": plugins}, sort_keys=False, allow_unicode=True
        )
        if index is None:
            updated = text + "\n" + block
        else:
            start = text.rfind("\n", 0, pairs[index][0].start_mark.index) + 1
            end = (
                text.rfind("\n", 0, pairs[index + 1][0].start_mark.index) + 1
                if index + 1 < len(pairs)
                else len(text)
            )
            updated = text[:start] + block + "\n" + text[end:]
        expected = dict(current, plugins=plugins)
        if yaml.safe_load(updated) != expected:
            raise ValueError("Unrelated Hermes configuration changed")
        if path.read_bytes() != original:
            raise RuntimeError("Concurrent Hermes config edit")
        fd, temp = tempfile.mkstemp(dir=home, suffix=".scarx-hybrid.tmp")
        with os.fdopen(fd, "wb") as handle:
            handle.write(updated.encode())
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    (backup / "receipt.json").write_text(
        json.dumps(receipt, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "installed": True,
                "route": "/scarx-hybrid",
                "backup": str(backup),
                "core_patches": False,
            }
        )
    )


if __name__ == "__main__":
    main()
