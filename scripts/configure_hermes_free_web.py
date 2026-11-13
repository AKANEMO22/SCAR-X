"""Back up and use Hermes config CLI; no credential/core edits or new service."""

from datetime import datetime, timezone
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import yaml

ROOT = Path(__file__).resolve().parents[1]


def main():
    home = Path(os.environ["LOCALAPPDATA"]) / "hermes"
    path = home / "config.yaml"
    raw = path.read_bytes()
    before = yaml.safe_load(raw.decode("utf-8-sig"))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backup = ROOT / "local/hybrid-v1/hermes-free-web-backups" / stamp
    backup.mkdir(parents=True)
    (backup / "config.yaml").write_bytes(raw)
    changes = {
        "web.search_backend": "searxng",
        "web.extract_backend": "exa",
        "web.provider_tier.exa": "free",
    }
    env_path = home / ".env"
    if env_path.exists():
        (backup / "environment.env").write_bytes(env_path.read_bytes())
    subprocess.run(
        [
            str(home / "bin/hermes.exe"),
            "config",
            "set",
            "SEARXNG_URL",
            "http://127.0.0.1:18778",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    expected = copy.deepcopy(before)
    for key, value in changes.items():
        subprocess.run(
            [str(home / "bin/hermes.exe"), "config", "set", key, value],
            check=True,
            capture_output=True,
            text=True,
        )
        target = expected
        for part in key.split(".")[:-1]:
            target = target.setdefault(part, {})
        target[key.split(".")[-1]] = value
    after = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    if after != expected:
        raise ValueError(
            "Unrelated Hermes setting changed; inspect private backup before restoring"
        )
    result = {
        "changed": changes,
        "SEARXNG_URL": "http://127.0.0.1:18778",
        "before_sha256": hashlib.sha256(raw).hexdigest(),
        "after_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "backup": str(backup),
        "API_keys_changed": False,
        "extra_service": "one owned native Python SearXNG child; no Docker/WSL",
        "extract_note": "SearXNG is search-only; extract remains Exa free/keyless",
    }
    (backup / "receipt.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
