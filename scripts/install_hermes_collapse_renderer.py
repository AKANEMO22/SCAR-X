"""Install only the rebuilt Hermes renderer; preserve native code and backups.

Source fix: respect an explicit closed tool-run instead of retaining its live
ticker. Old hashed assets remain, so restoring the backed-up index/manifest
recovers the prior UI without deleting data or stopping an agent session.
"""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import uuid

ROOT = Path(__file__).resolve().parents[1]


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def atomic_write(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, suffix=".renderer.tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def main():
    build = (ROOT / "local/hermes-renderer-tool-collapse-20261113").resolve()
    hermes = (Path(os.environ["LOCALAPPDATA"]) / "hermes/hermes-agent").resolve()
    target = (
        hermes / "apps/desktop/release/win-unpacked/resources/app.asar.unpacked/dist"
    ).resolve()
    if not build.is_relative_to(
        (ROOT / "local").resolve()
    ) or not target.is_relative_to(hermes):
        raise ValueError("Unexpected renderer path")
    if (
        not (build / "index.html").is_file()
        or not (build / "renderer-manifest.json").is_file()
    ):
        raise ValueError("Build renderer first")
    native = {}
    for name in ("electron-main.mjs", "electron-preload.js", "hermes-build.json"):
        path = target / name
        if not path.is_file():
            raise ValueError("Expected installed native desktop generation: " + name)
        native[name] = sha(path.read_bytes())
    protected = set(native) | {"install-stamp.json", ".hermes-product"}
    files = [path for path in build.rglob("*") if path.is_file()]
    for path in files:
        if path.is_symlink() or not path.resolve().is_relative_to(build):
            raise ValueError("Unsafe renderer build member")
        if (
            path.name in protected
            or "native" in path.relative_to(build).parts
            or path.suffix in (".exe", ".dll", ".node")
        ):
            raise ValueError(
                "Renderer build includes native code; refusing package replacement"
            )
    manifest = json.loads(
        (build / "renderer-manifest.json").read_text(encoding="utf-8")
    )
    for entry in manifest.values():
        path = (build / entry["file"]).resolve()
        if not path.is_relative_to(build) or not path.is_file():
            raise ValueError("Incomplete renderer manifest")
    backup = (
        ROOT
        / "local"
        / (
            "hermes-collapse-renderer-backup-"
            + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "-"
            + uuid.uuid4().hex[:6]
        )
    )
    backup.mkdir(parents=True)
    receipt = {
        "build": str(build),
        "target": str(target),
        "backup": str(backup),
        "native_hashes": native,
        "files": {},
        "rollback": "Restore backed-up index.html and renderer-manifest.json; old hashed assets retained",
    }
    files.sort(key=lambda p: p.name in ("index.html", "renderer-manifest.json"))
    for source in files:
        relative = source.relative_to(build)
        destination = (target / relative).resolve()
        if not destination.is_relative_to(target):
            raise ValueError("Destination escapes renderer")
        raw = source.read_bytes()
        before = destination.read_bytes() if destination.exists() else None
        if raw == before:
            continue
        if before is not None:
            saved = backup / relative
            saved.parent.mkdir(parents=True, exist_ok=True)
            saved.write_bytes(before)
        receipt["files"][relative.as_posix()] = {
            "before": sha(before) if before is not None else None,
            "after": sha(raw),
        }
        atomic_write(destination, raw)
    for name, digest in native.items():
        if sha((target / name).read_bytes()) != digest:
            raise RuntimeError("Native desktop files changed unexpectedly")
    (backup / "receipt.json").write_text(
        json.dumps(receipt, indent=2), encoding="utf-8"
    )
    (ROOT / "local/hermes-collapse-renderer-current.json").write_text(
        json.dumps(receipt), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "changed_renderer_files": len(receipt["files"]),
                "backup": str(backup),
                "native_unchanged": True,
                "hermes_backend_restarted": False,
            }
        )
    )


if __name__ == "__main__":
    main()
