"""Pinned, read-only source/data intake; never install or execute upstream code."""

import ast
from collections import Counter
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import stat
import time
import uuid
from urllib.request import Request, urlopen
import zipfile

ROOT = Path(__file__).resolve().parents[1]
REVISION = "e05f4ce6132c80ed33392b81535b077d56ab28fd"
URL = "https://codeload.github.com/microsoft/ToolTalk/zip/" + REVISION


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    run = ROOT / "runs" / (
        time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        + "-tooltalk-intake-v1-" + uuid.uuid4().hex[:6])
    run.mkdir()
    target = ROOT / "local/upstream" / ("ToolTalk-" + REVISION)
    if target.exists():
        raise FileExistsError("Refuse silently replacing an existing upstream snapshot")
    headers = {"User-Agent": "SCAR-X-readonly-research-intake"}
    with urlopen(Request(URL, headers=headers), timeout=30) as response:
        raw = response.read(20_000_001)
    if len(raw) > 20_000_000:
        raise ValueError("Archive download exceeds budget")
    members = {}
    total = 0
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        prefix = archive.namelist()[0].split("/")[0]
        for item in archive.infolist():
            if not item.filename.startswith(prefix + "/"):
                raise ValueError("External archive root")
            relative = PurePosixPath(item.filename[len(prefix) + 1:])
            if (relative.is_absolute() or ".." in relative.parts
                    or "\\" in item.filename
                    or stat.S_ISLNK(item.external_attr >> 16)):
                raise ValueError("Unsafe archive member")
            if item.is_dir():
                continue
            if str(relative) in members:
                raise ValueError("Duplicate archive member")
            total += item.file_size
            if total > 50_000_000:
                raise ValueError("Archive expansion exceeds budget")
            members[str(relative)] = archive.read(item)
    for required in ("LICENSE", "NOTICE", "README.md"):
        if required not in members:
            raise ValueError("Missing original attribution: " + required)
    target.mkdir(parents=True)
    paths, imports, records, duplicate = {}, {}, [], {}
    for rel, content in members.items():
        path = target / rel
        if target.resolve() not in path.resolve().parents:
            raise ValueError("Resolved archive member escaped target")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        paths[rel] = digest(content)
        if rel.endswith(".py"):
            tree = ast.parse(content.decode("utf-8-sig"), filename=rel)
            imports[rel] = sorted({
                (node.module or "") if isinstance(node, ast.ImportFrom)
                else alias.name
                for node in ast.walk(tree)
                if isinstance(node, (ast.Import, ast.ImportFrom))
                for alias in node.names
            })
        if rel.startswith(("data/easy/", "data/tooltalk/")) and rel.endswith(".json"):
            value = json.loads(content)
            canonical = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
            d = digest(canonical)
            duplicate.setdefault(d, []).append(rel)
            records.append({
                "path": rel, "sha256": paths[rel], "type": type(value).__name__,
                "keys": sorted(value) if isinstance(value, dict) else None,
                "list_length": len(value) if isinstance(value, list) else None,
            })
    for filename in ("LICENSE", "NOTICE"):
        (run / filename).write_bytes(members[filename])
    report = {
        "repository": "https://github.com/microsoft/ToolTalk",
        "revision": REVISION, "archive_url": URL, "archive_sha256": digest(raw),
        "uncompressed_bytes": total, "local_snapshot": str(target),
        "file_sha256": paths, "python_AST_imports": imports,
        "conversation_files": records,
        "counts": dict(Counter(r["path"].split("/")[1] for r in records)),
        "exact_canonical_duplicate_conversations": [p for p in duplicate.values() if len(p) > 1],
        "executed_upstream_code": False, "installed_dependencies": False,
        "model_inference_or_training": False,
        "scope": "source/schema intake only; external corpus not yet a qualified SCAR-X benchmark",
    }
    (target.parent / ("ToolTalk-" + REVISION + ".zip")).write_bytes(raw)
    (run / "manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    sys_source = Path(__file__).read_bytes()
    (run / "intake-script.py").write_bytes(sys_source)
    (run / "protocol.md").write_bytes(
        (ROOT / "experiments/hybrid-external-tool-benchmark-intake-v1.md").read_bytes())
    print(json.dumps({"run_id": run.name, "counts": report["counts"],
                     "archive_bytes": len(raw), "uncompressed_bytes": total}))


if __name__ == "__main__":
    main()
