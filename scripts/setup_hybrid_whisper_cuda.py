"""Pinned official native FP32-capable Whisper CUDA worker; CPU install retained."""

import hashlib
import json
from pathlib import Path
import shutil
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
NAME = "whisper-bin-win-cuda-12.4.0-x64.zip"
SHA = "afef0b881c500958921c3f5523b50e59ee2ec9b6f5cbd25b324c51ed308a957a"
URL = "https://github.com/ggml-org/whisper.cpp/releases/download/b5454/" + NAME


def main():
    base = ROOT / "local/hybrid-v1"
    target = base / "runtime/whisper.cpp-b5454-cuda12.4"
    target.mkdir(parents=True, exist_ok=True)
    path = base / "downloads" / NAME
    if not path.exists():
        partial = path.with_suffix(".partial")
        with (
            urllib.request.urlopen(URL, timeout=60) as response,
            partial.open("wb") as stream,
        ):
            shutil.copyfileobj(response, stream, 1024 * 1024)
        partial.replace(path)
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != SHA:
        raise ValueError("Official CUDA Whisper asset integrity mismatch")
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate archive members")
        for entry in archive.infolist():
            dest = (target / entry.filename).resolve()
            if (
                not dest.is_relative_to(target.resolve())
                or ((entry.external_attr >> 16) & 0o170000) == 0o120000
            ):
                raise ValueError("Unsafe native-worker archive path")
            if entry.is_dir():
                dest.mkdir(parents=True, exist_ok=True)
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            raw = archive.read(entry)
            if dest.exists() and dest.read_bytes() != raw:
                raise ValueError("Refuse overwrite different runtime binary")
            if not dest.exists():
                dest.write_bytes(raw)
    result = {
        "url": URL,
        "published_sha256": SHA,
        "sha256": actual,
        "archive_bytes": path.stat().st_size,
        "original_cpu_preserved": True,
    }
    (target / "installation-receipt.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
