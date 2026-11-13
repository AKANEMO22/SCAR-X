"""Fetch exact official llama.cpp CUDA assets, verify published SHA256, keep CPU."""

import hashlib, json, os
from pathlib import Path
import shutil
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ASSETS = {
    "llama-b12360-bin-win-cuda-12.4-x64.zip": "70c0a244acf39f2f3fd8975a8447884e03fd35f499bf7dc284113845e894c382",
    "cudart-llama-bin-win-cuda-12.4-x64.zip": "8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6",
}


def main():
    data = ROOT / "local/hybrid-v1"
    downloads = data / "downloads"
    target = data / "runtime/llama.cpp-b12360-cuda12.4"
    downloads.mkdir(exist_ok=True)
    target.mkdir(exist_ok=True)
    receipts = []
    for name, expected in ASSETS.items():
        path = downloads / name
        if not path.exists():
            print(json.dumps({"downloading": name}), flush=True)
            with (
                urllib.request.urlopen(
                    "https://github.com/ggml-org/llama.cpp/releases/download/b12360/"
                    + name,
                    timeout=45,
                ) as r,
                path.open("xb") as f,
            ):
                shutil.copyfileobj(r, f, 1024 * 1024)
        with path.open("rb") as f:
            actual = hashlib.file_digest(f, "sha256").hexdigest()
        if actual != expected:
            raise ValueError("Official asset hash mismatch: " + name)
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            if len(names) != len(set(names)):
                raise ValueError("Duplicate zip names")
            for info in z.infolist():
                dest = (target / info.filename).resolve()
                if (
                    not dest.is_relative_to(target.resolve())
                    or ((info.external_attr >> 16) & 0o170000) == 0o120000
                ):
                    raise ValueError("Unsafe archive path")
                if info.is_dir():
                    dest.mkdir(parents=True, exist_ok=True)
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                content = z.read(info)
                if dest.exists() and dest.read_bytes() != content:
                    raise ValueError(
                        "Refuse overwrite different runtime file: " + info.filename
                    )
                if not dest.exists():
                    dest.write_bytes(content)
        receipts.append(
            {
                "name": name,
                "sha256": actual,
                "published_sha256": expected,
                "bytes": path.stat().st_size,
            }
        )
        print(json.dumps({"verified": name}), flush=True)
    (target / "installation-receipt.json").write_text(
        json.dumps(
            {
                "source": "https://github.com/ggml-org/llama.cpp/releases/tag/b12360",
                "assets": receipts,
                "original_CPU_runtime_preserved": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
