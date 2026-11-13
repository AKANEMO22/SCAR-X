"""Pinned, hash-checked native assets. Does not quantize or change live caches."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
from urllib.request import Request, urlopen
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from scarx.hybrid.config import HybridConfig
from scarx.hybrid.store import Store, encode

DATA = ROOT / "local/hybrid-v1"


def remote_json(url):
    with urlopen(
        Request(url, headers={"User-Agent": "SCAR-X-Hybrid/1.0"}), timeout=45
    ) as response:
        return json.load(response)


def download(url, path, store, sha=None, size=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def digest(file):
        h = hashlib.sha256()
        with file.open("rb") as handle:
            for block in iter(lambda: handle.read(4 << 20), b""):
                h.update(block)
        return h.hexdigest()

    if (
        path.exists()
        and (size is None or path.stat().st_size == size)
        and (not sha or digest(path) == sha)
    ):
        store.event(
            "asset_reused", {"file": str(path), "sha256": digest(path)}, "orchestrator"
        )
        return path
    temporary = path.with_suffix(path.suffix + ".partial")
    store.event(
        "download_started",
        {"url": url, "file": str(path), "expected_bytes": size},
        "orchestrator",
    )
    h = hashlib.sha256()
    count = 0
    last = time.monotonic()
    with (
        urlopen(
            Request(url, headers={"User-Agent": "SCAR-X-Hybrid/1.0"}), timeout=60
        ) as response,
        temporary.open("wb") as handle,
    ):
        while block := response.read(2 << 20):
            handle.write(block)
            h.update(block)
            count += len(block)
            if time.monotonic() - last > 3:
                store.event(
                    "download_progress",
                    {"file": path.name, "bytes": count, "expected_bytes": size},
                    "orchestrator",
                )
                last = time.monotonic()
    if (sha and h.hexdigest() != sha) or (size and count != size):
        raise ValueError("Downloaded asset failed integrity check: " + path.name)
    temporary.replace(path)
    store.event(
        "download_verified",
        {"file": str(path), "bytes": count, "sha256": h.hexdigest()},
        "orchestrator",
    )
    return path


def extract(archive, destination):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        for item in bundle.infolist():
            target = (destination / item.filename).resolve()
            if (
                not target.is_relative_to(destination)
                or (item.external_attr >> 16) & 0o170000 == 0o120000
            ):
                raise ValueError("Unsafe archive member")
        bundle.extractall(destination)


def release_asset(repo, tag, name, store):
    release = remote_json(f"https://api.github.com/repos/{repo}/releases/tags/{tag}")
    asset = next(a for a in release["assets"] if a["name"] == name)
    sha = asset.get("digest", "").removeprefix("sha256:")
    if len(sha) != 64:
        raise ValueError("Release asset has no SHA256")
    archive = download(
        asset["browser_download_url"],
        DATA / "downloads" / name,
        store,
        sha,
        asset["size"],
    )
    target = DATA / "runtime" / (repo.split("/")[-1] + "-" + tag)
    extract(archive, target)
    return dict(repo=repo, tag=tag, sha256=sha, path=str(target))


def hf_model(repo, revision, store):
    listing = remote_json(f"https://huggingface.co/api/models/{repo}/tree/{revision}")
    target = DATA / "models" / repo.split("/")[-1]
    files = []
    for item in listing:
        name = item["path"]
        if item.get("type") != "file" or not name.endswith(
            (".safetensors", ".json", ".jinja", ".txt")
        ):
            continue
        if name in ("README.md",) or name.endswith(".bin"):
            continue
        lfs = item.get("lfs") or {}
        file = download(
            f"https://huggingface.co/{repo}/resolve/{revision}/{name}",
            target / name,
            store,
            lfs.get("oid"),
            item["size"],
        )
        files.append(
            dict(name=name, bytes=file.stat().st_size, source_sha256=lfs.get("oid"))
        )
    return dict(repo=repo, revision=revision, path=str(target), files=files)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-only", action="store_true")
    args = parser.parse_args()
    store = Store(DATA)
    config = HybridConfig()
    manifest = {
        "format": "scarx-hybrid-assets-v1",
        "precision": "original checkpoint BF16/F32; Whisper FP32",
        "quantization": False,
        "pruning": False,
        "runtime": [],
        "models": [],
    }
    try:
        store.put(
            "runtime",
            "status",
            {
                "phase": "setup",
                "note": "Downloading original-precision assets; no quality claim",
            },
        )
        manifest["runtime"].append(
            release_asset(
                "ggml-org/llama.cpp",
                "b12360",
                "llama-b12360-bin-win-cpu-x64.zip",
                store,
            )
        )
        manifest["runtime"].append(
            release_asset("ggml-org/whisper.cpp", "b5454", "whisper-bin-x64.zip", store)
        )
        # Converter source is pinned to the same release as the executable.
        archive = download(
            "https://codeload.github.com/ggml-org/llama.cpp/zip/refs/tags/b12360",
            DATA / "downloads/llama-source-b12360.zip",
            store,
        )
        extract(archive, DATA / "sources")
        if not args.runtime_only:
            manifest["models"].append(
                hf_model("Qwen/Qwen3.5-0.8B", config.model_revision, store)
            )
            manifest["models"].append(
                hf_model("openai/whisper-base", config.asr_revision, store)
            )
            # Mel filter bank is a preprocessing asset, not a pretrained model.
            download(
                "https://raw.githubusercontent.com/openai/whisper/v20250625/whisper/assets/mel_filters.npz",
                DATA / "sources/whisper/assets/mel_filters.npz",
                store,
            )
        (DATA / "assets.json").write_text(encode(manifest), encoding="utf-8")
        store.event("setup_downloads_complete", manifest, "orchestrator")
        print(encode(manifest), flush=True)
    except Exception as exc:
        store.event("setup_failed", {"error": str(exc)}, "orchestrator")
        raise
    finally:
        store.close()


if __name__ == "__main__":
    main()
