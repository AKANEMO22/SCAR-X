"""Pinned official native FP32 ASR candidate; no serving config/model edits."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import traceback
import uuid
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from scarx.hybrid.store import Store, encode

REV = "973afd24965f72e36ca33b3055d56a652f456b4d"
SOURCE_SHA = "1d7734884874f1a1513ed9aa760a4f8e97aaa02fd6d93a3a85d27b2ae9ca596b"


def filehash(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(2 << 20):
            h.update(block)
    return h.hexdigest()


def main():
    name = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-whisper-small-native-v1-" + uuid.uuid4().hex[:6]
    run = ROOT / "runs" / name
    run.mkdir()
    candidate = ROOT / "local/hybrid-v1/candidates/whisper-small"
    candidate.mkdir(parents=True, exist_ok=True)
    trace = Store(ROOT / "local/hybrid-v1")
    report = {"run_id": name, "source_revision": REV, "candidate_dir": str(candidate),
              "failures": [], "live_ASR_changed": False}
    sources = [
        Path(__file__), ROOT / "scripts/hybrid_convert.py",
        ROOT / "experiments/hybrid-whisper-small-native-v1.md",
    ]
    for path in sources:
        dest = run / "source" / path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(path.read_bytes())
    (run / "manifest.json").write_text(encode({
        "source_sha256": {p.relative_to(ROOT).as_posix(): filehash(p) for p in sources},
        "model_source_sha256": SOURCE_SHA, "model_source_bytes": 966_995_080,
        "download_started_epoch": time.time(),
    }), encoding="utf-8")

    def fetch(url, path, expected_sha=None, size=None, limit=1_100_000_000):
        if path.exists():
            if (size is None or path.stat().st_size == size) and (
                    expected_sha is None or filehash(path) == expected_sha):
                return
            raise ValueError("Existing candidate asset mismatch: " + path.name)
        temp = path.with_suffix(path.suffix + ".partial")
        if temp.exists():
            raise FileExistsError("Refuse reusing incomplete download without audit")
        count, last = 0, 0.0
        h = hashlib.sha256()
        with urlopen(Request(url, headers={"User-Agent": "SCAR-X-native-ASR-research"}), timeout=60) as response, temp.open("xb") as out:
            while chunk := response.read(2 << 20):
                count += len(chunk)
                if count > limit:
                    raise ValueError("Candidate download exceeds budget")
                out.write(chunk)
                h.update(chunk)
                if time.monotonic() - last > 5:
                    trace.event("ASR_candidate.download_progress",
                        {"run_id": name, "asset": path.name, "bytes": count,
                         "expected_bytes": size, "live_ASR_changed": False}, "orchestrator")
                    last = time.monotonic()
        if size is not None and count != size or expected_sha and h.hexdigest() != expected_sha:
            raise ValueError("Pinned asset integrity failure")
        temp.replace(path)

    started = time.monotonic()
    try:
        for filename in ("config.json", "vocab.json", "README.md"):
            fetch(f"https://huggingface.co/openai/whisper-small/resolve/{REV}/{filename}",
                  candidate / filename, limit=2_000_000)
        fetch(f"https://huggingface.co/openai/whisper-small/resolve/{REV}/model.safetensors",
              candidate / "model.safetensors", SOURCE_SHA, 966_995_080)
        fetch("https://raw.githubusercontent.com/openai/whisper/v20250625/LICENSE",
              candidate / "UPSTREAM_LICENSE.txt", limit=100_000)
        report["source_model_sha256"] = filehash(candidate / "model.safetensors")
        filters = ROOT / "local/hybrid-v1/sources/whisper/assets/mel_filters.npz"
        spec = importlib.util.spec_from_file_location("audited_direct_ASR_converter", ROOT / "scripts/hybrid_convert.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        target = candidate / "whisper-small-f32.bin"
        if target.exists():
            raise FileExistsError("Refuse overwrite candidate export")
        tensors = module.convert_whisper(candidate, target, filters)
        if len({t["target"] for t in tensors}) != len(tensors):
            raise ValueError("Converted tensor name collision")
        report["tensors"] = tensors
        report["native_export"] = {"path": str(target), "bytes": target.stat().st_size,
                                  "sha256": filehash(target), "precision": "originalFP32",
                                  "every_tensor_exact_readback": True}
        report["mel_filters_sha256"] = filehash(filters)
        report["status"] = "NATIVE_FP32_STORAGE_READY_INFERENCE_PENDING"
        report["download_and_conversion_seconds"] = time.monotonic() - started
    except Exception:
        report["failures"].append(traceback.format_exc())
        report["status"] = "CANDIDATE_SETUP_FAILED"
    finally:
        with (run / "setup-result.json").open("x", encoding="utf-8") as f:
            f.write(encode(report))
        if not report["failures"]:
            (candidate / "native-storage-receipt.json").write_text(encode({
                "run_id": name, "status": report["status"], "source_revision": REV,
                "source_sha256": SOURCE_SHA, "export": report["native_export"],
                "tensor_count": len(report["tensors"]), "quality_qualified": False,
            }), encoding="utf-8")
            (ROOT / "local/current-whisper-small-run.txt").write_text(str(run))
        trace.event("ASR_candidate.setup_result",
            {k:v for k,v in report.items() if k != "tensors"}, "orchestrator")
        trace.close()
        print(encode({k:report.get(k) for k in (
            "run_id", "status", "failures", "native_export", "download_and_conversion_seconds")}), flush=True)
    return not report["failures"]


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
