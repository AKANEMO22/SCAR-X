"""Hash-lock the existing native exports, retain every failed audit receipt."""

import hashlib
import json
from pathlib import Path
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from scarx.hybrid.store import Store, encode


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(2 << 20), b""):
            value.update(block)
    return value.hexdigest()


if __name__ == "__main__":
    data = ROOT / "local/hybrid-v1"
    store = Store(data)
    run_id = (
        time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        + "-hybrid-native-audit-"
        + uuid.uuid4().hex[:6]
    )
    run = ROOT / "runs" / run_id
    run.mkdir(parents=True)
    report = {
        "run_id": run_id,
        "source": {},
        "exports": {},
        "quantized": False,
        "scope": "Storage integrity/native tensor types; full semantic quality not established",
    }
    expected = {
        "models/Qwen3.5-0.8B/model.safetensors-00001-of-00001.safetensors": "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696",
        "models/whisper-base/model.safetensors": "07cadb9f25677c8d50df603e66a98fbd842cce45047139baeb16e6219a1e807b",
    }
    try:
        for relative, sha in expected.items():
            actual = digest(data / relative)
            if actual != sha:
                raise ValueError("Original source hash mismatch: " + relative)
            report["source"][relative] = {
                "sha256": actual,
                "bytes": (data / relative).stat().st_size,
            }
        sys.path.insert(0, str(data / "sources/llama.cpp-b12360/gguf-py"))
        import gguf

        for name in ("qwen-native-bf16.gguf", "qwen-vision-native-bf16.gguf"):
            path = data / "models/native" / name
            reader = gguf.GGUFReader(str(path))
            types = {}
            for tensor in reader.tensors:
                kind = tensor.tensor_type.name
                types[kind] = types.get(kind, 0) + 1
                if kind not in ("BF16", "F32"):
                    raise ValueError("Reduced/unsupported precision in native export")
            report["exports"][name] = {
                "bytes": path.stat().st_size,
                "sha256": digest(path),
                "tensor_types": types,
                "tensors": len(reader.tensors),
            }
            del reader
        name = "whisper-base-f32.bin"
        path = data / "models/native" / name
        conversion = json.loads(
            (data / "native-conversion.json").read_text(encoding="utf-8")
        )
        report["exports"][name] = {
            "bytes": path.stat().st_size,
            "sha256": digest(path),
            "dtype": "F32",
            "direct_exact_tensor_readback_count": len(conversion["whisper"]["tensors"]),
        }
        report["status"] = "NATIVE_STORAGE_INTEGRITY_PASS"
        (data / "native-integrity.json").write_text(encode(report), encoding="utf-8")
    except Exception as exc:
        report["status"] = "FAIL"
        report["error"] = str(exc)
    (run / "receipt.json").write_text(encode(report), encoding="utf-8")
    store.event("native_integrity_audit", report, "orchestrator")
    store.close()
    print(json.dumps(report, ensure_ascii=True))
