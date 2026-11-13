"""Original-BF16 optional VLM repacking and independent source-forward reference."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import time
import uuid

for var in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[var] = "1"
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "local/hybrid-v1/candidates/lfm2.5-vl-1.6b"
IDENTITY = {
    "name": "LiquidAI/LFM2.5-VL-1.6B",
    "revision": "919fde3d022e3f90a4716006f993938ee8c2eb97",
}


def prepare():
    name = (
        time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        + "-native-lfm16-v1-"
        + uuid.uuid4().hex[:6]
    )
    run = ROOT / "runs" / name
    run.mkdir()
    hashes = {}
    for path in [
        Path(__file__),
        ROOT / "scripts/setup_hybrid_lfm16.py",
        ROOT / "experiments/hybrid-native-lfm16-vram-v1.md",
    ]:
        rel = path.relative_to(ROOT)
        out = run / "source" / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(path.read_bytes())
        hashes[rel.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "prepared": time.time(),
                "identity": IDENTITY,
                "source_sha256": hashes,
                "live_changed": False,
                "no_weight_compression": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (ROOT / "local/current-lfm16-run.txt").write_text(str(run))
    print(run)


def convert(run):
    out = DATA / "native"
    out.mkdir(exist_ok=True)
    converter = ROOT / "local/hybrid-v1/sources/llama.cpp-b12360/convert_hf_to_gguf.py"
    exports = {}
    for flag, name in [
        ([], "model-native-bf16.gguf"),
        (["--mmproj"], "vision-native-bf16.gguf"),
    ]:
        target = out / name
        if target.exists():
            raise ValueError("Refuse overwrite native candidate")
        with (run / (name + ".log")).open("w", encoding="utf-8") as log:
            subprocess.run(
                [
                    sys.executable,
                    str(converter),
                    str(DATA / "source"),
                    "--outtype",
                    "bf16",
                    "--outfile",
                    str(target),
                    *flag,
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
    sys.path.insert(0, str(converter.parent / "gguf-py"))
    import gguf

    for p in out.glob("*.gguf"):
        reader = gguf.GGUFReader(str(p))
        tensors = [
            {
                "name": t.name,
                "type": t.tensor_type.name,
                "shape": list(map(int, t.shape)),
            }
            for t in reader.tensors
        ]
        del reader
        if any(r["type"] not in ("BF16", "F32", "F64") for r in tensors):
            raise ValueError("Unexpected reduced-precision export")
        with p.open("rb") as stream:
            sha = hashlib.file_digest(stream, "sha256").hexdigest()
        exports[p.name] = {"bytes": p.stat().st_size, "sha256": sha, "tensors": tensors}
    shutil.copyfile(DATA / "source/LICENSE", out / "LICENSE-LFM-Open-1.0.txt")
    (run / "conversion.json").write_text(
        json.dumps(
            {
                "identity": IDENTITY,
                "quantized": False,
                "exports": exports,
                "note": "BF16 checkpoint repacked; FP32 norm/metadata promotion allowed, no lower precision",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"converted": True, "exports": {k: v["bytes"] for k, v in exports.items()}}
        ),
        flush=True,
    )


def reference(run, keymap_v2=False):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    start = time.monotonic()
    report = {
        "run_id": run.name,
        "identity": IDENTITY,
        "cases": [],
        "mismatches": [],
        "unused_tensors": [],
        "loader_key_renames": [],
        "key_mapping_version": 2 if keymap_v2 else 1,
    }
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            DATA / "source",
            local_files_only=True,
            dtype="auto",
            attn_implementation="eager",
        )
        model.eval()
        processor = AutoProcessor.from_pretrained(
            DATA / "source", local_files_only=True
        )
        state = model.state_dict()
        with (DATA / "source/model.safetensors").open("rb") as f:
            length = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(length))
            base = 8 + length
            for name, tensor in header.items():
                if name == "__metadata__":
                    continue
                target = name
                if keymap_v2 and target.startswith("model.vision_tower.vision_model."):
                    target = target.replace(
                        "model.vision_tower.vision_model.", "model.vision_tower.", 1
                    )
                    report["loader_key_renames"].append(
                        {"source": name, "loaded": target}
                    )
                if target not in state:
                    report["unused_tensors"].append(name)
                    continue
                value = state[target]
                if (
                    value.dtype != torch.bfloat16
                    or list(value.shape) != tensor["shape"]
                ):
                    report["mismatches"].append(name)
                    continue
                a, b = tensor["data_offsets"]
                f.seek(base + a)
                h = hashlib.sha256()
                remaining = b - a
                while remaining:
                    block = f.read(min(remaining, 1 << 20))
                    h.update(block)
                    remaining -= len(block)
                raw = value.detach().contiguous().view(torch.uint8).numpy()
                if h.digest() != hashlib.sha256(memoryview(raw).cast("B")).digest():
                    report["mismatches"].append(name)
        if report["mismatches"] or report["unused_tensors"]:
            raise ValueError("Original reference dtype/coverage/weight mismatch")
        for prompt in [
            "Say hello in one short sentence.",
            "Chào bạn, trả lời ngắn gọn bằng tiếng Việt.",
            "What is 7 plus 8?",
        ]:
            text = processor.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = processor.tokenizer(text, return_tensors="pt")
            t = time.monotonic()
            with torch.inference_mode():
                prob = torch.softmax(model(**inputs).logits[0, -1].float(), dim=-1)
                values, ids = torch.topk(prob, 8)
            report["cases"].append(
                {
                    "prompt": prompt,
                    "input_ids": inputs["input_ids"][0].tolist(),
                    "top_ids": ids.tolist(),
                    "top_probs": values.tolist(),
                    "seconds": time.monotonic() - t,
                }
            )
        report["status"] = "REFERENCE_READY"
    except Exception as exc:
        report.update(
            status="REFERENCE_FAILED", error=type(exc).__name__ + ": " + str(exc)
        )
    report["seconds"] = time.monotonic() - start
    (run / ("reference-keymap-v2.json" if keymap_v2 else "reference.json")).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in ("cases", "mismatches", "unused_tensors")
            },
            ensure_ascii=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "mode", choices=["prepare", "convert", "reference", "reference-keymap-v2"]
    )
    p.add_argument("--run", type=Path)
    a = p.parse_args()
    if a.mode == "prepare":
        prepare()
    else:
        run = a.run.resolve()
        if ROOT / "runs" not in run.parents:
            raise ValueError("Run outside workspace")
        marker = run / (a.mode + ".started")
        if marker.exists():
            raise ValueError("Refuse rerun")
        marker.write_text(str(time.time()))
        if a.mode == "convert":
            convert(run)
        else:
            reference(run, a.mode == "reference-keymap-v2")
