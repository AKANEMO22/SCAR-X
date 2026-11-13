"""Native repacking + CPU source reference of the isolated official2B candidate."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import time

for key in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[key] = "1"
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "local/hybrid-v1/candidates/qwen3.5-2b"


def reference(run):
    import torch
    import numpy as np
    from transformers import AutoModelForImageTextToText, AutoProcessor

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    started = time.monotonic()
    report = {
        "run_id": run.name,
        "cases": [],
        "dtype_restorations": [],
        "unused_tensors": [],
        "mismatches": [],
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
        with (DATA / "source/model.safetensors-00001-of-00001.safetensors").open(
            "rb"
        ) as stream:
            length = struct.unpack("<Q", stream.read(8))[0]
            header = json.loads(stream.read(length))
            base = 8 + length
            for name, row in header.items():
                if name == "__metadata__":
                    continue
                if name not in state:
                    report["unused_tensors"].append(name)
                    continue
                parameter = state[name]
                expected = {"BF16": torch.bfloat16, "F32": torch.float32}[row["dtype"]]
                begin, end = row["data_offsets"]
                stream.seek(base + begin)
                if expected == torch.float32 and parameter.dtype != expected:
                    native = (
                        np.frombuffer(stream.read(end - begin), dtype="<f4")
                        .copy()
                        .reshape(row["shape"])
                    )
                    model.get_parameter(name).data = torch.from_numpy(native)
                    state = model.state_dict()
                    parameter = state[name]
                    report["dtype_restorations"].append(name)
                stream.seek(base + begin)
                h = hashlib.sha256()
                remaining = end - begin
                while remaining:
                    block = stream.read(min(1 << 20, remaining))
                    h.update(block)
                    remaining -= len(block)
                raw = parameter.detach().contiguous().view(torch.uint8).numpy()
                if (
                    parameter.dtype != expected
                    or list(parameter.shape) != row["shape"]
                    or h.digest() != hashlib.sha256(memoryview(raw).cast("B")).digest()
                ):
                    report["mismatches"].append(name)
        if report["mismatches"]:
            raise ValueError("Original candidate reference changed weights/dtype")
        for prompt in [
            "Say hello in one short sentence.",
            "Chào bạn, trả lời ngắn gọn bằng tiếng Việt.",
            "What is 7 plus 8?",
        ]:
            messages = [{"role": "user", "content": prompt}]
            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = processor(text=[text], return_tensors="pt")
            t = time.monotonic()
            with torch.inference_mode():
                p = torch.softmax(model(**inputs).logits[0, -1].float(), dim=-1)
                values, ids = torch.topk(p, 8)
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
    report["seconds"] = time.monotonic() - started
    (run / "reference.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in ["cases", "unused_tensors", "dtype_restorations"]
            }
        ),
        flush=True,
    )


def convert(run):
    output = DATA / "native"
    output.mkdir(exist_ok=True)
    converter = ROOT / "local/hybrid-v1/sources/llama.cpp-b12360/convert_hf_to_gguf.py"
    for extra, name in [
        ([], "qwen-native-bf16.gguf"),
        (["--mmproj"], "qwen-vision-native-bf16.gguf"),
    ]:
        dest = output / name
        if dest.exists():
            raise ValueError("Refuse overwrite candidate export")
        with (run / (name + ".conversion.log")).open("w", encoding="utf-8") as log:
            subprocess.run(
                [
                    sys.executable,
                    str(converter),
                    str(DATA / "source"),
                    "--outtype",
                    "bf16",
                    "--outfile",
                    str(dest),
                    *extra,
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
    sys.path.insert(0, str(converter.parent / "gguf-py"))
    import gguf

    entries = {}
    for dest in output.glob("*.gguf"):
        reader = gguf.GGUFReader(str(dest))
        tensors = [
            {
                "name": t.name,
                "type": t.tensor_type.name,
                "shape": list(map(int, t.shape)),
            }
            for t in reader.tensors
        ]
        if any(t["type"] not in ("BF16", "F32", "F64") for t in tensors):
            raise ValueError("Reduced source precision in native candidate")
        with dest.open("rb") as stream:
            h = hashlib.file_digest(stream, "sha256").hexdigest()
        entries[dest.name] = {
            "bytes": dest.stat().st_size,
            "sha256": h,
            "tensors": tensors,
        }
        del reader
    (run / "conversion.json").write_text(
        json.dumps(
            {
                "source": "pinned original BF16/F32 checkpoint",
                "live_unchanged": True,
                "quantization": False,
                "exports": entries,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "conversion_done": True,
                "exports": {
                    k: {"bytes": v["bytes"], "sha256": v["sha256"]}
                    for k, v in entries.items()
                },
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["convert", "reference"])
    p.add_argument("run", type=Path)
    a = p.parse_args()
    run = a.run.resolve()
    if ROOT / "runs" not in run.parents:
        raise ValueError("Run outside workspace")
    if not run.exists():
        run.mkdir()
    if (run / (a.mode + ".started")).exists():
        raise ValueError("Refuse rerun")
    (run / (a.mode + ".started")).write_text(str(time.time()))
    if a.mode == "convert":
        convert(run)
    else:
        reference(run)
