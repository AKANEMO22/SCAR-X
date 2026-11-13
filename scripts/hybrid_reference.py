"""Offline original-checkpoint reference; never imported by the product runtime."""

import os

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import hashlib
import json
from pathlib import Path
import sys
import time
import uuid
import struct
import torch
import numpy as np
from safetensors import safe_open
from transformers import AutoModelForImageTextToText, AutoProcessor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from scarx.hybrid.store import Store, encode
from scarx.hybrid.resources import ResourceMonitor, windows_tree_commit_limit


def main():
    data = ROOT / "local/hybrid-v1"
    source = data / "models/Qwen3.5-0.8B"
    store = Store(data)
    run_id = (
        time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        + "-hybrid-reference-"
        + uuid.uuid4().hex[:6]
    )
    run = ROOT / "runs" / run_id
    run.mkdir(parents=True)
    previous = data / "original-reference.json"
    if previous.exists():
        old = previous.read_bytes()
        old_data = json.loads(old)
        if "run_id" not in old_data:
            archive = (
                ROOT
                / "runs"
                / ("hybrid-reference-legacy-" + hashlib.sha256(old).hexdigest()[:12])
            )
            archive.mkdir(exist_ok=True)
            if not (archive / "reference.json").exists():
                (archive / "reference.json").write_bytes(old)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    monitor = ResourceMonitor().start()
    job = windows_tree_commit_limit()
    report = {
        "format": "hybrid-original-reference-v1",
        "run_id": run_id,
        "job": job,
        "dtype_mismatches": [],
        "native_dtype_restorations": [],
        "parameter_mismatches": [],
        "official_forward_unused_source_tensors": [],
        "cases": [],
    }
    try:
        store.event(
            "original_reference_started",
            {
                "backend": "Transformers offline reference; not production runtime",
                "budget": 4_000_000_000,
            },
            "orchestrator",
        )
        model = AutoModelForImageTextToText.from_pretrained(
            source, local_files_only=True, dtype="auto", attn_implementation="eager"
        )
        model.eval()
        processor = AutoProcessor.from_pretrained(source, local_files_only=True)
        state = model.state_dict()
        for path in source.glob("*.safetensors"):
            # A second safetensors COW mapping reserves the whole 1.75GB again
            # on Windows. Compare bytes incrementally instead of mapping twice.
            with path.open("rb") as checkpoint:
                header_bytes = struct.unpack("<Q", checkpoint.read(8))[0]
                header = json.loads(checkpoint.read(header_bytes))
                data_start = 8 + header_bytes
                for name, tensor in header.items():
                    if name == "__metadata__":
                        continue
                    if name not in state:
                        report["official_forward_unused_source_tensors"].append(name)
                        continue
                    actual = state[name]
                    expected_dtype = {"BF16": torch.bfloat16, "F32": torch.float32}[
                        tensor["dtype"]
                    ]
                    if (
                        expected_dtype == torch.float32
                        and actual.dtype != expected_dtype
                    ):
                        # Transformers dtype='auto' casts even checkpoint-F32
                        # A_log/norm tensors down to BF16. Restore the exact
                        # native values from the authenticated source file.
                        begin, end = tensor["data_offsets"]
                        checkpoint.seek(data_start + begin)
                        native = (
                            np.frombuffer(checkpoint.read(end - begin), dtype="<f4")
                            .copy()
                            .reshape(tensor["shape"])
                        )
                        parameter = model.get_parameter(name)
                        parameter.data = torch.from_numpy(native)
                        report["native_dtype_restorations"].append(
                            {
                                "name": name,
                                "from": str(actual.dtype),
                                "to": "torch.float32",
                            }
                        )
                        state = model.state_dict()
                        actual = state[name]
                    if expected_dtype != actual.dtype:
                        report["dtype_mismatches"].append(name)
                    begin, end = tensor["data_offsets"]
                    checkpoint.seek(data_start + begin)
                    digest = hashlib.sha256()
                    remaining = end - begin
                    while remaining:
                        block = checkpoint.read(min(remaining, 1 << 20))
                        digest.update(block)
                        remaining -= len(block)
                    raw = actual.detach().contiguous().view(torch.uint8).numpy()
                    if (
                        list(actual.shape) != tensor["shape"]
                        or digest.digest()
                        != hashlib.sha256(memoryview(raw).cast("B")).digest()
                    ):
                        report["parameter_mismatches"].append(name)
        if report["dtype_mismatches"] or report["parameter_mismatches"]:
            raise ValueError(
                "Original reference loading changed source precision/weights"
            )
        for prompt in (
            "Say hello in one short sentence.",
            "Chào bạn, trả lời ngắn gọn bằng tiếng Việt.",
            "What is 7 plus 8?",
        ):
            messages = [{"role": "user", "content": prompt}]
            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = processor(text=[text], return_tensors="pt")
            started = time.monotonic()
            with torch.inference_mode():
                logits = model(**inputs).logits[0, -1].float()
                p = torch.softmax(logits, dim=-1)
                values, indices = torch.topk(p, 8)
            monitor.check()
            case = {
                "prompt": prompt,
                "input_ids": inputs["input_ids"][0].tolist(),
                "top_ids": indices.tolist(),
                "top_probs": values.tolist(),
                "top1_text": processor.tokenizer.decode([int(indices[0])]),
                "seconds": time.monotonic() - started,
            }
            report["cases"].append(case)
            store.event("original_reference_case", case, "orchestrator")
        report["status"] = "REFERENCE_READY"
    except Exception as exc:
        report["status"] = "REFERENCE_FAILED"
        report["error"] = str(exc)
        store.event("original_reference_failed", {"error": str(exc)}, "orchestrator")
    finally:
        report["resources"] = monitor.close()
        (run / "reference.json").write_text(encode(report), encoding="utf-8")
        (data / "original-reference.json").write_text(encode(report), encoding="utf-8")
        store.event(
            "original_reference_complete",
            {
                "status": report["status"],
                "resources": report["resources"],
                "dtype_mismatches": len(report["dtype_mismatches"]),
                "parameter_mismatches": len(report["parameter_mismatches"]),
            },
            "orchestrator",
        )
        store.close()
        print(encode({k: v for k, v in report.items() if k != "cases"}), flush=True)


if __name__ == "__main__":
    main()
