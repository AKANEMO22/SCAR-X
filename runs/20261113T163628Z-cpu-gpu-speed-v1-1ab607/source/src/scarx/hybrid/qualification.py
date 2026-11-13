"""Predeclared engineering parity gate, distinct from P1/P2 quality gates."""

import json
import math
import hashlib
import time
from urllib.request import Request, urlopen
from .store import encode


def verify_native_storage(directory):
    report = json.loads(
        (directory / "native-integrity.json").read_text(encoding="utf-8")
    )
    if (
        report.get("status") != "NATIVE_STORAGE_INTEGRITY_PASS"
        or report.get("quantized") is not False
    ):
        raise ValueError("Native storage audit is missing or failed")
    expected = {
        "qwen-native-bf16.gguf",
        "qwen-vision-native-bf16.gguf",
        "whisper-base-f32.bin",
    }
    if set(report.get("exports", {})) != expected:
        raise ValueError("Unexpected native model manifest")
    for name, entry in report["exports"].items():
        path = directory / "models/native" / name
        if path.stat().st_size != entry["bytes"]:
            raise ValueError("Native model size changed: " + name)
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(2 << 20), b""):
                digest.update(block)
        if digest.hexdigest() != entry["sha256"]:
            raise ValueError("Native model checksum changed: " + name)
    return {
        "status": "PASS",
        "audit_run": report["run_id"],
        "model_files": 3,
        "no_quantization": True,
    }


def native_parity(backend, directory):
    path = directory / "original-reference.json"
    if not path.exists():
        raise ValueError("Original-checkpoint reference is missing")
    reference = json.loads(path.read_text(encoding="utf-8"))
    if reference.get("status") != "REFERENCE_READY":
        raise ValueError("Original reference did not pass source-precision checks")
    report = {
        "reference_run": reference["run_id"],
        "top1_required": True,
        "max_probability_difference": 0.03,
        "cases": [],
    }
    for case in reference["cases"]:
        body = {
            "prompt": case["input_ids"],
            "n_predict": 1,
            "n_probs": 8,
            "temperature": 1.0,
            "top_k": 0,
            "top_p": 1.0,
            "min_p": 0.0,
            "repeat_penalty": 1.0,
            "cache_prompt": False,
            "post_sampling_probs": False,
            "seed": 260113,
        }
        with urlopen(
            Request(
                backend.vlm_url + "/completion",
                data=encode(body).encode(),
                headers=backend.headers(),
            ),
            timeout=60,
        ) as response:
            native = json.loads(response.read(1_000_000))
        probabilities = native.get("completion_probabilities", [])
        if not probabilities:
            raise ValueError("Runtime did not expose numerical probability evidence")
        rows = (
            probabilities[0].get("top_logprobs")
            or probabilities[0].get("top_probs")
            or probabilities[0].get("probs")
        )
        if not rows:
            raise ValueError("Runtime did not expose top-token probabilities")
        values = {
            int(row.get("id", row.get("token_id", -1))): (
                math.exp(float(row["logprob"]))
                if "logprob" in row
                else float(row.get("prob", row.get("p", 0.0)))
            )
            for row in rows
        }
        top = max(values, key=values.get)
        difference = max(
            abs(values.get(token, 0.0) - expected)
            for token, expected in zip(case["top_ids"], case["top_probs"])
        )
        row = {
            "prompt": case["prompt"],
            "native_top1": top,
            "reference_top1": case["top_ids"][0],
            "max_probability_difference": difference,
            "passed": top == case["top_ids"][0] and difference <= 0.03,
            "native_probabilities": rows,
        }
        report["cases"].append(row)
        backend.store.event("native_parity_case", row, "orchestrator")
        if not row["passed"]:
            raise ValueError("Native/HF probability parity failed: " + encode(row))
    report["status"] = "PASS"
    return report
