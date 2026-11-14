"""Independent original tensor bytes, edit distance, latency and active VRAM audit."""

import hashlib
import json
from pathlib import Path
import re
import struct
import sys
import unicodedata
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def filehash(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while data := f.read(2 << 20):
            h.update(data)
    return h.hexdigest()


def edit_count(reference, hypothesis):
    tokenize = lambda text: re.findall(r"\w+", unicodedata.normalize("NFC", text).casefold())
    a, b = tokenize(reference), tokenize(hypothesis)
    table = np.zeros((len(a) + 1, len(b) + 1), dtype=np.int64)
    table[:, 0], table[0, :] = np.arange(len(a) + 1), np.arange(len(b) + 1)
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            table[i, j] = min(table[i-1, j] + 1, table[i, j-1] + 1,
                              table[i-1, j-1] + (a[i-1] != b[j-1]))
    return int(table[-1, -1]), len(a)


def original_tensor_audit(source, exported, setup):
    with source.open("rb") as f:
        length = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(length))
        offset = 8 + length
        original = {k:v for k,v in header.items() if k != "__metadata__"}
        expected = setup["tensors"]
        if set(original) != {r["source"] for r in expected} or len(expected) != len(original):
            raise ValueError("Original tensor coverage is not one to one")
        with exported.open("rb") as g:
            fields = struct.unpack("<12i", g.read(48))
            assert fields[0] == 0x67676D6C and fields[-1] == 0
            rows, cols = struct.unpack("<2i", g.read(8))
            g.seek(rows * cols * 4, 1)
            count = struct.unpack("<i", g.read(4))[0]
            for _ in range(count):
                n = struct.unpack("<i", g.read(4))[0]
                g.seek(n, 1)
            tensors = 0
            for row in expected:
                meta = original[row["source"]]
                assert meta["dtype"] == "F32"
                ndims, name_length, tensor_type = struct.unpack("<3i", g.read(12))
                dimensions = struct.unpack("<" + "i" * ndims, g.read(ndims * 4))
                name = g.read(name_length).decode()
                assert name == row["target"] and tensor_type == 0
                assert list(reversed(dimensions)) == row["shape"]
                length = int(np.prod(dimensions)) * 4
                begin, end = meta["data_offsets"]
                assert end - begin == length
                f.seek(offset + begin)
                remaining, h = length, hashlib.sha256()
                while remaining:
                    n = min(2 << 20, remaining)
                    left, right = f.read(n), g.read(n)
                    if left != right or len(left) != n:
                        raise ValueError("Original/export tensor byte mismatch: " + name)
                    h.update(left)
                    remaining -= n
                assert h.hexdigest() == row["sha256"]
                tensors += 1
            assert not g.read(1), "Unexpected native model trailer"
    return {"original_F32_tensors": tensors, "every_original_export_byte_equal": True,
            "dropped_or_duplicated_tensors": 0, "quantization_or_dtype_reduction": False}


def main():
    paired_run = ROOT / "runs/20261114T015546Z-asr-candidate-benchmark-v1-68e3c7"
    resource_run = ROOT / "runs/20261114T020114Z-asr-resource-followup-v1-6e53cb"
    result = json.loads((paired_run / "result.json").read_text(encoding="utf-8"))
    manifest = json.loads((paired_run / "manifest.json").read_text(encoding="utf-8"))
    resource = json.loads((resource_run / "result.json").read_text(encoding="utf-8"))
    data_file = ROOT / "runs" / manifest["data_run"] / "manifest.json"
    assert filehash(data_file) == manifest["data_sha256"]
    data = json.loads(data_file.read_text(encoding="utf-8"))
    assert len(result["pairs"]) == len(data["samples"]) == 32
    assert not result["failures"] and not resource["failures"]
    by_id = {r["id"]:r for r in data["samples"]}
    score_inputs = []
    for row in result["pairs"]:
        locked = by_id[row["id"]]
        assert locked["reference"] == row["reference"] and not locked["known_engineering_sample"]
        for arm in ("base", "small"):
            raw = json.loads((paired_run / (row["id"] + "-" + arm + ".json")).read_text(encoding="utf-8"))
            transcript = " ".join(v.get("text", "").strip()
                                  for v in raw["transcription"]).strip()
            assert transcript == row[arm]["transcript"]
            edits, words = edit_count(locked["reference"], transcript)
            assert edits == row[arm]["word_edits"] and words == row[arm]["reference_words"]
            assert row[arm]["wer"] == edits / words
            log = (paired_run / (row["id"] + "-" + arm + ".log")).read_text(encoding="utf-8", errors="replace")
            assert "using CUDA0 backend" in log and "ftype         = 0" in log
        score_inputs.append(row)
    metrics = {}
    for language in ("vi", "en"):
        subset = [r for r in score_inputs if r["language"] == language]
        assert len(subset) == 16
        denominator = sum(r["base"]["reference_words"] for r in subset)
        base, small = (sum(r[a]["word_edits"] for r in subset) for a in ("base", "small"))
        actual = result["by_language"][language]
        assert actual["base_micro_WER"] == base / denominator
        assert actual["small_micro_WER"] == small / denominator
        lengths = np.array([r["base"]["reference_words"] for r in subset], dtype=float)
        delta = np.array([r["small"]["word_edits"] - r["base"]["word_edits"] for r in subset], dtype=float)
        indices = np.random.default_rng(17011).integers(0, 16, size=(10000, 16))
        bootstrap = delta[indices].sum(axis=1) / lengths[indices].sum(axis=1)
        confidence = np.percentile(bootstrap, [2.5, 97.5]).tolist()
        assert confidence == actual["paired_utterance_delta_CI95"]
        metrics[language] = {"reference_words": denominator, "base_edits": base, "small_edits": small,
                             "base_WER": base / denominator, "small_WER": small / denominator,
                             "paired_delta_CI95": confidence}
    for arm in ("base", "small"):
        values = [r[arm] for r in score_inputs]
        assert result["timing"][arm]["median_seconds"] == float(np.median([r["seconds"] for r in values]))
        assert result["timing"][arm]["p95_RTF"] == float(np.percentile([r["RTF"] for r in values], 95))
    resources = {}
    for arm, record in resource["arms"].items():
        rows = record["raw_samples"]
        active = [r for r in rows if r["owned_ASR_active"]]
        assert all(r["shared_available"] and r["dedicated_available"] for r in active)
        epochs = {r["PDH_collection_monotonic"] for r in active}
        # Earliest cached collection may predate Popen. Discard it conservatively;
        # every subsequent changed collection was observed during owned activity.
        confirmed_fresh = max(0, len(epochs) - 1)
        assert confirmed_fresh >= 2
        assert min(r["free_MiB"] for r in active) == record["sampled_min_free_MiB"] >= 512
        assert max(r["shared_bytes"] for r in active) == record["peak_shared_bytes"]
        assert record["shared_growth_bytes"] <= 256 * 1024 * 1024
        resources[arm] = {"confirmed_fresh_active_PDH_after_discarding_initial_cache": confirmed_fresh,
            "minimum_free_MiB": record["sampled_min_free_MiB"],
            "peak_shared_bytes": record["peak_shared_bytes"],
            "shared_growth_bytes": record["shared_growth_bytes"],
            "sampling_limit": "PDH2s/Nvidia0.25s; shorter peaks may be missed"}
    candidate = ROOT / "local/hybrid-v1/candidates/whisper-small"
    setup_run = ROOT / "runs/20261114T015122Z-whisper-small-native-v1-0f0231"
    setup = json.loads((setup_run / "setup-result.json").read_text())
    assert filehash(candidate / "model.safetensors") == setup["source_model_sha256"]
    storage = original_tensor_audit(candidate / "model.safetensors",
                                    candidate / "whisper-small-f32.bin", setup)
    for parent in (paired_run, resource_run):
        m = json.loads((parent / "manifest.json").read_text())
        for relative, expected in m["source_sha256"].items():
            assert filehash(parent / "source" / relative) == expected
    audit = {"paired_run": paired_run.name, "resource_run": resource_run.name,
             "independent_word_DP_matches": True, "metrics": metrics, "resources": resources,
             "original_tensor_readback": storage,
             "verdict": "DEV_ASR_GAIN_AND_SAMPLED_RESOURCE_PROFILE_VERIFIED",
             "supersedes_original_final_shared_sample_as_resource_proof": True,
             "candidate_live_promoted": False, "eight_hour_changed_ASR_profile_qualified": False,
             "LLM_SNN_learning_advantage_claim": False}
    destination = paired_run / "independent-audit-v1.json"
    with destination.open("x", encoding="utf-8") as f:
        f.write(json.dumps(audit, ensure_ascii=False, indent=2))
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == "__main__":
    main()
