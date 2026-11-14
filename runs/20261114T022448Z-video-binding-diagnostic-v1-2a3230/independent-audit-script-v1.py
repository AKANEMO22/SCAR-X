"""Data-only rescore of frozen video binding diagnostic; no model calls."""

from pathlib import Path
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/20261114T022448Z-video-binding-diagnostic-v1-2a3230"


def rank(text):
    text = re.sub(r"\[E\d+\]", "", text, flags=re.I).strip().casefold()
    if re.fullmatch(r"\d+", text):
        return int(text)
    values = set(map(int, re.findall(
        r"\b(?:xếp\s+thứ|hạng|thứ|rank(?:ed)?)\s*:?\s*(\d+)\b", text)))
    for term, value in (("fifteenth", 15), ("ninth", 9), ("mười lăm", 15), ("chín", 9)):
        if re.search(r"\b" + term + r"\b", text):
            values.add(value)
    return next(iter(values)) if len(values) == 1 else None


def main():
    manifest = json.loads((RUN / "manifest.json").read_text(encoding="utf-8"))
    result = json.loads((RUN / "result.json").read_text(encoding="utf-8"))
    problems = []
    for relative, expected in manifest["source_sha256"].items():
        if hashlib.sha256((RUN / "source" / relative).read_bytes()).hexdigest() != expected:
            problems.append("Source hash mismatch: " + relative)
    if result["failures"]:
        problems.append("Original run failed")
    native = {}
    for path in RUN.glob("native-*.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        # Original runner hashes the message list, not native response/randomID.
        digest = hashlib.sha256(json.dumps(value["messages"], ensure_ascii=False,
            separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        if path.stem != "native-" + digest:
            problems.append("Input/cache hash mismatch")
        if value["native_raw"]["choices"][0]["message"].get("content") != value["answer"]:
            problems.append("Stored answer differs from native output")
        native[digest] = value
    for row in result["records"]:
        value = native[row["input_sha256"]]
        actual = rank(value["answer"])
        if (row["actual_answer"] != value["answer"] or row["diagnostic_rank"] != actual
                or row["diagnostic_rank_correct"] != (actual == row["expected"])
                or row["strict_integer_correct"] != (value["answer"].strip() == str(row["expected"]))):
            problems.append("Scoring mismatch")
    assert len(native) == result["unique_native_calls"] == 16
    assert len(result["records"]) == 28
    report = {
        "run_id": RUN.name, "integrity_problems": problems,
        "unique_actual_native_calls": len(native), "arm_records": len(result["records"]),
        "untrained_SNN_not_independent_language_seed_evidence": True,
        "verdict": "KNOWN_DIAGNOSTIC_RAW_AND_SCORES_VERIFIED" if not problems else "AUDIT_FAIL",
        "is_learning_or_policy_promotion": False,
    }
    with (RUN / "independent-audit-v1.json").open("x", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
