"""Audit real public-hook fixture observations; no inferred neural capability."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(run, output=None):
    manifest = json.loads((run / "manifest.json").read_text())
    report = json.loads((run / "baseline-actual-file-results.json").read_text(encoding="utf-8"))
    assert not report["failures"] and len(report["calls"]) == 14
    for rel, expected in manifest["source_before_sha256"].items():
        assert hashlib.sha256((run / "source-before" / rel).read_bytes()).hexdigest() == expected
    rows = {r["case"]:r for r in report["calls"]}
    decode = lambda case: json.loads(rows[case]["raw_tool_result"])
    assert decode("empty") == decode("exclusive"), "Expected identical false-empty envelope"
    source = Path(rows["exclusive"]["arguments"]["path"])
    assert source.resolve().is_relative_to((ROOT / "local" / run.name).resolve())
    # The primary Windows helper used write_text with normal CRLF translation.
    assert source.read_bytes() == b"Owned readable source.\r\n"
    for row in report["calls"]:
        observed = row["public_hook_observation"]
        assert observed["tool"] == "read_file"
        assert observed["reported_status"] == row["metadata_extractor"]["status"]
        assert observed["result_sha256"] == hashlib.sha256(json.dumps(
            json.loads(row["raw_tool_result"]),ensure_ascii=False,sort_keys=True,
            separators=(",",":"),allow_nan=False).encode()).hexdigest()
        assert str(ROOT) not in json.dumps(observed)
    assert decode("partial")["truncated"] and not decode("full")["truncated"]
    assert not decode("misleading").get("error") and rows["misleading"]["public_hook_observation"]["reported_status"] == "ok"
    changes = [r["public_hook_observation"]["result_sha256"] for r in report["calls"] if r["case"] == "changing"]
    assert len(set(changes)) == 3
    result = {"run_id":run.name,"actual_dispatcher_public_hook_calls":14,
        "actual_public_hook_data_integrity_verified":True,
        "empty_and_exclusively_locked_file_reported_envelopes_identical":True,
        "locked_source_bytes_after_handle_closed":source.stat().st_size,
        "locked_source_construction": f"Primary Windows fixture writes{source.stat().st_size}CRLFbytes then holds a GENERIC_READ/share0 handle throughout actual tool call; no write/delete during call",
        "metadata_only_parser_can_distinguish_this_pair":False,
        "binary_fixture_tool_report":decode("binary"),
        "binary_fixture_caveat":"Tool heuristically interpreted23oddbytes as UTF16BE with replacement. Harness intended binary; no universal encoding oracle is inferred.",
        "changed_result_digest_preserved":True,"new_neural_learning":False,
        "observer_candidate_deployed":False,
        "verdict":"UPSTREAM_FALSE_EMPTY_CONTRACT_FAIL_DO_NOT_TRAIN_ON_REPORTED_SUCCESS",
        "next_action":"Isolated fail-closed source-read repair; keep this raw failure and active8h source unchanged"}
    with (output or run / "independent-contract-audit-v1.json").open("x",encoding="utf-8") as f:
        json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k != "binary_fixture_tool_report"},ensure_ascii=False))


if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument("run",type=Path);p.add_argument("--output",type=Path)
    a=p.parse_args();main(a.run.resolve(),a.output)
