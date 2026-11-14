"""Data-only current small-profile engineering audit; no model execution."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import time
import uuid
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from scarx.hybrid.store import Store, encode
from scarx.runtime.asr_profiles import PROFILES, verify_execution_log
from scarx.evaluation.endurance_independent_audit import video_coverage, resource_coverage, original_long_gate
from scarx.evaluation.hermes_resource_roots import verified_gateway_pid
from audit_hybrid_endurance import private_snapshot, readonly, native_identity
from monitor_hybrid_endurance import inspect


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main(run):
    run = run.resolve()
    if not run.is_relative_to(ROOT / "runs"):
        raise ValueError("Expected owned small-profile run")
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["runner"] == "scripts/hybrid_small_mixed_endurance.py"
    state = inspect(run)
    if state["actual_process_live"] or not (run / "result.json").exists():
        raise RuntimeError("A live process or missing receipt cannot be judged completed")
    final = json.loads((run / "result.json").read_text(encoding="utf-8"))
    archive_valid = all(sha((run / "source" / rel).read_bytes()) == digest
                        for rel, digest in manifest["source_sha256"].items())
    private = ROOT / "local" / run.name
    assert private == Path(manifest["private_data"]).resolve()
    db = readonly(ROOT / "local/hybrid-v1/hybrid.sqlite")
    try:
        events = [{"id": i, "created": created, "kind": kind, "payload": json.loads(raw)} for i, created, kind, raw in db.execute(
            "SELECT id,created,kind,payload FROM events WHERE actor='orchestrator' AND kind IN ('shared_endurance.started','shared_endurance.heartbeat','shared_endurance.measured_video','shared_endurance.completed') AND json_extract(payload,'$.run_id')=? ORDER BY id", (run.name,))]
    finally:
        db.close()
    starts = [e for e in events if e["kind"] == "shared_endurance.started"]
    beats = [e["payload"] for e in events if e["kind"] == "shared_endurance.heartbeat"]
    cycles = [e["payload"] for e in events if e["kind"] == "shared_endurance.measured_video"]
    ends = [e["payload"] for e in events if e["kind"] == "shared_endurance.completed"]
    assert len(starts) == len(ends) == 1 and ends[0]["verdict"] == final["verdict"]
    with readonly(private / "hybrid.sqlite") as db:
        ASR_rows = [json.loads(row[0]) for row in db.execute("SELECT payload FROM events WHERE kind='asr_result' ORDER BY id")]
        workers = [json.loads(row[0]) for row in db.execute("SELECT payload FROM events WHERE kind='worker_started' ORDER BY id")]
        errors = db.execute("SELECT COUNT(*) FROM events WHERE kind IN ('video_asr_failed','cognitive_quantum_failed')").fetchone()[0]
    profile = PROFILES["openai/whisper-small"]
    log = (private / "asr.log").read_text(encoding="utf-8", errors="replace")
    marker = "whisper_init_from_file_with_params_no_state: loading model from "
    segments = log.split(marker)[1:]
    ASR_integrity = bool(len(ASR_rows) == len(workers) == len(segments) > 0)
    for row, worker, segment in zip(ASR_rows, workers, segments):
        command = worker["command"]
        selected = Path(command[command.index("-m") + 1]).resolve()
        ASR_integrity &= (selected == private / profile.relative_path
                          and row["ASR_identity"] == profile.identity()
                          and row["actual_neural_device"] == "CUDA0"
                          and row["execution_device_log_verified"] is True
                          and row["precision_log_verified"] is True
                          and command[command.index("-dev") + 1] == "0" and "-ng" not in command)
        verify_execution_log(marker + segment, "gpu", 0)
        raw = Path(row["raw_result"]).resolve()
        assert raw.is_relative_to(private)
        value = json.loads(raw.read_text(encoding="utf-8"))
        assert row["text"] == " ".join(item.get("text", "").strip() for item in value["transcription"]).strip()
    native = native_identity(starts[0]["payload"], starts[0]["created"])
    checkpoints = private_snapshot(private)
    video = video_coverage(cycles)
    resources = resource_coverage(beats, verified_gateway_pid(Path(os.environ["LOCALAPPDATA"]) / "hermes"))
    dynamic_roots = bool(beats) and all(beat["actual_Hermes_gateway_pid"] in
        beat["resources"]["included_client_pids"] for beat in beats)
    native_pids = []
    for process in psutil.process_iter(["exe"]):
        try:
            if process.info["exe"] and Path(process.info["exe"]).resolve().is_relative_to(private):
                native_pids.append(process.pid)
        except psutil.Error:
            continue
    agreement = (len(cycles) == len(final["video_cycles"]) == len(checkpoints["streams"])
                 and len(final["checkpoints"]) == len(checkpoints["pinned_checkpoint_ids"])
                 and final["final_neural"]["updates"] == checkpoints["active_checkpoint"]["updates"]
                 and len(ASR_rows) == final.get("actual_small_ASR_results"))
    mechanical = (final["elapsed_seconds"] >= manifest["duration_seconds"] and len(cycles) >= 1
                  and len(final["checkpoints"]) >= 1 and checkpoints["active_checkpoint"]["updates"] > 0
                  and final["final_neural"]["device"] == "gpu" and errors == 0
                  and all(row["status"] == "complete" for row in checkpoints["streams"]))
    if manifest["duration_seconds"] == 28800:
        mechanical &= original_long_gate(elapsed=final["elapsed_seconds"], cycles=len(cycles),
            checkpoints=len(final["checkpoints"]), updates=final["final_neural"]["updates"],
            replay_seen=final["final_neural"]["replay_seen"], device=final["final_neural"]["device"],
            critical_failures=errors, active_capture=any(row["status"] != "complete" for row in checkpoints["streams"]))
        mechanical &= final["final_neural"]["replay_seen"] > 0
        prior = ROOT / "runs" / manifest["preflight_run_id"]
        assert sha((prior / "result.json").read_bytes()) == manifest["preflight_result_sha256"]
        assert json.loads((prior / "manifest.json").read_text(encoding="utf-8"))["source_sha256"] == manifest["source_sha256"]
        assert json.loads((prior / "small-mixed-audit-v1.json").read_text(encoding="utf-8"))["engineering_data_verified"]
    passed = bool(mechanical and archive_valid and ASR_integrity and dynamic_roots and agreement
                  and resources["sampled_512MiB_reserve_met"] and not final["failures"]
                  and not native_pids and final["gate_pass"] and native["validated_current_identity_matches"]
                  and video["skipped_intervals"] == video["ASR_failures_in_stream_records"] == 0)
    report = {"run_id": run.name, "verdict": "SMALL_MIXED_ENGINEERING_DATA_VERIFIED_ONLY" if passed else "SMALL_MIXED_ENGINEERING_DATA_NOT_QUALIFIED",
              "engineering_data_verified": passed, "duration_seconds": manifest["duration_seconds"],
              "process": state, "source_archive_valid": archive_valid,
              "ASR_calls_verified_against_native_command_log_JSON": len(ASR_rows),
              "ASR_integrity": ASR_integrity, "dynamic_Hermes_roots_each_heartbeat": dynamic_roots,
              "private_checkpoint_and_receipt_agree": agreement, "private_residual_native_pids": native_pids,
              "video": video, "resources": resources, "private_active_checkpoint": checkpoints["active_checkpoint"],
              "current_original_native_identity": native, "model_or_training_calls_in_audit": 0,
              "formal_quality_retention_self_repair_or_production_PASS": False}
    (run / "audit-observer-events-v1.json").write_text(encode(events), encoding="utf-8")
    with (run / "small-mixed-audit-v1.json").open("x", encoding="utf-8") as output:
        output.write(encode(report))
    trace = Store(ROOT / "local/hybrid-v1")
    try:
        trace.event("small_mixed_data_audit.completed", {k: v for k, v in report.items() if k not in ("video", "resources")}, "orchestrator")
    finally:
        trace.close()
    print(encode({k: report[k] for k in ("run_id", "verdict", "engineering_data_verified", "ASR_calls_verified_against_native_command_log_JSON")}))
    return passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    raise SystemExit(0 if main(parser.parse_args().run) else 1)
