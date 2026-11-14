"""Data-only current-source/state audit after selected-ASR runtime activation."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from scarx.hybrid.store import Store, encode
from scarx.evaluation.shared_native import SharedNative
from scarx.runtime.asr_profiles import PROFILES, verify_asr_storage
from scarx.evaluation.hermes_resource_roots import verified_gateway_pid
from activate_hybrid_ASR_profile import snapshot


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main(run):
    run = run.resolve()
    if not run.is_relative_to(ROOT / "runs"):
        raise ValueError("Expected owned integration run")
    read = lambda name: json.loads((run / name).read_text(encoding="utf-8"))
    private, activation, receipt = (read(name) for name in
        ("private-audit-v1.json", "live-activation-result.json", "source-activation-v1.json"))
    assert private["actual_ASR_calls"] == 4
    assert private["verdict"] == "ACTUAL_SELECTED_ASR_PRIVATE_P0_DATA_AUDIT_PASS_ONLY"
    assert activation["verdict"] == "LIVE_SELECTED_ASR_RUNTIME_P0_PASS_NOT_LONG_RUN_OR_QUALITY_PASS"
    assert not activation["failures"] and activation["seconds"] <= 300
    assert activation["normal_constructor_full_state_restore"]["full_saved_payload_roundtrip_exact"]
    manifest = read("manifest.json")
    for filename, expected in manifest["candidate_source_sha256"].items():
        assert sha((ROOT / "src/scarx/hybrid" / filename).read_bytes()) == expected
    assert sha((ROOT / "src/scarx/hybrid/resources.py").read_bytes()) == manifest["resource_candidate_sha256"]
    for rel, identity in receipt["targets"].items():
        assert sha((ROOT / rel).read_bytes()) == identity["after_sha256"]
    backup = Path(receipt["private_backup"])
    assert backup.resolve().is_relative_to(ROOT / "local/hybrid-v1/profile-backups")
    old_payload = (backup / "private-full-live-checkpoint.bin").read_bytes()
    assert sha(old_payload) == receipt["saved_private_live_checkpoint_sha256"] == activation["saved_full_checkpoint_sha256"]
    _, new_payload, _ = snapshot()
    with np.load(io.BytesIO(old_payload), allow_pickle=False) as old, np.load(io.BytesIO(new_payload), allow_pickle=False) as new:
        assert set(old.files) == set(new.files)
        for key in old.files:
            if key != "metadata":
                assert old[key].dtype == new[key].dtype and np.array_equal(old[key], new[key])
        old_meta = json.loads(old["metadata"].tobytes())
        new_meta = json.loads(new["metadata"].tobytes())
        assert {k: v for k, v in old_meta.items() if k != "controller"} == {
            k: v for k, v in new_meta.items() if k != "controller"}
    budget = json.loads((ROOT / "local/ASR-native-integration-p0-v1-budget.json").read_text(encoding="utf-8"))
    assert budget == {"limits": {"ASR": 4, "VLM": 4}, "attempted": {"ASR": 4, "VLM": 4}}
    events = read("actual-startup-events.json")
    parity = [e["payload"] for e in events if e["kind"] == "native_parity_case"]
    assert len(parity) == 3 and all(row["passed"] for row in parity)
    assert all(e["id"] > activation["startup_event_cursor"] for e in events)
    assert read("actual-postrestart-native-answer.json")["choices"][0]["message"]["content"].strip() == "15"
    native = SharedNative(ROOT)
    status = native.status()
    assert native.owner.pid == activation["new_gateway_pid"] and native.worker.pid == activation["new_native_worker_pid"]
    assert status["ready"] and not status["failure"]
    assert status["neural"]["updates"] == old_meta["updates"] == new_meta["updates"]
    profile = PROFILES["openai/whisper-small"]
    storage = verify_asr_storage(ROOT / "local/hybrid-v1", profile.name, profile.revision, "gpu")
    assert status["ASR_identity"]["name"] == profile.name and status["ASR_identity"]["storage_verified"]
    assert all(status["execution_placement"][key] == "gpu" for key in
               ("VLM_vision", "ASR", "ALIF_forward_learning"))
    gateway = verified_gateway_pid(Path.home() / "AppData/Local/hermes")
    assert gateway in status["resources"]["included_client_pids"]
    report = {"run_id": run.name, "verdict": "LIVE_ASR_SOURCE_STORAGE_PERSISTENCE_DATA_VERIFIED_NOT_PRODUCTION_PASS",
              "native_owner_pid": native.owner.pid, "native_worker_pid": native.worker.pid,
              "ASR_model": profile.identity(), "actual_ASR_private_calls": 4,
              "actual_VLM_startup_and_probe_calls": 4, "private_original_precision_storage": storage,
              "installed_source_matches_reviewed_candidate": True,
              "full_controller_restore_before_start_verified": True,
              "learned_arrays_RNG_replay_preserved_after_start": True,
              "updates": new_meta["updates"], "hidden_updates": new_meta["hidden_updates"],
              "dynamic_Hermes_gateway_counted": True,
              "new_ASR_live_transcription_or_8h_profile_claim": False,
              "neural_quality_or_self_repair_PASS": False,
              "scope": "Same-orchestrator data-only audit, not outside peer review"}
    with (run / "live-data-audit-v1.json").open("x", encoding="utf-8") as out:
        out.write(encode(report))
    store = Store(ROOT / "local/hybrid-v1")
    try:
        event = store.event("ASR_profile.live_activation_verified", report, "orchestrator")
    finally:
        store.close()
    print(encode({"verdict": report["verdict"], "trace_event_id": event,
                  "updates_preserved": report["updates"], "native_worker_pid": native.worker.pid}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    main(parser.parse_args().run)
