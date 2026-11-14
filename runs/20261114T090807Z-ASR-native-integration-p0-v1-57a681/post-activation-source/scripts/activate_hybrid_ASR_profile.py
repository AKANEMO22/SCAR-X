"""Bounded exact-profile activation after native P0; no live feedback or training."""

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
import traceback
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from scarx.runtime.maintenance import MaintenanceLease
from scarx.evaluation.integration_call_budget import charge
from scarx.evaluation.shared_native import SharedNative
from scarx.hybrid.store import encode
from accept_hybrid_ASR_profile import prerequisites
from install_hybrid_ASR_profile import main as install
from hybrid_service import start, stop, status


def snapshot():
    database = (ROOT / "local/hybrid-v1/hybrid.sqlite").resolve()
    with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as db:
        pointer = json.loads(db.execute("SELECT payload FROM objects WHERE kind='runtime' AND id='active_checkpoint'").fetchone()[0])
        digest, payload = db.execute("SELECT digest,payload FROM checkpoints WHERE id=?", (pointer["id"],)).fetchone()
        cursor = db.execute("SELECT COALESCE(MAX(id),0) FROM events").fetchone()[0]
    if hashlib.sha256(payload).hexdigest() != digest:
        raise ValueError("Live checkpoint hash mismatch")
    return digest, payload, cursor


def main(run):
    run = run.resolve()
    if not run.is_relative_to(ROOT / "runs"):
        raise ValueError("Expected owned integration run")
    prerequisites()
    before = status()
    if (not before or not before["ready"] or before.get("failure")
            or before.get("cognition", {}).get("phase") in ("serving", "observing")):
        raise RuntimeError("Live user work or unavailable gateway; defer activation")
    install(run, False)  # Review every target before any operational change.
    with (run / "live-activation-started.json").open("x", encoding="utf-8") as out:
        out.write(encode({"started_epoch": time.time(), "old_gateway_pid": before["pid"],
                          "scope": "Selected ASR runtime engineering acceptance; no new quality test"}))
    begun = time.monotonic()
    maintenance = MaintenanceLease(ROOT / "local/hybrid-v1")
    report = {"run_id": run.name, "old_gateway_pid": before["pid"], "failures": [],
              "new_training_or_user_feedback": False, "VLM_calls_reserved": 0}
    try:
        report["graceful_stop"] = stop()
        install(run, True)
        activation = json.loads((run / "source-activation-v1.json").read_text(encoding="utf-8"))
        backup = Path(activation["private_backup"])
        saved = (backup / "private-full-live-checkpoint.bin").read_bytes()
        digest, payload, cursor = snapshot()
        if payload != saved:
            raise ValueError("Saved state changed during source activation")
        # Exercise the same normal HybridEngine constructor used by Gateway.
        # It restores full state but starts no background/model worker.
        program = '''
import sys,json,hashlib
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path.cwd()/'src'))
from scarx.hybrid.engine import HybridEngine
from scarx.hybrid.config import load_runtime_config
from scarx.hybrid.store import Store
root=Path(sys.argv[1]); store=Store(root)
try:
 cfg=load_runtime_config(root)
 engine=HybridEngine(store,SimpleNamespace(),SimpleNamespace(),cfg)
 raw=engine.model.dumps(engine.state)
 if hashlib.sha256(raw).hexdigest()!=sys.argv[2]:raise ValueError('Normal controller restore differs')
 print(json.dumps({'full_saved_payload_roundtrip_exact':True,'updates':engine.model.updates,'hidden_updates':engine.model.hidden_updates,'replay_items':len(engine.model.replay)}))
finally:store.close()
'''
        restored = subprocess.run([sys.executable, "-c", program, str(ROOT / "local/hybrid-v1"), digest],
                                  cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=20)
        (run / "normal-controller-restore.log").write_text(restored.stdout + restored.stderr, encoding="utf-8")
        if restored.returncode:
            raise RuntimeError("Normal controller restore validation failed; inspect private log")
        report["normal_constructor_full_state_restore"] = json.loads(restored.stdout)
        report["saved_full_checkpoint_sha256"] = digest
        report["startup_event_cursor"] = cursor
        budget = ROOT / "local/ASR-native-integration-p0-v1-budget.json"
        charge(budget, "VLM", 3)  # Required parity, even if Hermes wins the auto-start race.
        report["VLM_calls_reserved"] = 3
        maintenance.close()
        report["service_start"] = start()
        while True:
            if time.monotonic() - begun >= 300:
                raise TimeoutError("Activation phase exceeded 300 seconds")
            current = status()
            if current and current.get("failure"):
                raise RuntimeError("New gateway qualification failed: " + str(current["failure"]))
            if current and current.get("ready"):
                break
            time.sleep(.25)
        native = SharedNative(ROOT)
        database = (ROOT / "local/hybrid-v1/hybrid.sqlite").resolve()
        with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as db:
            events = [{"id": i, "kind": kind, "payload": json.loads(raw)} for i, kind, raw in
                      db.execute("SELECT id,kind,payload FROM events WHERE id>? AND kind IN ('native_parity_case','gateway_native_gate','native_storage_verified','neural_execution_device') ORDER BY id", (cursor,))]
        (run / "actual-startup-events.json").write_text(encode(events), encoding="utf-8")
        parity = [e["payload"] for e in events if e["kind"] == "native_parity_case"]
        if len(parity) != 3 or not all(row["passed"] for row in parity):
            raise ValueError("Expected exactly three successful startup parity calls")
        if (current["pid"] == before["pid"] or current["ASR_identity"]["name"] != "openai/whisper-small"
                or not current["ASR_identity"]["storage_verified"]
                or current["execution_placement"]["ALIF_forward_learning"] != "gpu"
                or current["neural"]["updates"] != before["neural"]["updates"]):
            raise ValueError("Runtime identity/device/learned state disagrees")
        charge(budget, "VLM")
        report["VLM_calls_reserved"] = 4
        answer = native.generate({"messages": [{"role": "user", "content": "What is 7 plus 8? Reply only with the integer."}], "max_tokens": 8})
        (run / "actual-postrestart-native-answer.json").write_text(encode(answer), encoding="utf-8")
        text = answer["choices"][0]["message"]["content"].strip()
        if text != "15":
            raise ValueError("Fixed post-restart probe did not return 15")
        _, after_payload, _ = snapshot()
        with np.load(io.BytesIO(saved), allow_pickle=False) as old, np.load(io.BytesIO(after_payload), allow_pickle=False) as new:
            arrays_exact = set(old.files) == set(new.files) and all(
                old[k].dtype == new[k].dtype and np.array_equal(old[k], new[k]) for k in old.files if k != "metadata")
            old_meta = json.loads(old["metadata"].tobytes())
            new_meta = json.loads(new["metadata"].tobytes())
            # Controller clock/phase may legitimately change after background start.
            learning_metadata_exact = {k: v for k, v in old_meta.items() if k != "controller"} == {
                k: v for k, v in new_meta.items() if k != "controller"}
        if not arrays_exact or not learning_metadata_exact:
            raise ValueError("Live learned arrays/RNG/replay changed during activation")
        report.update(verdict="LIVE_SELECTED_ASR_RUNTIME_P0_PASS_NOT_LONG_RUN_OR_QUALITY_PASS",
                      new_gateway_pid=current["pid"], new_native_worker_pid=native.worker.pid,
                      actual_startup_parity_calls=len(parity), fixed_postrestart_answer=text,
                      ASR_identity=current["ASR_identity"], execution_placement=current["execution_placement"],
                      live_neural_arrays_and_learning_RNG_replay_exact=True,
                      startup_controller_clock_changes_excluded_only_from_poststart_comparison=True,
                      resources=current["resources"], old_private_state_kept_local=True,
                      new_ASR_inference_on_live_profile=False, full_production_PASS=False)
    except Exception:
        report["failures"].append(traceback.format_exc())
        report["verdict"] = "LIVE_ASR_ACTIVATION_INCOMPLETE_INSPECT_BEFORE_ANY_EXTRA_CALL"
    finally:
        maintenance.close()
        report["seconds"] = time.monotonic() - begun
        with (run / "live-activation-result.json").open("x", encoding="utf-8") as out:
            out.write(encode(report))
        print(encode({k: report[k] for k in ("run_id", "verdict", "failures", "seconds")}))
    return not report["failures"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    raise SystemExit(0 if main(parser.parse_args().run) else 1)
