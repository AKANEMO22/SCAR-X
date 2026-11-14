"""Read-only process/source/receipt inspection; never restart a lab or gateway."""

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import time
import psutil

ROOT = Path(__file__).resolve().parents[1]


def read_heartbeat(path):
    for attempt in range(3):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            if attempt == 2:
                return None  # Concurrent overwrite is not proof the process died.
            time.sleep(.05)


def inspect(run):
    run = Path(run).resolve()
    if ROOT / "runs" not in run.parents:
        raise ValueError("Run must be inside this checkout")
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    runner = manifest.get("runner", "scripts/hybrid_shared_endurance.py")
    if runner not in ("scripts/hybrid_shared_endurance.py", "scripts/hybrid_small_mixed_endurance.py"):
        raise ValueError("Unregistered endurance runner")
    process = None
    identity_path = run / "process-identity-v1.json"
    identity = json.loads(identity_path.read_text(encoding="utf-8")) if identity_path.exists() else {}
    try:
        candidate = psutil.Process(manifest["pid"])
        if candidate.is_running() and any(
            Path(value).resolve() == ROOT / runner
            for value in candidate.cmdline()
        ) and (not identity.get("process_created_epoch")
               or abs(candidate.create_time() - identity["process_created_epoch"]) < .01):
            process = candidate
    except psutil.Error:
        pass
    final_path = run / "result.json"
    heartbeat_path = run / "heartbeat.json"
    heartbeat = (read_heartbeat(heartbeat_path)
                 if heartbeat_path.exists() else None)
    if final_path.exists():
        final = json.loads(final_path.read_text(encoding="utf-8"))
        state = "terminal_receipt_present"
    else:
        final = None
        state = "verified_live_process" if process else "pid_missing_no_final_receipt"
    database = ROOT / "local/hybrid-v1/hybrid.sqlite"
    after = None
    with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as db:
        rows = db.execute(
            "SELECT id,payload FROM events WHERE kind='shared_endurance.started' ORDER BY id DESC LIMIT 10"
        ).fetchall()
        for ident, payload in rows:
            if json.loads(payload)["run_id"] == run.name:
                after = ident - 1
                break
    return {
        "checked_epoch": time.time(), "run_id": run.name, "state": state,
        "actual_process_live": bool(process),
        "pid": process.pid if process else manifest["pid"],
        "process_created_epoch": process.create_time() if process else None,
        "frozen_source_matches": all(
            hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == digest
            for rel, digest in manifest["source_sha256"].items()),
        "heartbeat_elapsed_seconds": heartbeat["elapsed_seconds"] if heartbeat else None,
        "heartbeat_file_age_seconds": time.time() - heartbeat_path.stat().st_mtime
            if heartbeat_path.exists() else None,
        "completed_video_cycles": heartbeat["video_cycles"] if heartbeat else None,
        "checkpoint_count": heartbeat["checkpoints"] if heartbeat else None,
        "lab_neural": heartbeat["neural"] if heartbeat else None,
        "final_verdict": final["verdict"] if final else None,
        "trace_link": f"http://127.0.0.1:8775/?after={after}&tab=trace"
            if after is not None else None,
        "inspection_does_not_restart": True,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("--receipt", type=Path)
    args = p.parse_args()
    result = inspect(args.run)
    if args.receipt:
        destination = args.receipt.resolve()
        if destination.parent != args.run.resolve():
            raise ValueError("Receipt must stay in inspected run")
        with destination.open("x", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False))
