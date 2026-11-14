"""Reversible exact-source reader repair; preserve all model/state/config files."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def replace_file(target, raw):
    fd, temporary = tempfile.mkstemp(prefix=".scarx-reader-", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd,"wb") as f:
            f.write(raw);f.flush();os.fsync(f.fileno())
        os.replace(temporary,target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main(receipt_path, apply):
    receipt_path = receipt_path.resolve()
    assert receipt_path.is_relative_to(ROOT / "runs")
    acceptance = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert acceptance["verdict"] == "ENGINEERING_READ_CONTRACT_REPAIR_QUALIFIED_WITH_EXISTING_OS_TEST_FAILURES"
    assert not acceptance["new_canonical_failures"] and acceptance["canonical_new_contract_tests_passed"] == 11
    folder = receipt_path.parent / acceptance["source_artifacts"]
    candidate = (folder / "file_operations.py").read_bytes()
    sibling = (folder / "file_operations_read_contract.py").read_bytes()
    assert digest(candidate) == acceptance["core_after_sha256"]
    assert digest(sibling) == acceptance["sibling_sha256"]
    tools = Path(os.environ["LOCALAPPDATA"]) / "hermes/hermes-agent/tools"
    target, companion = tools / "file_operations.py", tools / "file_operations_read_contract.py"
    assert target.resolve().is_relative_to(tools.resolve())
    original = target.read_bytes()
    if original == candidate and companion.is_file() and companion.read_bytes() == sibling:
        print(json.dumps({"already_installed":True,"running_workers_reloaded":False}));return
    if digest(original) != acceptance["core_before_sha256"]:
        raise RuntimeError("Live Hermes source changed; refuse overwriting concurrent edits")
    if companion.exists():
        raise RuntimeError("Sibling module already exists; review before overwriting")
    spec=importlib.util.spec_from_file_location("reader_patch",ROOT / "integrations/hermes/file_read_contract_patch.py")
    patch=importlib.util.module_from_spec(spec);spec.loader.exec_module(patch)
    assert patch.patch_bytes(original) == candidate
    compile(candidate,str(target),"exec");compile(sibling,str(companion),"exec")
    if not apply:
        print(json.dumps({"reviewed_source_matches":True,"core_after_sha256":digest(candidate),"files_to_change":2,"applied":False}));return
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backup=ROOT / "local/hermes-file-read-contract-backups" / stamp
    backup.mkdir(parents=True)
    (backup / "file_operations.py.before").write_bytes(original)
    if target.read_bytes() != original or companion.exists():
        raise RuntimeError("Concurrent Hermes edit after backup; installation refused")
    replace_file(companion,sibling)
    try:
        if target.read_bytes() != original:
            raise RuntimeError("Concurrent Hermes edit; core replacement refused")
        replace_file(target,candidate)
    except Exception:
        if companion.read_bytes() == sibling:companion.unlink()
        raise
    assert target.read_bytes() == candidate and companion.read_bytes() == sibling
    result={"installed_UTC":stamp,"acceptance_receipt":str(receipt_path),"target":str(target),
        "before_sha256":digest(original),"after_sha256":digest(candidate),
        "sibling":str(companion),"sibling_sha256":digest(sibling),"backup":str(backup),
        "running_workers_reloaded":False,"neural_or_user_state_files_changed":False}
    (backup / "receipt.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument("receipt",type=Path);p.add_argument("--apply",action="store_true")
    a=p.parse_args();main(a.receipt,a.apply)
