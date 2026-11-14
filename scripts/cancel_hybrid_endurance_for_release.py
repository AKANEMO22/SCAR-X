"""User-directed cancellation, preserving the interrupted run without a PASS.

Gracefully close the registered shared gateway so its client run detects the
owner exit and executes its existing finally/cleanup. No forced process kill,
research source edit, historical receipt replacement or experiment restart.
"""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from scarx.runtime.maintenance import MaintenanceLease


def write_new(path, value):
    with path.open('x', encoding='utf-8') as out:
        json.dump(value, out, ensure_ascii=False, indent=2)


def main():
    run = Path((ROOT/'local/current-small-mixed-endurance-run.txt').read_text().strip()).resolve()
    if not run.is_relative_to(ROOT/'runs'):
        raise ValueError('Unexpected run location')
    manifest = json.loads((run/'manifest.json').read_text())
    identity = json.loads((run/'process-identity-v1.json').read_text())
    worker = psutil.Process(manifest['pid'])
    if abs(worker.create_time()-identity['process_created_epoch']) >= .01:
        raise RuntimeError('Worker PID was reused')
    if not any(Path(arg).resolve() == ROOT/manifest['runner'] for arg in worker.cmdline()[1:] if not arg.startswith('--')):
        raise RuntimeError('Worker command differs from this registered run')
    for name, digest in manifest['source_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != digest:
            raise RuntimeError('Frozen source changed before cancellation: '+name)
    request = {'action':'cancel_8h_for_product_release', 'requested_epoch':time.time(),
               'user_instruction':'bỏ cái tets 8h đi tập trung xử lý nhanh ra sản phẩm đi',
               'run_id':run.name, 'pid':worker.pid, 'process_created_epoch':worker.create_time(),
               'source_matches_before_cancellation':True, 'eight_hour_PASS_claimed':False}
    write_new(run/'user-cancellation-request-v1.json', request)
    spec = importlib.util.spec_from_file_location('_cancel_registered_hybrid_service', ROOT/'scripts/hybrid_service.py')
    service = importlib.util.module_from_spec(spec);spec.loader.exec_module(service)
    lease = MaintenanceLease(ROOT/'local/hybrid-v1')
    try:
        stopped = service.stop()
        print(json.dumps({'registered_gateway_stopped_gracefully':stopped,'awaiting_private_worker':worker.pid}), flush=True)
        worker.wait(timeout=120)
        receipt_path = run/'result.json'
        if not receipt_path.is_file():
            raise RuntimeError('Worker terminal without final receipt; inspect before source activation')
        raw = receipt_path.read_bytes();receipt = json.loads(raw)
        if receipt.get('gate_pass') or receipt.get('formal_P1_P2_P3_or_lifelong_pass'):
            raise RuntimeError('Interrupted run cannot claim a full-duration PASS')
        matches = all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest
                      for name,digest in manifest['source_sha256'].items())
        # No private lab process may remain after its owner exits.
        residual = []
        private = str(Path(manifest['private_data']).resolve()).casefold()
        for process in psutil.process_iter(['pid','cmdline']):
            command = process.info['cmdline'] or []
            if process.pid != psutil.Process().pid and any(private in arg.casefold() for arg in command):
                residual.append(process.pid)
        audit = {'verdict':'USER_CANCELLED_NOT_EIGHT_HOUR_PASS', 'run_id':run.name,
                 'checked_epoch':time.time(), 'pid':worker.pid,
                 'process_created_epoch':identity['process_created_epoch'],
                 'actual_worker_terminal':True, 'source_matches_before_activation':matches,
                 'request_sha256':hashlib.sha256((run/'user-cancellation-request-v1.json').read_bytes()).hexdigest(),
                 'result_sha256':hashlib.sha256(raw).hexdigest(),
                 'original_verdict':receipt.get('verdict'), 'elapsed_seconds':receipt['elapsed_seconds'],
                 'completed_video_cycles':len(receipt['video_cycles']), 'checkpoints':len(receipt['checkpoints']),
                 'residual_private_worker_pids':residual, 'service_stop':stopped,
                 'engineering_eight_hour_qualified':False, 'research_or_production_PASS':False}
        write_new(run/'user-cancellation-audit-v1.json', audit)
        if not matches or residual:
            raise RuntimeError('Cancellation cleanup/source custody requires inspection')
        print(json.dumps(audit), flush=True)
    finally:
        lease.close()


if __name__ == '__main__':
    main()
