"""Release smoke test on temporary data directories.

Does not open local/generation1. A live session on port 8765 can keep running.
Accepted teaching and rejected teaching are separate sessions. Neither result
is allowed to satisfy the other.
"""
import hashlib
import io
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def free_port():
    for _ in range(16):
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        if port != 8765:
            return port
    raise RuntimeError('could not reserve a port other than 8765')


def call(port, path, payload=None, token=None, timeout=30):
    headers = {'Host': f'127.0.0.1:{port}', 'Origin': f'http://127.0.0.1:{port}'}
    data = None
    if payload is not None:
        data = json.dumps(payload).encode()
        headers['Content-Type'] = 'application/json'
        headers['X-SCARX-Token'] = token or ''
    req = urllib.request.Request(f'http://127.0.0.1:{port}{path}', data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            body = {'error': raw.decode('utf-8', 'replace')}
        return exc.code, body


def download(port, path, token=None, timeout=30):
    headers = {'Host': f'127.0.0.1:{port}', 'Origin': f'http://127.0.0.1:{port}'}
    if token:
        headers['X-SCARX-Token'] = token
    req = urllib.request.Request(f'http://127.0.0.1:{port}{path}', headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read()


# session.json fields restored by a rejected teach. `sequence` is the audit
# cursor: a refusal may keep a rejection event id, so it is not part of the
# neural/controller identity. The events table is a log and may gain a row.
RESTORED_SESSION_FIELDS = (
    'format', 'sha256', 'dialogue_learning', 'rng', 'memories', 'probe',
    'replay_cursor', 'idle_enabled', 'idle_hz', 'idle_steps', 'active',
)


def export_parts(blob):
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        model = archive.read('model.zip')
        session = json.loads(archive.read('session.json'))
    return model, session


def model_state_sha(blob):
    """SHA-256 of model.zip only.

    That blob is the neural dump: weights, recurrent state, ticks, update
    counter, vocabulary and the capacity RNG. It is not a full-session hash.
    Controller policy and the app RNG live in session.json.
    """
    model, _session = export_parts(blob)
    return hashlib.sha256(model).hexdigest()


def model_sha(blob):
    return model_state_sha(blob)


def restored_session(session):
    missing = [name for name in RESTORED_SESSION_FIELDS if name not in session]
    if missing:
        raise KeyError('session export missing ' + ', '.join(missing))
    return {name: session[name] for name in RESTORED_SESSION_FIELDS}


def rollback_observation(before_blob, after_blob):
    """Compare a rejected or restored export without calling it a byte-identical session."""
    before_model, before_session = export_parts(before_blob)
    after_model, after_session = export_parts(after_blob)
    before_sha = hashlib.sha256(before_model).hexdigest()
    after_sha = hashlib.sha256(after_model).hexdigest()
    return {
        'model_state_sha_before': before_sha,
        'model_state_sha_after': after_sha,
        'model_state_sha_unchanged': before_sha == after_sha,
        'controller_rng_dialogue_match': restored_session(before_session) == restored_session(after_session),
        'audit_sequence_before': before_session.get('sequence'),
        'audit_sequence_after': after_session.get('sequence'),
        'identity': 'model.zip plus session controller, app RNG, dialogue, memories and idle policy',
        'excluded_audit': 'sequence cursor and the events/trace log may change',
        'regression_test': 'tests/test_rollback_full_state.py',
    }


def request(port, path, payload=None, token=None):
    status, body = call(port, path, payload, token)
    if status >= 400:
        raise urllib.error.HTTPError(path, status, str(body), hdrs=None, fp=None)
    return status, body


def wait_status(port, proc):
    deadline = time.time() + 60
    last = None
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f'server exited {proc.returncode} before status')
        try:
            status, body = request(port, '/api/status')
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last = exc
            time.sleep(0.3)
            continue
        if status == 200 and body.get('general_chat') is False and body.get('general_vision') is False:
            return body
        last = body
        time.sleep(0.3)
    raise RuntimeError(f'status not ready: {last}')


def stop(proc):
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
    log = getattr(proc, 'log', None)
    if log is not None and not log.closed:
        log.close()


def launch(port, data_dir, idle=False):
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    log = open(Path(data_dir) / 'server.log', 'w', encoding='utf-8')
    command = [sys.executable, str(ROOT / 'run_app.py'), '--port', str(port), '--data-dir', str(data_dir)]
    if not idle:
        command.append('--no-idle')
    proc = subprocess.Popen(
        command,
        cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, text=True,
    )
    proc.log = log
    return proc


def main():
    port = free_port()
    data = Path(tempfile.mkdtemp(prefix='scarx-smoke-'))
    report = {'port': port, 'data_dir': str(data), 'live_generation1_untouched': True}
    try:
        proc = launch(port, data)
        try:
            first = wait_status(port, proc)
            report['first_ticks'] = first['ticks']
            report['first_hidden'] = first['hidden']
            report['ram_bytes'] = first['ram_bytes']
            _, chat = request(port, '/api/chat', {'prompt': 'xin chao', 'max_words': 8}, first['token'])
            neural = chat.get('neural_draft') or chat.get('text')
            if not neural or chat.get('quality_gate_passed') is not False:
                raise RuntimeError(f'chat response is not an unverified neural draft: {chat}')
            if chat.get('learning_question') and not str(chat.get('text', '')).startswith('[Hỏi để học]'):
                raise RuntimeError('learning question replaced the answer without a visible label')
            _, completion = request(port, '/v1/chat/completions', {
                'model': 'scarx-generation1', 'messages': [{'role': 'user', 'content': 'xin chao'}], 'max_tokens': 8,
            })
            draft = completion.get('scarx', {}).get('neural_draft')
            if not isinstance(draft, str) or not draft.strip():
                raise RuntimeError(f'OpenAI response hid the neural draft: {completion.get("scarx")}')
            report['chat_text'] = chat['text']
            report['neural_draft'] = neural
            report['chat_source'] = chat['source']
            report['openai_exposes_neural_draft'] = draft
            taught_status = wait_status(port, proc)
            code, taught = call(port, '/api/teach', {
                'prompt': 'smoke lesson', 'answer': 'Day la bai thu.', 'repeats': 1, 'confirmation': True,
            }, taught_status['token'], timeout=90)
            report['accepted_http'] = code
            report['updates_before_teach'] = taught_status['updates']
            if code != 200:
                raise RuntimeError(f'accepted scenario was not accepted: {code} {taught}')
            metrics = taught.get('learning_metrics') or {}
            if metrics.get('accepted') is not True or metrics.get('rolled_back') is not False or not taught.get('checkpoint'):
                raise RuntimeError(f'accepted scenario missing checkpoint or metrics: {taught}')
            report['accepted_checkpoint'] = taught['checkpoint']
            report['teach'] = 'accepted'
            report['newly_acquired'] = False
            after_teach = wait_status(port, proc)
            report['updates_after_teach'] = after_teach['updates']
            if report['teach'] == 'accepted' and after_teach['updates'] <= taught_status['updates']:
                raise RuntimeError('accepted teach did not advance the update counter')
            after = after_teach
            report['ticks_after_chat'] = after['ticks']
            if after['ticks'] <= first['ticks']:
                raise RuntimeError('chat did not advance the neural clock')
        finally:
            stop(proc)
        proc = launch(port, data)
        try:
            restored = wait_status(port, proc)
            report['ticks_after_restart'] = restored['ticks']
            if restored['ticks'] != report['ticks_after_chat']:
                raise RuntimeError(f'restart lost ticks: {report["ticks_after_chat"]} -> {restored["ticks"]}')
            if restored['updates'] != report['updates_after_teach']:
                raise RuntimeError(f'restart lost updates: {report["updates_after_teach"]} -> {restored["updates"]}')
            report['updates_after_restart'] = restored['updates']
            reject_port = free_port()
            reject_data = Path(tempfile.mkdtemp(prefix='scarx-smoke-reject-'))
            report['reject_data_dir'] = str(reject_data)
            reject = launch(reject_port, reject_data)
            try:
                base = wait_status(reject_port, reject)
                before_ids = [item['id'] for item in request(reject_port, '/api/checkpoints', None, base['token'])[1]]
                before_blob = download(reject_port, '/api/export', base['token'])
                code, refused = call(reject_port, '/api/teach', {
                    'prompt': 'cảm ơn', 'answer': 'Không có gì.', 'repeats': 16, 'confirmation': True,
                }, base['token'], timeout=120)
                text = json.dumps(refused, ensure_ascii=False)
                if code != 400 or 'guard' not in text.lower():
                    raise RuntimeError(f'rejected scenario was not refused by the guard: {code} {text}')
                after = wait_status(reject_port, reject)
                after_ids = [item['id'] for item in request(reject_port, '/api/checkpoints', None, after['token'])[1]]
                observed = rollback_observation(before_blob, download(reject_port, '/api/export', after['token']))
                if after['updates'] != base['updates'] or after_ids != before_ids or not observed['model_state_sha_unchanged'] or not observed['controller_rng_dialogue_match']:
                    raise RuntimeError('rejected teach did not restore model state and controller/RNG session fields')
                report['rejected_http'] = code
                report['rejected_updates'] = after['updates']
                report['rollback'] = observed
                report['rejected_checkpoints_unchanged'] = True
            finally:
                stop(reject)
                if report.get('rollback', {}).get('model_state_sha_unchanged'):
                    shutil.rmtree(reject_data, ignore_errors=True)
            report['passed'] = True
        finally:
            stop(proc)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        shutil.rmtree(data, ignore_errors=True)
        return 0
    except Exception:
        print(f'smoke log kept at {data / "server.log"}', file=sys.stderr)
        raise


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f'SMOKE FAILED: {exc}', file=sys.stderr)
        raise SystemExit(1)
