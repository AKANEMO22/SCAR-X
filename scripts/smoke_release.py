"""Release smoke test on a temporary data directory.

Does not open local/generation1 and does not teach. A live session on port 8765
can keep running.
"""
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
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


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


def launch(port, data_dir):
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    log = open(Path(data_dir) / 'server.log', 'w', encoding='utf-8')
    proc = subprocess.Popen(
        [sys.executable, str(ROOT / 'run_app.py'), '--port', str(port), '--data-dir', str(data_dir), '--no-idle'],
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
            report['teach_http'] = code
            report['updates_before_teach'] = taught_status['updates']
            message = json.dumps(taught, ensure_ascii=False)
            if code == 200:
                metrics = taught.get('learning_metrics') or {}
                if metrics.get('accepted') is not True or metrics.get('rolled_back') is not False:
                    raise RuntimeError(f'teach 200 without an accepted update: {message}')
                report['teach'] = 'accepted'
            elif code == 400 and 'guard' in message.lower():
                report['teach'] = 'rejected_and_rolled_back'
            else:
                raise RuntimeError(f'unexpected teach result {code}: {message}')
            after_teach = wait_status(port, proc)
            report['updates_after_teach'] = after_teach['updates']
            if report['teach'] == 'accepted' and after_teach['updates'] <= taught_status['updates']:
                raise RuntimeError('accepted teach did not advance the update counter')
            if report['teach'] != 'accepted' and after_teach['updates'] != taught_status['updates']:
                raise RuntimeError('rejected teach changed the update counter')
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
