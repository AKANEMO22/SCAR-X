"""Product end-to-end and stability checks on temporary sessions.

Does not open local/generation1 and does not bind port 8765.
"""
import json
import os
import shutil
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import psutil

sys.path.insert(0, str(Path(__file__).resolve().parent))
import smoke_release as smoke


ROOT = smoke.ROOT
TWO_GB = 2_000_000_000


def listening(port):
    with socket.socket() as sock:
        sock.settimeout(0.4)
        return sock.connect_ex(('127.0.0.1', port)) == 0


def listener_pid(port):
    script = (
        f"(Get-NetTCPConnection -LocalPort {int(port)} -State Listen -ErrorAction SilentlyContinue "
        "| Select-Object -ExpandProperty OwningProcess -First 1)"
    )
    result = subprocess.run(
        ['powershell', '-NoProfile', '-Command', script],
        capture_output=True, text=True, timeout=30,
    )
    text = (result.stdout or '').strip()
    return int(text) if text.isdigit() else None


def processes_for(data_dir):
    needle = str(data_dir)
    found = []
    for proc in psutil.process_iter(['pid', 'cmdline']):
        command = ' '.join(proc.info.get('cmdline') or [])
        if needle in command and proc.pid != os_pid():
            found.append(proc.pid)
    return found


def os_pid():
    return psutil.Process().pid


def sqlite_report(data_dir):
    database = Path(data_dir) / 'generation1.sqlite'
    connection = sqlite3.connect(f'file:{database}?mode=ro', uri=True)
    try:
        integrity = connection.execute('PRAGMA integrity_check').fetchone()[0]
        journal = connection.execute('PRAGMA journal_mode').fetchone()[0]
        checkpoints = connection.execute('SELECT COUNT(*) FROM checkpoints').fetchone()[0]
    finally:
        connection.close()
    return {
        'integrity_check': integrity,
        'journal_mode': journal,
        'wal_present': (Path(data_dir) / 'generation1.sqlite-wal').exists(),
        'checkpoint_rows': checkpoints,
    }


def checkpoint_ids(port, token):
    _, rows = smoke.request(port, '/api/checkpoints', None, token)
    return [row['id'] for row in rows]


def reserve_port():
    port = smoke.free_port()
    if port == 8765:
        raise RuntimeError('refusing to bind the live-session port')
    return port


def provenance():
    sha = subprocess.run(
        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True, timeout=15,
    )
    return {
        'role': 'automated_e2e',
        'command': 'python scripts/e2e_release.py',
        'python': sys.version,
        'git_sha': (sha.stdout or '').strip() or None,
        'not_a_ci_certificate': True,
    }


def main():
    live_before = listener_pid(8765)
    report = {
        'provenance': provenance(),
        'live_port_8765': {'before': live_before, 'status': 'present' if live_before is not None else 'N/A'},
        'live_generation1_untouched': True,
        'checks': {},
    }
    failures = []

    def check(name, ok, detail):
        report['checks'][name] = {'pass': bool(ok), 'detail': detail}
        if not ok:
            failures.append(name)

    data = Path(tempfile.mkdtemp(prefix='scarx-e2e-'))
    port = reserve_port()
    report['port'] = port
    report['data_dir'] = str(data)
    proc = None
    try:
        proc = smoke.launch(port, data)
        status = smoke.wait_status(port, proc)
        token = status['token']
        report['ram_bytes'] = status['ram_bytes']
        report['memory_limit'] = status.get('memory_limit')
        report['hidden'] = status['hidden']
        report['updates_loaded'] = status['updates']
        check('ram_under_2gb', status['ram_bytes'] < TWO_GB, status['ram_bytes'])
        check('general_chat_false', status.get('general_chat') is False, status.get('general_chat'))
        check('general_vision_false', status.get('general_vision') is False, status.get('general_vision'))

        missing, missing_body = smoke.call(port, '/api/does-not-exist')
        check('http_404', missing == 404, {'status': missing, 'body': missing_body})
        denied, denied_body = smoke.call(port, '/api/chat', {'prompt': 'xin chao'}, token='')
        check('http_403_without_token', denied == 403, {'status': denied, 'body': denied_body})
        unknown, unknown_body = smoke.call(port, '/api/no-such-action', {'x': 1}, token)
        check('http_404_post', unknown == 404, {'status': unknown, 'body': unknown_body})

        _, chat = smoke.request(port, '/api/chat', {'prompt': 'xin chao', 'max_words': 8}, token)
        neural = chat.get('neural_draft') or chat.get('text')
        check('chat', bool(neural) and chat.get('quality_gate_passed') is False, {
            'text': chat.get('text'), 'neural_draft': chat.get('neural_draft'), 'source': chat.get('source'),
        })
        _, completion = smoke.request(port, '/v1/chat/completions', {
            'model': 'scarx-generation1', 'messages': [{'role': 'user', 'content': 'xin chao'}], 'max_tokens': 8,
        })
        draft = (completion.get('scarx') or {}).get('neural_draft')
        check('openai_neural_draft', isinstance(draft, str) and bool(draft.strip()), draft)
        after_chat = smoke.wait_status(port, proc)
        foreground = [row for row in after_chat.get('trace', []) if row.get('mode') == 'foreground']
        check('foreground_trace', len(foreground) > 0, len(foreground))

        _, examples = smoke.request(port, '/api/examples')
        sample = examples[0]
        _, vision = smoke.request(port, '/api/vision', {'image': sample['image']}, after_chat['token'])
        check('digit_vision_endpoint', isinstance(vision.get('digit'), int) and 0 <= vision['digit'] <= 9 and 'digit' in str(vision.get('scope', '')), {
            'predicted': vision.get('digit'), 'expected_label_not_fed_to_model': sample.get('expected_digit'), 'scope': vision.get('scope'),
        })

        seen = smoke.wait_status(port, proc)
        saved_code, saved = smoke.call(port, '/api/checkpoint', {}, seen['token'])
        baseline = smoke.wait_status(port, proc)
        baseline_blob = smoke.download(port, '/api/export', baseline['token'])
        baseline_sha = smoke.model_state_sha(baseline_blob)
        baseline_ids = checkpoint_ids(port, baseline['token'])
        baseline_id = saved.get('checkpoint') if saved_code == 200 else None
        check('manual_checkpoint', saved_code == 200 and baseline_id in baseline_ids and baseline['updates'] == seen['updates'], {
            'http': saved_code, 'checkpoint': baseline_id, 'updates': baseline['updates'],
        })

        code, taught = smoke.call(port, '/api/teach', {
            'prompt': 'smoke lesson', 'answer': 'Day la bai thu.', 'repeats': 1, 'confirmation': True,
        }, baseline['token'], timeout=90)
        metrics = (taught or {}).get('learning_metrics') or {}
        accepted = code == 200 and metrics.get('accepted') is True and metrics.get('rolled_back') is False and bool(taught.get('checkpoint'))
        after_accept = smoke.wait_status(port, proc)
        accept_blob = smoke.download(port, '/api/export', after_accept['token'])
        accept_sha = smoke.model_state_sha(accept_blob)
        accept_ids = checkpoint_ids(port, after_accept['token'])
        check('accepted_teach', accepted and after_accept['updates'] > baseline['updates'] and accept_sha != baseline_sha and taught['checkpoint'] in accept_ids and baseline_id in accept_ids, {
            'http': code, 'updates_before': baseline['updates'], 'updates_after': after_accept['updates'],
            'checkpoint': taught.get('checkpoint'), 'weight_delta': taught.get('weight_delta'),
            'model_state_sha_changed': accept_sha != baseline_sha,
            'newly_acquired_skill': False,
        })

        code, refused = smoke.call(port, '/api/teach', {
            'prompt': 'cảm ơn', 'answer': 'Không có gì.', 'repeats': 16, 'confirmation': True,
        }, after_accept['token'], timeout=120)
        text = json.dumps(refused, ensure_ascii=False)
        after_reject = smoke.wait_status(port, proc)
        reject_obs = smoke.rollback_observation(accept_blob, smoke.download(port, '/api/export', after_reject['token']))
        reject_ids = checkpoint_ids(port, after_reject['token'])
        check('rejected_model_and_controller_rollback', code == 400 and 'guard' in text.lower() and after_reject['updates'] == after_accept['updates'] and reject_obs['model_state_sha_unchanged'] and reject_obs['controller_rng_dialogue_match'] and reject_ids == accept_ids and taught.get('checkpoint') in reject_ids, {
            'http': code, 'message': refused, 'updates': after_reject['updates'], 'rollback': reject_obs,
        })

        restored_code, restored = smoke.call(port, '/api/restore', {'checkpoint': baseline_id}, after_reject['token'], timeout=60)
        after_restore = smoke.wait_status(port, proc)
        restore_blob = smoke.download(port, '/api/export', after_restore['token'])
        restore_sha = smoke.model_state_sha(restore_blob)
        _, baseline_session = smoke.export_parts(baseline_blob)
        _, restore_session = smoke.export_parts(restore_blob)
        policy = [name for name in smoke.RESTORED_SESSION_FIELDS if name != 'active']
        policy_match = {name: baseline_session[name] for name in policy} == {name: restore_session[name] for name in policy}
        restore_ids = checkpoint_ids(port, after_restore['token'])
        check('restore_baseline', restored_code == 200 and after_restore['updates'] == baseline['updates'] and restore_sha == baseline_sha and policy_match and baseline_id in restore_ids and taught.get('checkpoint') in restore_ids, {
            'http': restored_code,
            'updates': after_restore['updates'],
            'expected_updates': baseline['updates'],
            'model_state_sha_matches_snapshot': restore_sha == baseline_sha,
            'controller_rng_dialogue_match_except_new_checkpoint_id': policy_match,
            'active_checkpoint_after_restore': restore_session.get('active'),
            'audit_sequence_before': baseline_session.get('sequence'),
            'audit_sequence_after': restore_session.get('sequence'),
            'snapshot_kept': baseline_id in restore_ids,
            'accepted_checkpoint_kept': taught.get('checkpoint') in restore_ids,
            'restored_from': restored.get('restored_from') if isinstance(restored, dict) else restored,
        })
        report['committed_updates_after_restore'] = after_restore['updates']

        other = Path(tempfile.mkdtemp(prefix='scarx-e2e-port-'))
        starter = subprocess.run(
            ['powershell', '-NoProfile', '-File', str(ROOT / 'Start-SCARX.ps1'), '--port', str(port), '--data-dir', str(other), '--no-idle'],
            capture_output=True, text=True, timeout=40, cwd=ROOT, encoding='utf-8', errors='replace',
        )
        check('occupied_port_refused', starter.returncode != 0 and 'already in use' in (starter.stderr + starter.stdout).lower(), {
            'returncode': starter.returncode, 'stderr': (starter.stderr or '')[-500:], 'stdout': (starter.stdout or '')[-500:],
        })
        shutil.rmtree(other, ignore_errors=True)

        locked_port = reserve_port()
        lock_env = os.environ.copy()
        lock_env['PYTHONIOENCODING'] = 'utf-8'
        locker = subprocess.run(
            [sys.executable, str(ROOT / 'run_app.py'), '--port', str(locked_port), '--data-dir', str(data), '--no-idle'],
            capture_output=True, text=True, timeout=40, cwd=ROOT, env=lock_env, encoding='utf-8', errors='replace',
        )
        combined = locker.stdout + locker.stderr
        check('session_lock', locker.returncode != 0 and 'đang được một phiên' in combined, {
            'returncode': locker.returncode, 'output': combined[-500:],
        })

        ram_again = smoke.wait_status(port, proc)['ram_bytes']
        report['ram_bytes_after_work'] = ram_again
        check('ram_after_work_under_2gb', ram_again < TWO_GB, ram_again)
    except Exception as exc:
        check('session_exception', False, f'{type(exc).__name__}: {exc}')
    finally:
        if proc is not None:
            smoke.stop(proc)

    if proc is not None and proc.poll() is None:
        failures.append('shutdown')
    else:
        check('clean_shutdown', not listening(port) and not processes_for(data), {
            'listening': listening(port), 'orphans': processes_for(data),
        })

    if (data / 'generation1.sqlite').exists():
        try:
            original_sql = sqlite_report(data)
            check('sqlite_integrity', original_sql['integrity_check'] == 'ok' and original_sql['journal_mode'].lower() == 'wal', original_sql)
            backup = Path(tempfile.mkdtemp(prefix='scarx-e2e-backup-'))
            shutil.copytree(data, backup, dirs_exist_ok=True)
            copy_sql = sqlite_report(backup)
            check('backup_copy_integrity', copy_sql['integrity_check'] == 'ok', copy_sql)
            copy_port = reserve_port()
            copy_proc = smoke.launch(copy_port, backup)
            try:
                copy_status = smoke.wait_status(copy_port, copy_proc)
                expected_updates = report.get('committed_updates_after_restore')
                check('backup_starts', expected_updates is not None and copy_status['updates'] == expected_updates, {
                    'updates': copy_status['updates'], 'expected': expected_updates, 'ram_bytes': copy_status['ram_bytes'],
                })
                copy_proc.kill()
                copy_proc.wait(timeout=20)
                report['crash_killed_pid'] = copy_proc.pid
                restarted = smoke.launch(copy_port, backup)
                try:
                    crashed = smoke.wait_status(copy_port, restarted)
                    check('crash_restart_keeps_committed_updates', crashed['updates'] == copy_status['updates'], {
                        'before_kill': copy_status['updates'], 'after_restart': crashed['updates'],
                    })
                finally:
                    smoke.stop(restarted)
            finally:
                smoke.stop(copy_proc)
            if report['checks'].get('crash_restart_keeps_committed_updates', {}).get('pass'):
                shutil.rmtree(backup, ignore_errors=True)
            report['backup_dir_removed_after_pass'] = not backup.exists()
        except Exception as exc:
            check('stability_exception', False, f'{type(exc).__name__}: {exc}')

    idle_data = Path(tempfile.mkdtemp(prefix='scarx-e2e-idle-'))
    idle_port = reserve_port()
    idle = smoke.launch(idle_port, idle_data, idle=True)
    report['idle_port'] = idle_port
    try:
        idle_status = smoke.wait_status(idle_port, idle)
        check('idle_starts_enabled', idle_status['idle']['enabled'] is True, idle_status['idle'])
        time.sleep(0.6)
        moved = smoke.wait_status(idle_port, idle)
        check('idle_ticks_advance', moved['ticks'] > idle_status['ticks'], {
            'before': idle_status['ticks'], 'after': moved['ticks'],
        })
        code, paused = smoke.call(idle_port, '/api/idle', {'enabled': False}, moved['token'])
        paused_status = smoke.wait_status(idle_port, idle)
        check('idle_switch_off', code == 200 and paused_status['idle']['enabled'] is False, paused_status['idle'])
        code, resumed = smoke.call(idle_port, '/api/idle', {'enabled': True, 'hz': 5}, paused_status['token'])
        resumed_status = smoke.wait_status(idle_port, idle)
        check('idle_switch_on', code == 200 and resumed_status['idle']['enabled'] is True and resumed_status['idle']['hz'] == 5, resumed_status['idle'])
        report['idle_toggle_bodies'] = {'off': paused, 'on': resumed}
    finally:
        smoke.stop(idle)
        if not processes_for(idle_data):
            shutil.rmtree(idle_data, ignore_errors=True)

    ui_data = Path(tempfile.mkdtemp(prefix='scarx-e2e-ui16-'))
    ui_port = reserve_port()
    ui = smoke.launch(ui_port, ui_data)
    report['ui_repeats_16'] = []
    try:
        ui_status = smoke.wait_status(ui_port, ui)
        for prompt, answer in (('smoke lesson', 'Day la bai thu.'), ('chào bạn', 'Chào bạn.')):
            code, body = smoke.call(ui_port, '/api/teach', {
                'prompt': prompt, 'answer': answer, 'repeats': 16, 'confirmation': True,
            }, ui_status['token'], timeout=120)
            metrics = (body or {}).get('learning_metrics') or {}
            outcome = {
                'prompt': prompt, 'answer': answer, 'http': code,
                'accepted': code == 200 and metrics.get('accepted') is True and metrics.get('rolled_back') is False,
                'message': body.get('error') if isinstance(body, dict) else body,
            }
            report['ui_repeats_16'].append(outcome)
            ui_status = smoke.wait_status(ui_port, ui)
            if outcome['accepted']:
                break
    finally:
        smoke.stop(ui)
        shutil.rmtree(ui_data, ignore_errors=True)

    live_after = listener_pid(8765)
    report['live_port_8765']['after'] = live_after
    if live_before is None and live_after is None:
        report['live_port_8765']['status'] = 'N/A'
        report['checks']['live_8765'] = {'pass': True, 'status': 'N/A', 'detail': 'no listener on 8765'}
    elif live_before is not None and live_after == live_before:
        report['live_port_8765']['status'] = 'unchanged'
        check('live_8765_unchanged', True, {'before': live_before, 'after': live_after})
    else:
        report['live_port_8765']['status'] = 'changed'
        check('live_8765_unchanged', False, {'before': live_before, 'after': live_after})
    report['failures'] = failures
    report['passed'] = not failures
    if report['passed']:
        shutil.rmtree(data, ignore_errors=True)
    else:
        report['kept_data_dir'] = str(data)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f'E2E FAILED: {type(exc).__name__}: {exc}', file=sys.stderr)
        raise SystemExit(1)
