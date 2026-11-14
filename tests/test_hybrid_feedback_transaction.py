"""Actual shipped HTTP handler + disposable ALIF; no VLM or live data."""

import ast
from concurrent.futures import ThreadPoolExecutor
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from http.client import RemoteDisconnected
import json
from pathlib import Path
import secrets
import socket
import threading
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen

import pytest
import scarx.hybrid.gateway as gateway_module
from scarx.hybrid.config import HybridConfig
from scarx.hybrid.engine import HybridEngine
from scarx.hybrid.learner import ACTIONS, features
from scarx.hybrid.store import Store, encode

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def fixture(tmp_path):
    store = Store(tmp_path)
    monitor = SimpleNamespace(check=lambda: None, sample=lambda: {'rss_bytes': 0})
    engine = HybridEngine(store, SimpleNamespace(), monitor, HybridConfig(execution_mode='gpu'))
    engine.model.to_device('gpu')
    # This is a declared transport state, not a model-generated question or a
    # true human label. The real neural update is confined to this fixture.
    sequence = features('Owned feedback transport fixture')
    engine.state['goals']['fixture'] = {'id': 'fixture', 'status': 'awaiting_human'}
    engine.state['decisions']['fixture'] = {'sequence': sequence.tolist(), 'action': 'ask'}
    engine.state['question'] = {'id': 'question-fixture', 'goal_id': 'fixture',
                                'decision_id': 'fixture', 'question': 'Owned test guidance?',
                                'status': 'pending'}
    initial = engine.save(pinned=True)
    token = secrets.token_hex(16)
    gateway = SimpleNamespace(store=store, engine=engine, token=token)
    # Execute the current source's nested Handler class unchanged. Only its
    # enclosing gateway/port wiring is supplied; all network requests are real.
    tree = ast.parse((ROOT / 'src/scarx/hybrid/gateway.py').read_text(encoding='utf-8'))
    serve = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'serve')
    handler = next(n for n in serve.body if isinstance(n, ast.ClassDef) and n.name == 'Handler')
    namespace = {**vars(gateway_module), 'gateway': gateway, 'port': 0, 'BaseHTTPRequestHandler': BaseHTTPRequestHandler,
                 'encode': encode, 'json': json, 'threading': threading,
                 'urlparse': urlparse, 'parse_qs': parse_qs, 'Path': Path}
    exec(compile(ast.Module(body=[handler], type_ignores=[]), 'actual-gateway-Handler', 'exec'), namespace)
    server = ThreadingHTTPServer(('127.0.0.1', 0), namespace['Handler'])
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    def post(body):
        request = Request(f'http://127.0.0.1:{server.server_port}/api/feedback',
                          data=encode(body).encode(), headers={'Host': '127.0.0.1:0',
                          'X-SCARX-Token': token, 'Content-Type': 'application/json'})
        try:
            with urlopen(request, timeout=5) as response:return response.status, json.load(response)
        except HTTPError as exc:return exc.code, json.load(exc)
    try:
        yield SimpleNamespace(store=store, engine=engine, post=post, initial=initial,
                              Handler=namespace['Handler'],
                              body={'id': 'question-fixture', 'text': 'Owned fixture feedback.',
                                    'helpful': True, 'client_id': 'fixture-send-identity'})
    finally:
        server.shutdown();server.server_close();worker.join(timeout=3);store.close()


def test_actual_http_commit_before_response_cache_loss_is_retryable_without_second_learning(fixture, monkeypatch):
    original = fixture.store.put
    def lost_receipt(kind, *args, **kwargs):
        if kind == 'api_result':
            raise OSError('Controlled loss of old post-commit HTTP cache')
        return original(kind, *args, **kwargs)
    monkeypatch.setattr(fixture.store, 'put', lost_receipt)
    first_status, first = fixture.post(fixture.body)
    assert first_status in (200, 503)
    assert fixture.engine.model.updates == 1
    status, result = fixture.post(fixture.body)
    assert status == 200 and result['accepted'], (first_status, first, status, result)
    assert fixture.engine.model.updates == 1 and fixture.engine.state['question'] is None


def test_same_send_identity_with_changed_reply_is_rejected_not_silently_cached(fixture):
    assert fixture.post(fixture.body)[0] == 200
    status, _ = fixture.post(dict(fixture.body, text='Different feedback, same identity'))
    assert status == 400 and fixture.engine.model.updates == 1


def test_actual_socket_response_loss_after_commit_retries_without_learning_twice(fixture, monkeypatch):
    original=fixture.Handler.send;dropped=[]
    def send(handler, value, status=200, *args, **kwargs):
        if status==200 and isinstance(value,dict) and value.get('accepted') and not dropped:
            dropped.append(True)
            handler.close_connection=True
            handler.connection.shutdown(socket.SHUT_RDWR)
            handler.connection.close()
            return
        return original(handler,value,status,*args,**kwargs)
    monkeypatch.setattr(fixture.Handler,'send',send)
    with pytest.raises((RemoteDisconnected, ConnectionResetError)):
        fixture.post(fixture.body)
    assert dropped and fixture.engine.model.updates==1
    before=fixture.engine.model.dumps(fixture.engine.state)
    status,result=fixture.post(fixture.body)
    assert status==200 and result['accepted']
    assert fixture.engine.model.dumps(fixture.engine.state)==before
    assert len(fixture.store.list('feedback_receipt'))==1


def test_concurrent_duplicate_actual_http_requests_commit_one_update(fixture):
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: fixture.post(fixture.body), range(2)))
    assert all(status == 200 for status, _ in results)
    assert results[0][1] == results[1][1] and fixture.engine.model.updates == 1


def test_receipt_persists_after_engine_restart_and_neural_rollback(fixture):
    status, result = fixture.post(fixture.body)
    assert status == 200
    store = fixture.store
    cold = HybridEngine(store, SimpleNamespace(), fixture.engine.monitor, fixture.engine.config)
    cold.model.to_device('gpu')
    assert cold.feedback_once('fixture-send-identity', 'question-fixture',
                             text=fixture.body['text'], helpful=True) == result
    cold.load(fixture.initial)
    assert cold.model.updates == 0
    assert cold.feedback_once('fixture-send-identity', 'question-fixture',
                             text=fixture.body['text'], helpful=True) == result
    assert cold.model.updates == 0, 'Neural rollback does not refund an already committed API request'


def test_receipt_commit_failure_rolls_back_question_weights_and_all_new_rows(fixture, monkeypatch):
    before = fixture.engine.model.dumps(fixture.engine.state)
    original = fixture.store.put
    def disk_failure(kind, *args, **kwargs):
        if kind == 'feedback_receipt':raise OSError('Controlled receipt write failure')
        return original(kind, *args, **kwargs)
    monkeypatch.setattr(fixture.store, 'put', disk_failure)
    status, _ = fixture.post(fixture.body)
    assert status == 503 and fixture.engine.model.dumps(fixture.engine.state) == before
    assert not fixture.store.search(fixture.body['text'], stream='conversation')
    assert not fixture.store.list('feedback_receipt')
    monkeypatch.setattr(fixture.store, 'put', original)
    assert fixture.post(fixture.body)[0] == 200 and fixture.engine.model.updates == 1


def test_actual_sqlite_abort_returns_retryable_error_and_reverts_all_nested_commits(fixture):
    before = fixture.engine.model.dumps(fixture.engine.state)
    counts = {name: fixture.store.db.execute('SELECT count(*) FROM '+name).fetchone()[0]
              for name in ('objects', 'checkpoints', 'events', 'evidence')}
    fixture.store.db.execute("""CREATE TRIGGER owned_receipt_fault BEFORE INSERT ON objects
        WHEN NEW.kind='feedback_receipt' BEGIN SELECT RAISE(ABORT,'owned SQLite fault'); END""")
    fixture.store.db.commit()
    status, result = fixture.post(fixture.body)
    assert status == 503 and 'owned SQLite fault' in str(result)
    assert fixture.engine.model.dumps(fixture.engine.state) == before
    assert all(fixture.store.db.execute('SELECT count(*) FROM '+n).fetchone()[0] == count
               for n,count in counts.items())
    fixture.store.db.execute('DROP TRIGGER owned_receipt_fault');fixture.store.db.commit()
    assert fixture.post(fixture.body)[0] == 200


def test_relevance_batch_and_user_statement_roll_back_together_on_receipt_fault(fixture, monkeypatch):
    items = [fixture.store.observe(text, source='owned fixture') for text in ('red event','blue event')]
    fixture.store.put('answer', 'answer-fixture', {'question':'Which event?',
                       'candidate_ids':[x['id'] for x in items], 'stream':None})
    body = {'client_id':'batch-fixture-send', 'id':'answer-fixture','text':'Owned batch guidance',
            'positive_ids':[items[0]['id']], 'negative_ids':[items[1]['id']]}
    before = fixture.engine.model.dumps(fixture.engine.state)
    original = fixture.store.put
    def fail(kind, *args, **kwargs):
        if kind=='feedback_receipt':raise OSError('owned batch receipt failure')
        return original(kind,*args,**kwargs)
    monkeypatch.setattr(fixture.store,'put',fail)
    assert fixture.post(body)[0] == 503
    assert fixture.engine.model.dumps(fixture.engine.state) == before
    assert not fixture.store.search(body['text'], stream='conversation')
    monkeypatch.setattr(fixture.store,'put',original)
    status,result=fixture.post(body)
    assert status==200 and len(result['learning'])==2
    updates=fixture.engine.model.updates
    assert fixture.post(body)==(status,result) and fixture.engine.model.updates==updates


def test_feedback_batch_is_bounded_and_duplicate_labels_fail_before_any_update(fixture):
    before = fixture.engine.model.dumps(fixture.engine.state)
    for ids in (['candidate'] * 129, ['candidate','candidate'], 'not-an-ID-list'):
        status,_=fixture.post(dict(fixture.body,positive_ids=ids))
        assert status==400 and fixture.engine.model.dumps(fixture.engine.state)==before
