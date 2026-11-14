"""Engineering regressions; separate from the recorded real-inference pilot."""

import base64
import importlib.util
from pathlib import Path
import threading
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from types import SimpleNamespace
import pytest
from PIL import Image
from scarx.hybrid.store import Store
from scarx.hybrid.screen import ScreenMemory
from scarx.hybrid.media import VideoMemory
from scarx.hybrid.gateway import Gateway
from scarx.hybrid.video_context import VideoContext


def test_word_error_accounts_for_insertions_deletions_and_case():
    p = Path(__file__).resolve().parents[1] / "scripts/hybrid_pipeline_audit.py"
    spec = importlib.util.spec_from_file_location("pipeline_audit", p)
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    assert audit.word_error("Xin CHÀO", "xin chào")["wer"] == 0
    assert audit.word_error("a c d", "a b c")["word_edits"] == 2
    assert audit.word_error("", "a b")["wer"] == 1


def test_vision_continues_after_non_resource_asr_error(tmp_path, monkeypatch):
    store = Store(tmp_path)
    called = []

    def asr(_):
        raise RuntimeError("controlled ASR codec failure, not OOM")

    engine = SimpleNamespace(
        store=store,
        foreground=threading.Event(),
        monitor=SimpleNamespace(check=lambda: None),
        backend=SimpleNamespace(
            transcribe=asr,
            describe=lambda *a, **k: called.append(True) or {"text": "fixture"},
        ),
        observe=lambda *a, **k: None,
    )
    memory = VideoMemory(engine)
    monkeypatch.setattr(
        "scarx.hybrid.media.subprocess.run",
        lambda *a, **k: SimpleNamespace(returncode=0),
    )
    try:
        memory._analyze(
            {
                "id": "a" * 32,
                "public_topic_id": None,
                "capture_frontier": 10,
                "asr_frontier": 0,
                "vision_frontier": 0,
                "skipped_intervals": [],
                "analyzed_intervals": [],
            },
            tmp_path / "segment.mkv",
            0,
            10,
        )
        assert called
    finally:
        store.close()


def test_first_screen_question_uses_first_frame(tmp_path):
    store = Store(tmp_path)
    sid = "b" * 32
    directory = tmp_path / "archive" / sid
    directory.mkdir(parents=True)
    first, last = directory / "first.jpg", directory / "last.jpg"
    Image.new("RGB", (20, 20), "blue").save(first)
    Image.new("RGB", (20, 20), "yellow").save(last)
    for time, path, color in ((0, first, "blue"), (10, last, "yellow")):
        store.observe(
            color + " circle",
            stream=sid,
            start=time,
            end=time,
            source=str(path),
            modality="sampled_frame",
        )
    store.put(
        "stream",
        sid,
        {
            "id": sid,
            "kind": "screen",
            "latest_frame": str(last),
            "status": "stopped",
            "capture_frontier": 10,
            "asr_frontier": 0,
            "vision_frontier": 10,
        },
    )
    engine = SimpleNamespace(
        store=store,
        model=SimpleNamespace(relevance=lambda *args: 0.0),
        rank_candidates=lambda query, items: items[:4],
    )
    gw = Gateway.__new__(Gateway)
    gw.store, gw.engine = store, engine
    gw.screen = ScreenMemory(engine)
    gw.video_context = VideoContext(store)
    gw.active_screen_streams = {}
    try:
        result = gw.prepare_agent_messages(
            {
                "scarx_stream_id": sid,
                "messages": [
                    {
                        "role": "user",
                        "content": "At the start, what color was the circle?",
                    }
                ],
            }
        )
        imagepart = next(p for p in result[-1]["content"] if p["type"] == "image_url")
        import io

        with Image.open(
            io.BytesIO(base64.b64decode(imagepart["image_url"]["url"].split(",", 1)[1]))
        ) as image:
            red, green, blue = image.getpixel((10, 10))
        assert blue > 200 and red < 30 and green < 30
    finally:
        store.close()


@pytest.mark.parametrize(
    "prompt,expected",
    [
        ("At the start, what color is the circle?", 0),
        ("At the end, what color is the circle?", 10),
        ("Màu hình tròn ở cuối video?", 10),
    ],
)
def test_explicit_time_constraint_cannot_be_overridden_by_relevance(
    tmp_path, prompt, expected
):
    store = Store(tmp_path)
    sid = "c" * 32
    store.put(
        "stream",
        sid,
        {"id": sid, "capture_frontier": 10, "asr_frontier": 0, "vision_frontier": 10},
    )
    for at, color in ((0, "blue"), (10, "yellow")):
        store.observe(
            color + " circle",
            stream=sid,
            start=at,
            end=at,
            source="fixture",
            modality="sampled_frame",
        )
    gw = Gateway.__new__(Gateway)
    gw.store = store
    gw.engine = SimpleNamespace(
        model=SimpleNamespace(relevance=lambda query, e: 100 - e["start"]),
        rank_candidates=lambda query, items: sorted(
            items, key=lambda e: 100 - e["start"], reverse=True
        )[:4],
    )
    gw.video = SimpleNamespace(snapshot=lambda *a: None)
    gw.video_context = VideoContext(store)
    gw.active_screen_streams = {}
    body = {"scarx_stream_id": sid, "messages": [{"role": "user", "content": prompt}]}
    try:
        gw.prepare_agent_messages(body)
        assert len(body["_scarx_evidence"]) == 1
        assert body["_scarx_evidence"][0]["start"] == expected
    finally:
        store.close()


def test_both_temporal_endpoints_do_not_collapse_to_one():
    from scarx.hybrid.temporal import temporal_direction

    assert temporal_direction("Compare first and last events") is None
    assert temporal_direction("backend function") is None
    assert temporal_direction("Ở đầu video") == "first"


def test_asr_resource_breach_still_aborts_instead_of_hiding_it(tmp_path, monkeypatch):
    store = Store(tmp_path)

    def asr(_):
        raise MemoryError("fixture cap exceeded")

    engine = SimpleNamespace(
        store=store,
        foreground=threading.Event(),
        monitor=SimpleNamespace(check=lambda: None),
        backend=SimpleNamespace(transcribe=asr),
    )
    memory = VideoMemory(engine)
    monkeypatch.setattr(
        "scarx.hybrid.media.subprocess.run",
        lambda *a, **k: SimpleNamespace(returncode=0),
    )
    try:
        with pytest.raises(MemoryError):
            memory._analyze({"id": "d" * 32}, tmp_path / "a.mkv", 0, 10)
        assert not store.events()
    finally:
        store.close()


def test_missing_historical_frame_does_not_fallback_to_current_frame(tmp_path):
    store = Store(tmp_path)
    sid = "e" * 32
    root = tmp_path / "archive" / sid
    root.mkdir(parents=True)
    Image.new("RGB", (20, 20), "yellow").save(root / "last.jpg")
    store.put(
        "stream",
        sid,
        {"id": sid, "kind": "screen", "latest_frame": str(root / "last.jpg")},
    )
    screen = ScreenMemory(SimpleNamespace(store=store))
    try:
        assert (
            screen.snapshot(
                sid,
                [
                    {
                        "id": "missing",
                        "start": 0,
                        "source": str(root / "first.jpg"),
                        "modality": "sampled_frame",
                    }
                ],
            )
            is None
        )
        assert store.events()[-1]["kind"] == "historical_screen_frame_unavailable"
    finally:
        store.close()


def test_nonce_rejection_consumes_only_bounded_request_body_before_response():
    import io
    p = Path(__file__).resolve().parents[1] / "scripts/hybrid_pipeline_audit.py"
    spec = importlib.util.spec_from_file_location("nonce_body_fixture", p)
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    calls, responses, timeouts = [], [], []

    class Base:
        def do_POST(self):
            calls.append("executed")

        def send(self, body, status):
            responses.append(status)

    wrapped = audit.isolated_handler(Base, "owned")
    for length, expected_consumed in (2, 2), (100_000, 0):
        handler = wrapped()
        handler.headers = {"Content-Length": str(length)}
        handler.rfile = io.BytesIO(b"{}")
        handler.connection = SimpleNamespace(
            gettimeout=lambda: None, settimeout=timeouts.append)
        handler.do_POST()
        assert handler.rfile.tell() == expected_consumed
        assert handler.close_connection is True
    assert calls == [] and responses == [503, 503]
    assert timeouts[-1] is None


def test_isolated_http_rejects_every_foreign_mutation_without_executing_it():
    p = Path(__file__).resolve().parents[1] / "scripts/hybrid_pipeline_audit.py"
    spec = importlib.util.spec_from_file_location("pipeline_audit_nonce", p)
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    called = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send(self, body, status=200):
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            called.append(self.path)
            self.send({"accepted": True})

    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), audit.isolated_handler(Handler, "fixture-audit-only")
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = "http://127.0.0.1:" + str(server.server_port)
        for route in (
            "/api/presence",
            "/api/shutdown",
            "/v1/chat/completions",
            "/api/observe",
        ):
            with pytest.raises(HTTPError) as error:
                urlopen(Request(base + route, data=b"{}"), timeout=2)
            assert error.value.code == 503
            error.value.close()
        assert called == []
        with urlopen(
            Request(
                base + "/api/presence",
                data=b"{}",
                headers={"X-SCARX-Audit": "fixture-audit-only"},
            ),
            timeout=2,
        ) as response:
            assert response.status == 200
        assert called == ["/api/presence"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)
