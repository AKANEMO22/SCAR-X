"""Scoring, admission and real SSE parser invariants; no inference here."""
import importlib.util
import io
from pathlib import Path
import sys

path = Path(__file__).resolve().parents[1] / "scripts/hybrid_context_benchmark.py"
spec = importlib.util.spec_from_file_location("hybrid_context_benchmark", path)
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)


def test_score_requires_exact_keys_values_and_types():
    assert b.score('{"port":9237,"days":6}', {"days":6,"port":9237})
    assert not b.score('{"port":"9237","days":6}', {"port":9237,"days":6})
    assert not b.score('{"port":9237,"days":6,"other":0}', {"port":9237,"days":6})
    assert not b.score('```json\n{"days":6}\n```', {"days":6})
    assert not b.score('{"n":true}', {"n":1})


def test_input_admission_reserves_generation():
    assert b.fits(12000, 16384)
    assert not b.fits(22000, 16384)
    assert b.fits(22000, 24576)
    assert not b.fits(16384 - b.OUT, 16384)


def test_stream_does_not_count_role_empty_delta_as_first_content():
    data = b'\n'.join([
        b'data: {"choices":[{"delta":{"role":"assistant"}}]}',
        b'data: {"choices":[{"delta":{"content":"{\\"n\\":1}"}}]}',
        b'data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"completion_tokens":5},"timings":{"prompt_n":10}}',
        b'data: [DONE]'])
    result = b.consume(io.BytesIO(data))
    assert result["answer"] == '{"n":1}'
    assert result["first_after_headers_seconds"] is not None
    assert result["finish_reason"] == "stop"
    assert result["usage"]["completion_tokens"] == 5
    assert result["timings"]["prompt_n"] == 10


def test_empty_stream_never_claims_successful_answer():
    result = b.consume(io.BytesIO(b'data: [DONE]\n'))
    assert result["first_after_headers_seconds"] is None
    assert result["finish_reason"] is None
    assert not b.score(result["answer"], {"n":1})


def test_independent_audit_rejects_duplicate_key_and_boolean_as_number():
    audit_path = path.parent / "audit_hybrid_context_benchmark.py"
    audit_spec = importlib.util.spec_from_file_location("context_audit", audit_path)
    audit = importlib.util.module_from_spec(audit_spec)
    audit_spec.loader.exec_module(audit)
    assert audit.correct('{"n":1}', {"n":1})
    assert not audit.correct('{"n":0,"n":1}', {"n":1})
    assert not audit.correct('{"n":true}', {"n":1})
    assert audit.outcome({"status":"COMPLETED", "records":[{"status":"TIMEOUT"}]}) == "TIMEOUT"
    assert audit.outcome({"status":"RAM_FAIL", "records":[{"status":"TIMEOUT"}]}) == "RAM_FAIL"
