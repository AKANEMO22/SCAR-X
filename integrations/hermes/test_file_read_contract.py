"""Malformed read transport must not manufacture a successful empty-file result."""

import base64
import pytest
from unittest.mock import MagicMock
from tools.file_operations import ShellFileOperations
from tools.file_operations_common import ExecuteResult


def make_ops(size="5", count="1", sample=b"text\n", content="text\n", sample_rc=0, fallback_code=0):
    segments = (size + "\n", base64.b64encode(sample).decode() + "\n", content,
                count + "\n", "1\n", f"{sample_rc} 0\n")
    response = {}
    def command(path, offset, end_line, clamp, sentinel):
        response["output"] = (sentinel + "\n").join(segments)
        return sentinel
    ops = ShellFileOperations(MagicMock())
    ops._native_read_enabled = lambda: False
    ops._read_probe_cmd = command
    ops._exec = lambda command: ExecuteResult(stdout=response["output"], exit_code=0)
    ops._head = lambda *args: ExecuteResult(stdout="text\n", exit_code=fallback_code)
    return ops


@pytest.mark.parametrize("value", ("", "unavailable", "-1"))
def test_unknown_or_negative_file_size_is_not_an_empty_file(value):
    result = make_ops(size=value).read_file("owned.txt")
    assert result.error and "size" in result.error
    assert not result.not_found


@pytest.mark.parametrize("value", ("", "unavailable", "-1"))
def test_unknown_or_negative_line_count_is_not_a_valid_read(value):
    result = make_ops(count=value).read_file("owned.txt")
    assert result.error and "line" in result.error
    assert not result.not_found


def test_failed_sample_does_not_allow_success_via_downstream_pipeline_status():
    result = make_ops(sample_rc=1, fallback_code=1).read_file("owned.txt")
    assert result.error and "sample" in result.error


def test_empty_sample_for_nonempty_file_is_not_a_successful_read():
    result = make_ops(sample=b"").read_file("owned.txt")
    assert result.error and "sample" in result.error


def test_true_empty_file_still_succeeds_and_does_not_become_missing():
    result = make_ops(size="0", count="0", sample=b"", content="").read_file("owned.txt")
    assert not result.error and not result.not_found
    assert result.file_size == 0 and result.content == ""


def test_consistent_nonempty_file_still_has_readable_numbered_content():
    result = make_ops().read_file("owned.txt")
    assert not result.error and result.content == "1|text"


def test_zero_size_with_returned_content_is_inconclusive_not_empty():
    result = make_ops(size="0").read_file("owned.txt")
    assert result.error and "Inconsistent" in result.error
