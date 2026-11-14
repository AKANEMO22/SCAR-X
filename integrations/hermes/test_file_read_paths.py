"""Read failures must survive both compound and sequential shell transport."""

import os
import subprocess
from types import SimpleNamespace
from unittest.mock import MagicMock
import pytest
from tools.file_operations import ShellFileOperations
from tools.file_operations_common import ExecuteResult, ReadResult
from tools.file_operations_read_contract import finish_compound_read


@pytest.mark.parametrize("status", ("0 0 17 0 0", "0 0 0 17 0", "0 0 0 0 17"))
def test_failed_probe_is_not_hidden_by_numeric_zero_metadata(status):
    ops = SimpleNamespace(
        _decode_base64_sample=lambda text: b"", _is_likely_binary=lambda *args: False,
        _is_likely_binary_bytes=lambda data: False,
        _assemble_read_result=lambda *args, **kw: ReadResult(file_size=kw["file_size"]))
    result = finish_compound_read(ops, "owned.txt", 1, 2, ("0", "", "", "0", "0", status))
    assert result.error and not result.not_found


def sequential_ops(size=5, status="ok", count="1", count_code=0, tail_code=0):
    ops = ShellFileOperations(MagicMock())
    ops._probe_regular_file = lambda path: (size, status)
    ops._is_image = lambda path: False
    ops._detect_binary = lambda path: (False, b"text\n")
    ops._sample_file_bytes = lambda path: b"text\n"
    def execute(command):
        if command.startswith("wc -l <"):
            return ExecuteResult(stdout=count, exit_code=count_code)
        if "tail -c 1" in command:
            return ExecuteResult(stdout="1", exit_code=tail_code)
        return ExecuteResult(stdout="text\n", exit_code=0)
    ops._exec = execute
    return ops


@pytest.mark.parametrize("size,status", ((0,"bad_size"),(-1,"ok")))
def test_sequential_unknown_or_negative_size_does_not_manufacture_empty(size,status):
    result = sequential_ops(size,status)._read_file_sequential("owned.txt",1,2)
    assert result.error and not result.not_found


@pytest.mark.parametrize("count,code", (("unavailable",0),("0",17),("-1",0)))
def test_sequential_count_failure_cannot_turn_into_valid_read(count,code):
    result = sequential_ops(count=count,count_code=code)._read_file_sequential("owned.txt",1,2)
    assert result.error and not result.not_found


def test_sequential_failed_tail_does_not_claim_correct_full_line_count():
    result = sequential_ops(tail_code=17)._read_file_sequential("owned.txt",1,2)
    assert result.error and not result.not_found


def real_bash_ops(directory, fail_sed):
    from tools.environments.local import _find_bash
    bash = _find_bash()
    def execute(command, **kwargs):
        script = ("sed() { return 17; }; " if fail_sed else "") + command
        completed = subprocess.run([bash,"-c",script],cwd=directory,capture_output=True,timeout=10)
        return {"output":(completed.stdout+completed.stderr).decode("utf-8","replace"),
                "returncode":completed.returncode}
    ops = ShellFileOperations(SimpleNamespace(cwd=str(directory),is_local=True,execute=execute))
    ops._native_read_enabled = lambda: False
    return ops,bash


@pytest.mark.platforms("windows")
@pytest.mark.parametrize("path", ("compound","sequential"))
def test_real_bash_failed_sed_is_not_hidden_by_successful_cut(tmp_path,path):
    source=tmp_path/"source.txt";source.write_bytes(b"owned\n")
    ops,_=real_bash_ops(tmp_path,True)
    result = (ops.read_file(str(source),limit=2) if path=="compound"
              else ops._read_file_sequential(str(source),1,2))
    assert result.error and not result.not_found


@pytest.mark.platforms("windows")
def test_read_probe_does_not_change_callers_shell_options(tmp_path):
    source=tmp_path/"source.txt";source.write_bytes(b"owned\n")
    ops,bash=real_bash_ops(tmp_path,False)
    script=ops._read_probe_cmd(str(source),1,2,100,"OWNED_READ_PROBE")
    completed=subprocess.run([bash,"-c","set +o pipefail; "+script+"; if [[ -o pipefail ]]; then exit 99; fi"],
                             cwd=tmp_path,capture_output=True,timeout=10)
    assert completed.returncode == 0
