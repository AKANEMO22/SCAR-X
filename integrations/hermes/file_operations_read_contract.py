"""Interpret one bounded compound read; unavailable metadata is not an empty file."""

from tools.file_operations_common import ReadResult, _strip_terminal_fence_leaks


def _nonnegative_count(segment, name):
    try:
        value = int(_strip_terminal_fence_leaks(segment).strip())
    except ValueError:
        return None, ReadResult(error=f"File read {name} metadata unavailable; source may be inaccessible or changing. No content was established.")
    if value < 0:
        return None, ReadResult(error=f"Invalid negative file {name} metadata; no content was established.")
    return value, None


def _read_sample(ops, path, sample_segment, sample_status, file_size):
    sample = ops._decode_base64_sample(sample_segment) if sample_status == 0 else None
    return _classify_sample(ops, path, sample, file_size)


def _classify_sample(ops, path, sample, file_size):
    if sample is not None:
        if not sample and file_size > 0:
            return None, None, ReadResult(error="File read sample was empty for a nonempty source; source may be inaccessible or changing.")
        return ops._is_likely_binary(path) or ops._is_likely_binary_bytes(sample), sample, None
    result = ops._head(path, 1000)
    if result.exit_code != 0:
        return None, None, ReadResult(error="File read fallback sample failed; no content was established.")
    content = _strip_terminal_fence_leaks(result.stdout)
    if not content and file_size > 0:
        return None, None, ReadResult(error="File read fallback sample was empty for a nonempty source; no content was established.")
    return ops._is_likely_binary(path, content), None, None


def finish_compound_read(ops, path, offset, end_line, segments):
    """Called after the facade has validated sentinel framing and segment count."""
    size_seg, sample_seg, page_seg, wc_seg, tail_seg, status_seg = segments
    status = _strip_terminal_fence_leaks(status_seg).split()
    try:
        if len(status) not in (2, 5):
            raise ValueError("Unrecognized read probe status frame")
        sample_rc, read_rc = int(status[0]), int(status[1])
        extra = [int(value) for value in status[2:]]
    except (IndexError, ValueError):
        return ops._read_file_sequential(path, offset, end_line - offset + 1)
    for name, code in zip(("byte-size", "line-count", "last-byte"), extra):
        if code != 0:
            return ReadResult(error=f"File read {name} probe failed; source or backend may be unavailable. No complete read was established.")
    size, error = _nonnegative_count(size_seg, "byte-size")
    if error:
        return error
    binary, sample, error = _read_sample(ops, path, sample_seg, sample_rc, size)
    if error:
        return error
    if binary:
        return ops._read_binary_file(path, offset, end_line - offset + 1, size, sample)
    if read_rc != 0:
        return ReadResult(error=f"Failed to read file: {_strip_terminal_fence_leaks(page_seg)}")
    content = _strip_terminal_fence_leaks(page_seg)
    lines, error = _nonnegative_count(wc_seg, "line-count")
    if error:
        return error
    tail = _strip_terminal_fence_leaks(tail_seg).strip()
    newline = tail == "1" if tail in ("0", "1") else None
    if size == 0 and (content or lines):
        return ReadResult(error="Inconsistent zero-size file read metadata; source may have changed during the read.")
    return ops._assemble_read_result(content, offset=offset, end_line=end_line,
                                    total_lines=lines, file_size=size, file_ends_with_newline=newline)


def checked_pipeline(command):
    """Confine shell options to this read; GitBash preserves leading failures."""
    return "(if (set -o pipefail) 2>/dev/null; then set -o pipefail; fi; " + command + ")"


def compound_read_command(ops, path, offset, end_line, clamp, sentinel, missing, nonregular):
    arg = ops._escape_shell_arg(path)
    mark = f"echo {sentinel}"
    return checked_pipeline(
        f"if [ -f {arg} ]; then "
        f"wc -c < {arg} 2>/dev/null; __hz=$?; {mark}; "
        f"head -c 1000 {arg} 2>/dev/null | base64 2>/dev/null; __hs=$?; {mark}; "
        f"sed -n '{offset},{end_line}p' {arg} 2>/dev/null | cut -b1-{clamp} 2>/dev/null; __hr=$?; {mark}; "
        f"wc -l < {arg} 2>/dev/null; __hc=$?; {mark}; "
        f"tail -c 1 {arg} 2>/dev/null | wc -l; __ht=$?; {mark}; "
        f'echo "$__hs $__hr $__hz $__hc $__ht"; '
        f"elif [ -e {arg} ] || [ -L {arg} ]; then echo {nonregular}; "
        f"else echo {missing}; fi")


def read_sequential(ops, path, offset, limit):
    """Same file/query/limits, with independent failure checks at every stage."""
    size, status = ops._probe_regular_file(path)
    if status == "missing":
        return ops._read_file_missing(path, offset, limit)
    if status == "not_regular":
        return ops._not_regular_error(path)
    if status == "bad_size":
        return ReadResult(error="File read byte-size metadata unavailable; no content was established.")
    if status == "env_unavailable":
        return ReadResult(error="File read probe failed; source or backend may be unavailable. No content was established.")
    if status != "ok":
        return ops._env_unavailable_error(path, status)
    size, error = _nonnegative_count(str(size), "byte-size")
    if error:
        return error
    if ops._is_image(path):
        return ops._image_redirect_result(size)
    binary, sample, error = _classify_sample(ops, path, ops._sample_file_bytes(path), size)
    if error:
        return error
    if binary:
        return ops._read_binary_file(path, offset, limit, size, sample)
    from tools.tool_output_limits import get_max_line_length
    clamp, end = 4 * get_max_line_length() + 1, offset + limit - 1
    arg = ops._escape_shell_arg(path)
    page = ops._exec(checked_pipeline(f"sed -n '{offset},{end}p' {arg} | cut -b1-{clamp}"))
    if page.exit_code != 0:
        return ReadResult(error="File page read failed; no complete read was established.")
    content = _strip_terminal_fence_leaks(page.stdout)
    count = ops._exec(f"wc -l < {arg}")
    if count.exit_code != 0:
        return ReadResult(error="File read line-count probe failed; no complete read was established.")
    lines, error = _nonnegative_count(count.stdout, "line-count")
    if error:
        return error
    newline = None
    if lines <= end and content.endswith("\n"):
        tail = ops._exec(checked_pipeline(f"tail -c 1 {arg} | wc -l"))
        flag = _strip_terminal_fence_leaks(tail.stdout).strip()
        if tail.exit_code != 0 or flag not in ("0", "1"):
            return ReadResult(error="File read last-byte probe failed; no complete read was established.")
        newline = flag == "1"
    if size == 0 and (content or lines):
        return ReadResult(error="Inconsistent zero-size file read metadata; source may have changed.")
    return ops._assemble_read_result(content, offset=offset, end_line=end,
                                    total_lines=lines, file_size=size, file_ends_with_newline=newline)
