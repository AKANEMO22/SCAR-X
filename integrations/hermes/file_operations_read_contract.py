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
    if sample is not None:
        if not sample and file_size > 0:
            return None, None, ReadResult(error="File read sample was empty for a nonempty source; source may be inaccessible or changing.")
        return ops._is_likely_binary_bytes(sample), sample, None
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
        sample_rc, read_rc = int(status[0]), int(status[1])
    except (IndexError, ValueError):
        return ops._read_file_sequential(path, offset, end_line - offset + 1)
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
