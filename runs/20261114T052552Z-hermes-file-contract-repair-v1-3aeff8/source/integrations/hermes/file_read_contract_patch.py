"""Exact-version fail-closed Hermes read repair; no neural or permission changes."""

import hashlib

SOURCE_SHA256 = "d33e0bc43516f9c5da4dd1ab75bd57b03837034c5c4accb91df1d78b92131b51"

REPLACEMENTS = (
    ('''        try:
            file_size = int(_strip_terminal_fence_leaks(size_seg).strip())
        except ValueError:
            file_size = 0
''', '''        try:
            file_size = int(_strip_terminal_fence_leaks(size_seg).strip())
        except ValueError:
            return ReadResult(error="File read byte-size metadata unavailable; source may be inaccessible or changing. No content was established.")
        if file_size < 0:
            return ReadResult(error="Invalid negative file byte-size metadata; no content was established.")
'''),
    ('''        if sample_bytes is not None:
            is_binary = self._is_likely_binary_bytes(sample_bytes)
        else:
''', '''        if sample_bytes is not None:
            if not sample_bytes and file_size > 0:
                return ReadResult(error="File read sample was empty for a nonempty source; source may be inaccessible or changing.")
            is_binary = self._is_likely_binary_bytes(sample_bytes)
        else:
'''),
    ('''            sample_output = _strip_terminal_fence_leaks(self._head(path, 1000).stdout)
            is_binary = self._is_likely_binary(path, sample_output)
''', '''            sample_result = self._head(path, 1000)
            if sample_result.exit_code != 0:
                return ReadResult(error="File read fallback sample failed; no content was established.")
            sample_output = _strip_terminal_fence_leaks(sample_result.stdout)
            if not sample_output and file_size > 0:
                return ReadResult(error="File read fallback sample was empty for a nonempty source; no content was established.")
            is_binary = self._is_likely_binary(path, sample_output)
'''),
    ('''        try:
            total_lines = int(_strip_terminal_fence_leaks(wc_seg).strip())
        except ValueError:
            total_lines = 0
''', '''        try:
            total_lines = int(_strip_terminal_fence_leaks(wc_seg).strip())
        except ValueError:
            return ReadResult(error="File read line-count metadata unavailable; source may be inaccessible or changing. No complete read was established.")
        if total_lines < 0:
            return ReadResult(error="Invalid negative file line-count metadata; no complete read was established.")
'''),
    ('''        tail_flag = _strip_terminal_fence_leaks(tail_seg).strip()
        file_ends_with_newline = tail_flag == "1" if tail_flag in ("0", "1") else None
        return self._assemble_read_result(
''', '''        tail_flag = _strip_terminal_fence_leaks(tail_seg).strip()
        file_ends_with_newline = tail_flag == "1" if tail_flag in ("0", "1") else None
        if file_size == 0 and (read_output or total_lines):
            return ReadResult(error="Inconsistent zero-size file read metadata; source may have changed during the read.")
        return self._assemble_read_result(
'''),
)


def patch_bytes(original):
    if hashlib.sha256(original).hexdigest() != SOURCE_SHA256:
        raise ValueError("Hermes source version changed; review before applying reader patch")
    eol = b"\r\n" if b"\r\n" in original else b"\n"
    result = original
    for old, new in REPLACEMENTS:
        before, after = old.encode().replace(b"\n", eol), new.encode().replace(b"\n", eol)
        if result.count(before) != 1:
            raise ValueError("Expected exactly one reviewed read-contract anchor")
        result = result.replace(before, after, 1)
    compile(result, "reviewed_hermes_file_operations.py", "exec")
    return result
