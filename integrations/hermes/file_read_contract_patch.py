"""Exact-version fail-closed Hermes read repair, delegated to a small sibling."""

import hashlib

SOURCE_SHA256 = "d33e0bc43516f9c5da4dd1ab75bd57b03837034c5c4accb91df1d78b92131b51"


def patch_bytes(original):
    if hashlib.sha256(original).hexdigest() != SOURCE_SHA256:
        raise ValueError("Hermes source version changed; review before applying reader patch")
    eol = b"\r\n" if b"\r\n" in original else b"\n"
    text = original.decode("utf-8").replace("\r\n", "\n")
    start_marker = "        size_seg, sample_seg, page_seg, wc_seg, tail_seg, status_seg = segments\n"
    end_marker = "    def _native_read_enabled(self) -> bool:"
    if text.count(start_marker) != 1 or text.count(end_marker) != 1:
        raise ValueError("Expected one reviewed compound-reader implementation")
    start = text.index(start_marker)
    stop = text.index(end_marker, start)
    replacement = ("        from tools.file_operations_read_contract import finish_compound_read\n"
                   "        return finish_compound_read(self, path, offset, end_line, segments)\n\n")
    changed = (text[:start] + replacement + text[stop:]).encode().replace(b"\n", eol)
    compile(changed, "reviewed_hermes_file_operations.py", "exec")
    return changed
