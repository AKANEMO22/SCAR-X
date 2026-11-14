"""Exact-source upgrade from the installed compound-only maintenance repair."""

import hashlib

SOURCE_SHA256 = "68bb3ef6fb07c12384641f1359a7744dbf4814e1d84a79b7b20407d2bad3dee9"
SIBLING_BEFORE_SHA256 = "e846b861d9a2f672f47f026b72aab2b4799109b93cef55f9720309192add5a2f"


def patch_bytes(original):
    if hashlib.sha256(original).hexdigest() != SOURCE_SHA256:
        raise ValueError("Hermes reader source changed; review before whole-path upgrade")
    eol = b"\r\n" if b"\r\n" in original else b"\n"
    text = original.decode("utf-8").replace("\r\n", "\n")
    start = text.index("        arg = self._escape_shell_arg(path)\n", text.index("    def _read_probe_cmd("))
    stop = text.index("    def _read_file_missing(", start)
    replacement = ("        from tools.file_operations_read_contract import compound_read_command\n"
        "        return compound_read_command(self, path, offset, end_line, line_clamp_bytes,\n"
        "                                     sentinel, MISSING_SENTINEL, NOT_REGULAR_SENTINEL)\n\n")
    text = text[:start] + replacement + text[stop:]
    start = text.index("        file_size, status = self._probe_regular_file(path)\n", text.index("    def _read_file_sequential("))
    stop = text.index("    def _assemble_read_result(", start)
    text = text[:start] + ("        from tools.file_operations_read_contract import read_sequential\n"
                          "        return read_sequential(self, path, offset, limit)\n\n") + text[stop:]
    result = text.encode().replace(b"\n", eol)
    compile(result, "hermes_file_operations_read_paths_v2.py", "exec")
    return result
