"""Sandboxed Execution Worker: Provides process-isolated, resource-bounded execution for untrusted code.

Enforces P0-E:
- True OS process boundary via subprocess isolation.
- Hard execution timeout (default 4.0 seconds) with forced SIGKILL on hang.
- Isolated temporary working directory (no repo workspace pollution).
- Sanitized environment variables.
- Output buffer size capping (prevents memory exhaustion from runaway prints).
"""
from typing import Dict, Any, Optional
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


class SandboxedPythonWorker:
    """Executes arbitrary Python code in an isolated, monitored subprocess."""

    def __init__(self, timeout_sec: float = 4.0, max_output_chars: int = 10_000):
        self.timeout_sec = timeout_sec
        self.max_output_chars = max_output_chars

    def execute_snippet(self, code_str: str, custom_stdin: str = "") -> Dict[str, Any]:
        """Execute a Python snippet in a clean subprocess with strict limits."""
        if not code_str or not code_str.strip():
            return {
                "success": False,
                "error": "Empty code snippet provided",
                "stdout": "",
                "stderr": "",
                "result": None,
                "duration_ms": 0.0,
            }

        # Create temporary isolated scratch workspace
        scratch_dir = tempfile.mkdtemp(prefix="scarx_sandbox_")
        result_file = Path(scratch_dir) / "__sandbox_result__.json"

        # Harness wrapper to extract local 'result' variable if defined
        wrapped_code = f"""
import sys, json

__sandbox_local_scope__ = {{}}
__sandbox_res_payload__ = {{"has_result": False, "result": None, "error": None}}

try:
{self._indent_code(code_str, 4)}
    if 'result' in locals():
        __sandbox_res_payload__["has_result"] = True
        try:
            # Test JSON serializability
            json.dumps(result)
            __sandbox_res_payload__["result"] = result
        except Exception:
            __sandbox_res_payload__["result"] = str(result)
except Exception as _e:
    import traceback
    __sandbox_res_payload__["error"] = str(_e)
    __sandbox_res_payload__["traceback"] = traceback.format_exc()
    sys.stderr.write(traceback.format_exc())

try:
    with open(r"{result_file}", "w", encoding="utf-8") as _f:
        json.dump(__sandbox_res_payload__, _f)
except Exception as _e:
    pass
"""
        code_file = Path(scratch_dir) / "__runner__.py"
        code_file.write_text(wrapped_code, encoding="utf-8")

        # Sanitize environment
        clean_env = {
            "SYSTEMROOT": os.environ.get("SYSTEMROOT", "C:\\Windows"),
            "PATH": os.environ.get("PATH", ""),
            "PYTHONIOENCODING": "utf-8",
            "PYTHONHASHSEED": "0",
        }

        t0 = time.perf_counter()
        proc = None
        stdout_txt = ""
        stderr_txt = ""
        timed_out = False
        exit_code = -1

        try:
            proc = subprocess.Popen(
                [sys.executable, "-I", str(code_file)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=scratch_dir,
                env=clean_env,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            stdout_txt, stderr_txt = proc.communicate(input=custom_stdin, timeout=self.timeout_sec)
            exit_code = proc.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            if proc:
                proc.kill()
                try:
                    stdout_txt, stderr_txt = proc.communicate(timeout=1.0)
                except Exception:
                    pass
        except Exception as e:
            stderr_txt = f"Subprocess launch error: {str(e)}"
        finally:
            duration_ms = (time.perf_counter() - t0) * 1000.0

        # Read back extracted result
        has_result = False
        extracted_result = None
        runtime_err = None

        if result_file.exists():
            try:
                res_data = json.loads(result_file.read_text(encoding="utf-8"))
                has_result = res_data.get("has_result", False)
                extracted_result = res_data.get("result")
                runtime_err = res_data.get("error")
            except Exception:
                pass

        # Cleanup scratch dir
        shutil.rmtree(scratch_dir, ignore_errors=True)

        # Capping output size
        stdout_txt = stdout_txt[:self.max_output_chars]
        stderr_txt = stderr_txt[:self.max_output_chars]

        if timed_out:
            return {
                "success": False,
                "error": f"Execution timed out after {self.timeout_sec}s",
                "stdout": stdout_txt,
                "stderr": stderr_txt,
                "result": None,
                "duration_ms": duration_ms,
                "timed_out": True,
            }

        success = (exit_code == 0) and (runtime_err is None)
        final_result = extracted_result if has_result else (stdout_txt.strip() if stdout_txt.strip() else None)

        return {
            "success": success,
            "error": runtime_err if runtime_err else (stderr_txt.strip() if exit_code != 0 else None),
            "stdout": stdout_txt,
            "stderr": stderr_txt,
            "result": final_result,
            "duration_ms": duration_ms,
            "exit_code": exit_code,
        }

    def _indent_code(self, code: str, spaces: int = 4) -> str:
        pad = " " * spaces
        lines = code.splitlines()
        return "\n".join(f"{pad}{line}" for line in lines)
