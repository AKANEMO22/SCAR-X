"""Pytest overlay loader for the owned Hermes environment; no live file edits."""

import importlib.util
import os
from pathlib import Path
import sys

candidate = os.environ.get("SCARX_HERMES_FILE_READ_CANDIDATE")
if candidate:
    path = Path(candidate).resolve()
    root = Path(__file__).resolve().parents[2]
    if not path.is_relative_to(root / "local"):
        raise ValueError("Expected owned private module overlay")
    spec = importlib.util.spec_from_file_location("tools.file_operations", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["tools.file_operations"] = module
    import tools
    tools.file_operations = module
    spec.loader.exec_module(module)
