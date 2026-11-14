"""Explicit test-only overlay for canonical Hermes runs; no live source mutation."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

config_path = Path(__file__).with_suffix(".json")
if not config_path.is_file():
    raise RuntimeError("Copy this test loader with its owned overlay manifest before use")
config = json.loads(config_path.read_text(encoding="utf-8"))
local_root = config_path.parent.parent.resolve()
import tools
for name in ("file_operations_read_contract", "file_operations"):
    entry = config[name]
    path = Path(entry["path"]).resolve()
    if not path.is_relative_to(local_root):
        raise ValueError("Expected owned private module overlay")
    if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
        raise ValueError("Test overlay source changed")
    spec = importlib.util.spec_from_file_location("tools." + name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["tools." + name] = module
    setattr(tools, name, module)
    spec.loader.exec_module(module)
