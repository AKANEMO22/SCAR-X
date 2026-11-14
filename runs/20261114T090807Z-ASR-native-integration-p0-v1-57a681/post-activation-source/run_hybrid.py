"""One entrypoint owns the gateway, VLM, ASR, memory, learner and trace."""

import os

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from scarx.hybrid.gateway import serve
from scarx.runtime.maintenance import check_startup

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "local/hybrid-v1")
    parser.add_argument("--port", type=int, default=8775)
    args = parser.parse_args()
    check_startup(args.data_dir)
    serve(args.data_dir, args.port)
