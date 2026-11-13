"""Start the passive hybrid implementation observer, never a training run."""

from pathlib import Path
import sys
import os

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from scarx.hybrid.store import Store
from scarx.hybrid.observer import serve

if __name__ == "__main__":
    directory = ROOT / "local/hybrid-v1"
    store = Store(directory)
    store.event(
        "implementation_started",
        {
            "branch": "codex/scarx-hybrid-20261113",
            "base": "37749041ef45d54951558fdeb0c3695616b47cb5",
            "ponytail": "not available in tool inventory",
            "protected_peer": "active scar-x-gemini-r7 and its processes left untouched",
            "phase": "Implementation and setup; models not loaded; no quality measurements",
        },
        "orchestrator",
    )
    store.close()
    serve(directory)
