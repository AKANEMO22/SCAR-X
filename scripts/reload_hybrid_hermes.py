"""Restart only the default Hermes gateway through its own graceful replace API."""

from pathlib import Path
import json
import os
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    directory = ROOT / "local/hybrid-v1"
    home = Path(os.environ["LOCALAPPDATA"]) / "hermes"
    binary = home / "bin/hermes.exe"
    if not binary.exists():
        raise FileNotFoundError("Hermes launcher missing")
    with (directory / "hermes-gateway-reload.log").open("ab") as log:
        process = subprocess.Popen(
            [
                str(binary),
                "--profile",
                "default",
                "gateway",
                "run",
                "--quiet",
                "--replace",
            ],
            stdout=log,
            stderr=log,
            cwd=home / "hermes-agent",
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    print(
        json.dumps(
            {
                "replacement_launcher_pid": process.pid,
                "scope": "default Hermes gateway only",
                "method": "Hermes built-in graceful --replace",
                "peer_experiments_untouched": True,
            }
        )
    )
