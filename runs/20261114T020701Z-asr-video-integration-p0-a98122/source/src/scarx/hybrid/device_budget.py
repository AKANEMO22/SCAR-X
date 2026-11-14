"""Physical Nvidia headroom for component scheduling, separate from host RAM."""

import subprocess

GPU_RESERVE_MIB = 512
ASR_WORKSPACE_MIB = 538


def gpu_headroom(device=0):
    result = subprocess.run(
        [
            "nvidia-smi",
            "--id=" + str(device),
            "--query-gpu=memory.free,memory.total",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=3,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        check=True,
    )
    rows = result.stdout.strip().splitlines()
    if len(rows) != 1:
        raise RuntimeError("Ambiguous Nvidia memory budget")
    free, total = [int(v.strip()) for v in rows[0].split(",")]
    if not 0 <= free <= total:
        raise RuntimeError("Invalid Nvidia memory reading")
    return {
        "free_MiB": free,
        "total_MiB": total,
        "reserve_MiB": GPU_RESERVE_MIB,
        "source": "nvidia-smi physical dedicatedGPU availability",
    }


def audio_requires_release(headroom):
    return headroom["free_MiB"] < GPU_RESERVE_MIB + ASR_WORKSPACE_MIB
