"""Matched H128 forward latency only; no teaching, quality or retention claim."""

import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time

os.environ["OPENBLAS_NUM_THREADS"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from scarx.hybrid.learner import LocalALIF, features


def main(path):
    if path.exists():
        raise ValueError("Refuse overwrite device benchmark")
    sequence = features(
        "A bounded verified event with a temporal relationship.", (0.2, 0.4, 0.1)
    )
    result = {
        "seed": 260113,
        "repetitions": 50,
        "warmups": 10,
        "same_FP32_input": True,
        "learning": False,
        "scope": "H128 fixed sequence, one CPU BLAS thread; tiny GPU work can be slower due to launches/transfers",
        "arms": [],
    }
    outputs = {}
    for device in ["cpu", "gpu"]:
        model = LocalALIF(260113).to_device(device)
        for _ in range(10):
            model.forward(sequence)
        samples = []
        for _ in range(50):
            begin = time.perf_counter()
            output = model.forward(sequence)[0]
            samples.append(time.perf_counter() - begin)
        outputs[device] = output
        result["arms"].append(
            {
                "device": device,
                "samples_seconds": samples,
                "median_seconds": statistics.median(samples),
                "p95_seconds": float(np.quantile(samples, 0.95)),
                "output": output.tolist(),
            }
        )
    result["max_probability_difference"] = float(
        np.max(abs(outputs["cpu"] - outputs["gpu"]))
    )
    result["parity_within_3e_5"] = bool(
        np.allclose(outputs["cpu"], outputs["gpu"], atol=3e-5, rtol=3e-5)
    )
    with path.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "arms"}))
    print(
        json.dumps(
            [
                {k: v for k, v in a.items() if k != "samples_seconds"}
                for a in result["arms"]
            ]
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    a = p.parse_args()
    main(a.out)
