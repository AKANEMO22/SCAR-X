"""Bounded DEV replay of the three locked purpose-overview prompts."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]


def main():
    old = Path((ROOT / "local/current-purpose-run.txt").read_text().strip())
    name = (
        time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        + "-workspace-purpose-v2-"
        + uuid.uuid4().hex[:6]
    )
    run, data = ROOT / "runs" / name, ROOT / "local" / name
    run.mkdir()
    data.mkdir()
    ids = ["actual-vi", "actual-en", "fixture-vi"]
    for ident in ids:
        job = json.loads((old / (ident + ".job.json")).read_text(encoding="utf-8"))
        job["home"] = str(data / ("home-" + ident))
        (run / (ident + ".job.json")).write_text(
            json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    hashes = {}
    for rel in [
        "src/scarx/hybrid/tool_routing.py",
        "src/scarx/hybrid/gateway.py",
        "scripts/hybrid_agent_workload.py",
        "scripts/run_hybrid_workspace_purpose.py",
        "integrations/hermes/compat_tests/read_only_audit_plugin.py",
        "experiments/hybrid-workspace-purpose-repair-v2.md",
    ]:
        p = ROOT / rel
        dest = run / "source" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(p.read_bytes())
        hashes[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "prepared": time.time(),
                "parent_run": old.name,
                "sources_sha256": hashes,
                "private_data": str(data),
                "wall_limit_per_job": 90,
                "scope": "control-flow repair, runtime notices cannot count as model-quality success",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (ROOT / "local/current-purpose-run.txt").write_text(str(run))
    rows = []
    for ident in ids:
        private = ident.startswith("actual")
        out = (data if private else run) / (ident + ".result.json")
        log = data / (ident + ".process.log")
        started = time.monotonic()
        row = {"job": ident}
        try:
            with log.open("w", encoding="utf-8") as stream:
                subprocess.run(
                    [
                        str(
                            ROOT
                            / "local/hermes-reliability-test-env/Scripts/python.exe"
                        ),
                        str(ROOT / "scripts/hybrid_agent_workload.py"),
                        "--job",
                        str(run / (ident + ".job.json")),
                        "--out",
                        str(out),
                    ],
                    cwd=ROOT,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    timeout=90,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            raw = json.loads(out.read_text(encoding="utf-8"))
            row.update(
                {
                    k: v
                    for k, v in raw.items()
                    if k not in ["transcript", "compacted_history"]
                }
            )
            row["semantic_quality_pass"] = (
                raw.get("answer_producer") != "runtime_controlled_halt"
                and all(raw.get("expected_terms_present", {}).values())
                and bool(raw.get("expected_terms_present"))
            )
            row["tool_loop_bounded"] = len(raw.get("tool_calls", [])) <= 4
            if private:
                row.pop("answer", None)
                row.update(
                    private_output=str(out),
                    private_output_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
                )
        except Exception as exc:
            row.update(completed=False, error=type(exc).__name__ + ": " + str(exc))
        row["wall_seconds"] = time.monotonic() - started
        rows.append(row)
        (run / "summary.json").write_text(
            json.dumps(
                {
                    "rows": rows,
                    "all_finished": len(rows) == len(ids),
                    "neural_learning_claim": False,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(json.dumps(row, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
