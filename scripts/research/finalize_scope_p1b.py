"""Final read-only regression/hash checks plus new custody receipt; no fit/forward."""
import os
for name in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):os.environ[name]="1"
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import time
import numpy as np
import psutil


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text(encoding="utf-8"))
def write(path,obj):
    with path.open("x",encoding="utf-8") as f:json.dump(obj,f,ensure_ascii=False,indent=2);f.write("\n")


def main():
    start=time.perf_counter();ap=argparse.ArgumentParser();ap.add_argument("--run",type=Path,required=True)
    run=ap.parse_args().run.resolve();root=Path(__file__).resolve().parents[2];phase=run/"learning";out=run/"closure"
    initial=read(out/"custody.json");fail=[]
    for rel,expected in initial["files"].items():
        if sha(run/rel)!=expected:fail.append("Original closure file changed: "+rel)
    if sha(out/"report-at-first-custody.md")!=initial["report_sha256"]:fail.append("Original report snapshot hash")
    # Verify CSV values exactly against the saved NPZ predictions, not just aggregate scores.
    rows=list(csv.DictReader((run/"summary/raw-predictions.csv").open(encoding="utf-8",newline="")))
    actual={(int(r["seed"]),int(r["sign"]),r["arm"],r["dataset"],int(r["update"]),r["case_id"]):r for r in rows}
    count=0
    for r in read(phase/"completed.json"):
        with np.load(run/r["relative_directory"]/"predictions.npz",allow_pickle=False) as z:
            with np.load(phase/f"fit-{r['seed']}.npz",allow_pickle=False) as f:oldids=f["old_ids"]
            specs=[(d,u,pkey,ykey,ids) for u in (0,48,96,192)
                   for d,pkey,ykey,ids in (("DEV",f"dev_p_u{u}","dev_y",z["dev_ids"]),
                                           ("old_valid",f"old_p_u{u}","old_y",oldids))]
            specs.append(("DEV_hidden_revert",192,"revert_hidden_p","dev_y",z["dev_ids"]))
            for dataset,u,pkey,ykey,ids in specs:
                for p,y,identifier in zip(z[pkey],z[ykey],ids):
                    row=actual[(r["seed"],r["sign"],r["arm"],dataset,u,str(identifier))]
                    if int(row["label"])!=int(y) or float(row["p0"])!=float(p[0]) or float(row["p1"])!=float(p[1]):
                        fail.append("CSV/NPZ mismatch")
                    count+=1
    if count!=len(rows) or count!=len(actual):fail.append("CSV unique row count")
    before=read(run/"hardware.json")["serving_before"];serving=[]
    for p in before["processes"]:
        try:
            process=psutil.Process(p["pid"])
            serving.append({"pid":p["pid"],"created":process.create_time(),"name":process.name()})
            if process.create_time()!=p["created"]:fail.append("Serving PID reused")
        except psutil.Error:fail.append("Serving process missing")
    expected_scripts={"run_scope_p1b_learning.py","intake_scope_p1b.py","preflight_scope_p1b.py",
                      "profile_scope_p1b_backend.py","restart_scope_p1b_student.py"}
    remaining=[]
    for proc in psutil.process_iter(["pid","name","cmdline"]):
        if not (proc.info["name"] or "").lower().startswith("python"):continue
        if any(Path(arg).name in expected_scripts for arg in (proc.info["cmdline"] or [])):remaining.append(proc.info["pid"])
    if remaining:fail.append("Research worker still running")
    report=root/"docs/research/SCOPE_P1B_TEMPORAL_TRANSFER_RESULT_20261116.md"
    links=re.findall(r"\]\(([^)]+)\)",report.read_text(encoding="utf-8"))
    for target in links:
        if not target.startswith("http"):
            path=(report.parent/target).resolve()
            if path!=out/"FINAL_MANIFEST.json" and not path.exists():fail.append("Broken report link: "+target)
    write(out/"documentation-amendment.json",{
        "changes":["Audit peak-RSS/commit explicitly marked unmeasured separately",
                   "Fixed aligned-equation line breaks; linked final custody receipt"],
        "original_report_sha256":initial["report_sha256"],"final_report_sha256":sha(report),
        "no_raw_metrics_protocol_source_or_checkpoints_modified":True})
    receipt={"pass":not fail,"failures":fail,"CSV_NPZ_exact_rows":count,
             "initial_custody_files_unchanged":len(initial["files"]),"serving_processes":serving,
             "remaining_research_workers":remaining,"model_forward_calls":0,"fit_calls":0,
             "memory_OS_self":psutil.Process().memory_info()._asdict(),
             "elapsed_seconds_before_final_manifest":time.perf_counter()-start,
             "Hermes_writes_or_telemetry_injection":False,
             "latest_no_Hermes_write_scope":"Results saved to research artifacts only; no live DB events appended"}
    write(out/"final-checks.json",receipt)
    files={str(p.relative_to(run)).replace("\\","/"):sha(p) for p in sorted(run.rglob("*")) if p.is_file()}
    write(out/"FINAL_MANIFEST.json",{"algorithm":"SHA256","files":files,"file_count":len(files),
                                   "excludes":"this FINAL_MANIFEST.json itself","final_report_sha256":sha(report),
                                   "finalizer_source_sha256":sha(Path(__file__))})
    print(json.dumps({**receipt,"custody_files":len(files),"wall_seconds":time.perf_counter()-start}))
    if fail:raise SystemExit(1)


if __name__=="__main__":main()
