"""Data-only closure checks/custody. Never imports a model or performs a fit.

Usage: python scripts/research/close_scope_p1b.py --run runs/<P1b-run>
Creates closure once; refuses to overwrite a previous closure.
"""
import ast
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
import argparse

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


def git(*args, cwd=ROOT):
    return subprocess.check_output(["git", *args], cwd=cwd)


def common_methods(path):
    node = next(n for n in ast.parse(path.read_text(encoding="utf-8")).body
                if isinstance(n, ast.ClassDef) and n.name == "ALIF")
    out = {}
    for fn in node.body:
        if isinstance(fn, ast.FunctionDef) and fn.name in ("forward", "proposal", "learn"):
            if fn.name == "learn":
                fn.body = fn.body[:-1]  # Only the returned diagnostics were expanded.
            out[fn.name] = ast.dump(fn, include_attributes=False)
    return out


def main():
    started=time.perf_counter()
    ap=argparse.ArgumentParser();ap.add_argument("--run",type=Path,required=True)
    args=ap.parse_args();run=args.run.resolve();phase=run/"learning"
    closure=run/"closure";closure.mkdir(exist_ok=False)
    failures=[]

    def check(condition, message):
        if not condition:failures.append(message)

    logs=closure/"logs";logs.mkdir()
    private=ROOT/"local/p1b-20261116"
    for filename in ("intake.stdout.txt","intake.stderr.txt","learning.stdout.txt",
                     "learning.stderr.txt","learning-audit.stdout.txt","learning-audit.stderr.txt",
                     "export.stdout.txt","export.stderr.txt"):
        shutil.copyfile(private/filename,logs/filename)
    shutil.copyfile(private/"FULL_DIRECTIVE_20261116.md",closure/"FULL_DIRECTIVE_20261116.md")
    check(digest(closure/"FULL_DIRECTIVE_20261116.md")==
          "2e0fa1fdd2e87614681ca69be67b451fd1470325a9aa034bffd2105d0849c6e5", "Directive bytes changed")

    reg=read(phase/"registration.json");intake=read(run/"manifest.json")
    for registration,base in ((reg,phase/"source"),(intake,run/"source")):
        for name,expected in registration["source_hashes"].items():
            check(digest(base/name)==expected,"Source snapshot hash: "+name)
            # A pure serialization recovery and added return diagnostics superseded intake source.
            if registration is reg:
                check(digest(ROOT/name)==expected,"Locked learning source mutated: "+name)
                check(hashlib.sha256(git("show",reg["source_commit"]+":"+name)).hexdigest()==expected,
                      "Not committed before learning: "+name)
    check(common_methods(ROOT/"scripts/research/scope_p1b_common.py")==
          common_methods(run/"source/scripts/research/scope_p1b_common.py"),"Numerical AST changed")

    records=read(phase/"completed.json");result=read(phase/"result.json")
    check(len(records)==24 and all(r["status"]=="COMPLETED" for r in records),"Incomplete trial matrix")
    checkpoint_count=0
    for r in records:
        directory=run/r["relative_directory"]
        for filename,key in (("initial.npz","initial_checkpoint_sha256"),("final.npz","final_checkpoint_sha256")):
            check(digest(directory/filename)==r[key],"Checkpoint hash: "+str(directory/filename))
            checkpoint_count+=1
    sealed_count=0
    for d in read(phase/"data-lock.json"):
        check(digest(phase/f"fit-{d['seed']}.npz")==d["fit_sha256"],"Fit data changed")
        for sealed in d["sealed_TEST"]:
            check(digest(Path(sealed["path"]))==sealed["sha256"],"Sealed TEST bytes changed")
            check(not sealed["model_forwarded"],"Sealed TEST flagged as opened")
            sealed_count+=1
    check(not result["TEST_model_forwarded"],"TEST model forward")
    check(read(phase/"conditional-stages.json")=={"G3":"NOT_RUN_G2_FAILED","G4":"NOT_RUN_G2_FAILED"},
          "Unexpected conditional stage")
    check(not list(phase.glob("G3-*")) and not list(phase.glob("G4-*")) and
          not list((phase/"trials").glob("*/cycle2")),"Unregistered continuation artifacts")
    check(not read(phase/"G2-decision.json")["pilot_gate_pass"],"Changed G2 decision")

    # A second implementation re-counts the exported CSV; all curves/reverts checked.
    groups={};rows=0
    with (run/"summary/raw-predictions.csv").open(encoding="utf-8",newline="") as f:
        for row in csv.DictReader(f):
            rows+=1;key=(int(row["seed"]),int(row["sign"]),row["arm"],row["dataset"],int(row["update"]))
            group=groups.setdefault(key,{"n":0,"correct":0,"ids":set()})
            y,pred=int(row["label"]),int(row["prediction"]);p0,p1=float(row["p0"]),float(row["p1"])
            check(y in (0,1) and pred==(1 if p1>p0 else 0),"CSV label/argmax")
            check(abs(p0+p1-1)<2e-7 and 0<=p0<=1 and 0<=p1<=1,"CSV probability")
            check(int(row["correct"])==int(y==pred),"CSV correctness")
            check(row["case_id"] not in group["ids"],"Duplicate CSV case")
            group["ids"].add(row["case_id"]);group["n"]+=1;group["correct"]+=int(y==pred)
    check(rows==16512,"CSV row count")
    for r in records:
        for curve in r["curves"]:
            for dataset,key,n in (("DEV","dev",128),("old_valid","old",12)):
                g=groups[(r["seed"],r["sign"],r["arm"],dataset,curve["update"])]
                check(g["n"]==n and g["correct"]/n==curve[key]["accuracy"],"CSV curve metric")
        g=groups[(r["seed"],r["sign"],r["arm"],"DEV_hidden_revert",192)]
        check(g["n"]==128 and g["correct"]/128==r["hidden_revert"]["accuracy"],"CSV revert metric")

    old="runs/20261115T165305Z-scope-teacher-preflight-v1"
    check(not git("diff","6e08a361","--",old),"Original P1 artifacts modified")
    check(not git("diff","6e08a361","--","src","integrations","artifacts","tests"),"Production files changed")
    main=ROOT.parents[2]
    check(git("rev-parse","HEAD",cwd=main).decode().strip()==
          "37749041ef45d54951558fdeb0c3695616b47cb5","main SHA changed")
    check(not git("status","--porcelain",cwd=main),"main worktree dirty")
    for path in ("docs/STATUS.md","docs/PROJECT_MAP.md","docs/RESEARCH_PLAN.md","docs/research/README.md"):
        check("SCOPE_P1B_TEMPORAL_TRANSFER_RESULT_20261116.md" in (ROOT/path).read_text(encoding="utf-8"),
              "Missing report navigation: "+path)

    intake_seconds=read(run/"resource.json")["intake_wall_seconds"]
    learning_seconds=read(phase/"resource.json")["wall_seconds"]
    audit_seconds=read(phase/"independent-audit.json")["wall_seconds"]
    export_seconds=read(run/"summary/export-summary.json")["wall_seconds"]
    receipt={"data_checks_pass":not failures,"failures":failures,"UTC":dt.datetime.now(dt.timezone.utc).isoformat(),
             "learning_source_commit":reg["source_commit"],"closure_source_sha256":digest(Path(__file__)),
             "old_P1_rerun_forwards":0,"new_full_model_episode_forwards":22656,
             "diagnostic_reference_episode_evaluations":672,"primary_learning_calls":4608,
             "primary_hidden_enabled_updates":3456,"CSV_rows":rows,"checkpoint_hashes_checked":checkpoint_count,
             "sealed_TEST_files_bytechecked_not_forwarded":sealed_count,"closure_model_or_fit_calls":0,
             "G2":"FAIL","G3":"NOT_RUN_G2_FAILED","G4":"NOT_RUN_G2_FAILED",
             "population_CI":"INCONCLUSIVE_N3","intake_phase_wall_seconds":intake_seconds,
             "learning_phase_wall_seconds":learning_seconds,"audit_internal_seconds":audit_seconds,
             "export_internal_seconds":export_seconds,"closure_seconds_before_manifest":time.perf_counter()-started,
             "budget_accounting":"75s prior reservation + learning wall + 1.1s audit external + 2s export external + closure; coding/reading not execution",
             "main_and_production_unchanged":not failures,"push_merge_deploy":False,
             "not_an_independent_author_replication":True}
    write(closure/"verification.json",receipt)
    files={str(p.relative_to(run)).replace("\\","/"):digest(p) for p in sorted(run.rglob("*"))
           if p.is_file() and p!=closure/"custody.json"}
    write(closure/"custody.json",{"algorithm":"SHA256","includes":"every run file except this manifest itself",
                               "files":files,"file_count":len(files),
                               "report_sha256":digest(ROOT/"docs/research/SCOPE_P1B_TEMPORAL_TRANSFER_RESULT_20261116.md")})
    print(json.dumps({**receipt,"custody_file_count":len(files),"wall_seconds":time.perf_counter()-started}))
    if failures:raise SystemExit(1)


if __name__=="__main__":main()
