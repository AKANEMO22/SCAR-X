"""Fixed public G1 baseline and separate, bounded local Qwen reference."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))

import psutil

from scarx.evaluation.basic_knowledge_eval import grade, reference_case, summarize
from scarx.evaluation.knowledge_factory import inference_input
from scarx.evaluation.runs import new_run, write_json
from scarx.runtime.experiment_status import progress
from scarx.runtime.limits import apply_memory_limit
from scarx.runtime.resources import ResourceMonitor
from scarx.runtime.work_trace import append
from public_research_mirror import publish


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def load_data():
    receipt=json.loads((ROOT/"artifacts/basic-knowledge-verified-v1.json").read_text())
    path=ROOT/receipt["data"]
    if sha(path)!=receipt["sha256"]:raise ValueError("Frozen source pool changed")
    ledger=ROOT/receipt["source_ledger"]
    if sha(ledger)!=receipt["source_ledger_sha256"]:raise ValueError("Source ledger changed")
    all_rows=json.loads(path.read_text(encoding="utf-8"))["items"]
    rows=sorted([r for r in all_rows if r["split"] in ("development","unknown_development")],key=lambda r:r["id"])
    if len(rows)!=96:raise ValueError("Registered 96 development cases required")
    facts={f["id"]:f for f in json.loads(ledger.read_text(encoding="utf-8"))["facts"]}
    return receipt, rows, facts


def make_run(phase,receipt,cases):
    return new_run(ROOT,"basic-knowledge-"+phase+"-baseline-v1",{
        "protocol":"experiments/basic-knowledge-baseline-v1.md","display_title":"Kiến thức cơ bản · "+phase+" · đo thật, không học",
        "observability_kind":"baseline_evaluation","cases":cases,"model_scope":"Public checkpoint, development only; no retrieval, labels in inference, private state or learning."},
        data_description="Frozen actual FlashHigh questions; labels joined only after generation for evaluation.",data_files=[receipt["data"],receipt["source_ledger"]])


def scarx():
    from scarx.neural.unified import UnifiedCore
    from scarx.evaluation.product_acceptance import canonical_reply,weights_sha256
    receipt,cases,facts=load_data()
    descriptor=json.loads((ROOT/"artifacts/generation1-release.json").read_text())
    checkpoint=ROOT/descriptor["checkpoint"]
    if sha(checkpoint)!=descriptor["sha256"]:raise ValueError("Public checkpoint hash mismatch")
    cap=apply_memory_limit(2_000_000_000)
    model=UnifiedCore.loads(checkpoint.read_bytes());before=model.dumps();weights=weights_sha256(model)
    run=make_run("scarx",receipt,96);records=[];error=None;start=time.perf_counter()
    (run/"report.md").write_text("# Public G1 knowledge baseline\n\nRunning fixed development-only inference; no learning.\n",encoding="utf-8")
    publish(run)
    with ResourceMonitor(interval=.02) as monitor:
        try:
            for i,row in enumerate(cases,1):
                if time.perf_counter()-start>120:raise TimeoutError("SCAR-X 120s bound")
                prompt=inference_input(row)["prompt"]
                began=time.perf_counter();reply=canonical_reply(model,prompt);seconds=time.perf_counter()-began
                record={"id":row["id"],"fact_id":row["fact_id"],"split":row["split"],"language":row["language"],
                        "prompt":prompt,"generated":reply["text"],"expected_reference":row["canonical_answer"],
                        "source_id":row["source_id"],"source_hash":row["source_hash"],"seconds":seconds,
                        "neural_metadata":reply,"grade":grade(reply["text"],row,facts[row["fact_id"]]),
                        "qwen_reference_case":reference_case(row)}
                records.append(record)
                append(run,"answer",{"case_id":row["id"],"prompt":prompt,"generated":reply["text"],
                     "expected_reference":row["canonical_answer"],"seconds":seconds,"allocated":model.core.config.hidden,
                     "checkpoint":descriptor["sha256"],"scope":"PUBLIC G1 / DEVELOPMENT / NO LEARNING / NOT PRODUCTION READY"})
                progress(run,"Đã đo "+row["id"],i,96,metrics={"new_learning_updates":0,"seconds":seconds})
                monitor.check()
            if model.dumps()!=before or weights_sha256(model)!=weights:raise AssertionError("Source model changed")
        except Exception as exc:error=type(exc).__name__+": "+str(exc)[:240]
    result={"decision":"NOT_PRODUCTION_READY_BASELINE_ONLY" if error is None else "BASELINE_ERROR","rows":records,
            "summary":summarize(records),"checkpoint":descriptor,"resources":monitor.report(),"cap":cap,
            "model_unchanged":model.dumps()==before and weights_sha256(model)==weights,"source_sha256":receipt["sha256"],
            "new_learning_updates":0,"final_test_opened":False,"live_state_read_or_written":False,
            "qwen_used_as_labels_or_proxy":False,"error":error,"seconds":time.perf_counter()-start}
    write_json(run/"result.json",result);write_json(ROOT/"artifacts/basic-knowledge-public-baseline-v1.json",{
        "run":run.relative_to(ROOT).as_posix(),"decision":result["decision"],"summary":result["summary"],"checkpoint_sha256":descriptor["sha256"],"error":error})
    (run/"report.md").write_text("# Public G1 development knowledge baseline\n\n"+json.dumps({k:v for k,v in result.items() if k!="rows"},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    publish(run);print(json.dumps({"run":run.name,"summary":result["summary"],"error":error},ensure_ascii=False),flush=True)
    return int(error is not None)


def qwen():
    receipt,cases,facts=load_data();cases=[r for r in cases if reference_case(r)]
    if len(cases)!=24:raise ValueError("Locked 24-case reference subset required")
    if any((p.info["name"] or "").casefold()=="llama-server.exe" for p in psutil.process_iter(["name"])):
        raise RuntimeError("Existing llama-server left untouched; reference not started")
    baseline=json.loads((ROOT/"artifacts/basic-knowledge-public-baseline-v1.json").read_text())
    source_run=ROOT/baseline["run"]
    prior=json.loads((source_run/"result.json").read_text(encoding="utf-8"))
    if prior["error"] or not prior["model_unchanged"]:raise ValueError("Public baseline must be complete")
    paired={r["id"]:r for r in prior["rows"]}
    descriptor=json.loads((ROOT/"artifacts/qwen4b-reference-download.json").read_text())
    modelpath=Path(descriptor["local_path"])
    if modelpath.stat().st_size!=descriptor["expected_bytes"] or sha(modelpath)!=descriptor["expected_sha256"]:
        raise ValueError("Qwen descriptor hash/size mismatch")
    exe=Path(os.environ["LOCALAPPDATA"])/"hermes/runtimes/llamacpp/b10679/cpu/llama-server.exe"
    if not exe.is_file():raise FileNotFoundError("Registered CPU runtime unavailable")
    version=subprocess.run([str(exe),"--version"],capture_output=True,text=True,encoding="utf-8",errors="replace",timeout=10)
    cap=apply_memory_limit(4_500_000_000)
    run=make_run("qwen",receipt,24);records=[];error=None;process=None;start=time.perf_counter()
    private=ROOT/"local/basic-knowledge-reference"/run.name;private.mkdir(parents=True)
    with socket.socket() as listener:listener.bind(("127.0.0.1",0));port=listener.getsockname()[1]
    command=[str(exe),"-m",str(modelpath),"--host","127.0.0.1","--port",str(port),"-c","1024","-t","2","-tb","2","-b","64","-ub","64","-ngl","0","-np","1","--no-mmap","--no-webui"]
    (run/"report.md").write_text("# Independent Qwen knowledge reference\n\nRunning 24 fixed public questions; no source/labels in inference.\n",encoding="utf-8")
    publish(run)
    with ResourceMonitor(limit=4_500_000_000,interval=.1) as monitor,(private/"stdout.log").open("wb") as out,(private/"stderr.log").open("wb") as err:
        try:
            process=subprocess.Popen(command,cwd=private,stdin=subprocess.DEVNULL,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0)
            psutil.Process(process.pid).nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
            print(json.dumps({"run":run.name,"owned_qwen_pid":process.pid,"selected_cases":24}),flush=True)
            base=f"http://127.0.0.1:{port}";startup=time.perf_counter()
            while True:
                try:json.load(urllib.request.urlopen(base+"/health",timeout=1));break
                except (OSError,urllib.error.HTTPError):
                    if process.poll() is not None or time.perf_counter()-startup>90:raise RuntimeError("Reference startup failed or timed out")
                    monitor.check();time.sleep(.2)
            for i,row in enumerate(cases,1):
                if time.perf_counter()-start>420:raise TimeoutError("Qwen420s reference bound")
                req=urllib.request.Request(base+"/v1/chat/completions",json.dumps({"model":"qwen4b-reference","messages":[{"role":"user","content":inference_input(row)["prompt"]}],"temperature":0,"seed":919,"max_tokens":128,"stream":False}).encode(),headers={"Content-Type":"application/json"})
                began=time.perf_counter();response=json.load(urllib.request.urlopen(req,timeout=30));choice=response["choices"][0];answer=choice["message"].get("content") or ""
                record={"id":row["id"],"fact_id":row["fact_id"],"split":row["split"],"language":row["language"],
                        "prompt":row["prompt"],"generated":answer,"finish_reason":choice.get("finish_reason"),"usage":response.get("usage"),
                        "seconds":time.perf_counter()-began,"expected_reference":row["canonical_answer"],
                        "grade":grade(answer,row,facts[row["fact_id"]]),"scarx":paired[row["id"]]}
                records.append(record)
                append(run,"answer",{"case_id":row["id"],"prompt":row["prompt"],"generated":paired[row["id"]]["generated"],
                    "expected_reference":row["canonical_answer"],"qwen":record,"scope":"PUBLIC G1 / QWEN REFERENCE / DEVELOPMENT24 / NO TEACHING"})
                progress(run,"Đã đối chiếu "+row["id"],i,24,metrics={"new_learning_updates":0,"seconds":record["seconds"]})
                monitor.check()
                if i%4==0:print(json.dumps({"completed":i,"expected":24}),flush=True)
        except Exception as exc:error=type(exc).__name__+": "+str(exc)[:240]
        finally:
            if process is not None and process.poll() is None:
                process.terminate()
                try:process.wait(timeout=8)
                except subprocess.TimeoutExpired:process.kill();process.wait(timeout=5)
    result={"decision":"INDEPENDENT_REFERENCE_ONLY" if error is None else "REFERENCE_PARTIAL_ERROR","rows":records,
            "summary":summarize(records),"descriptor":descriptor,"runtime_sha256":sha(exe),"runtime_version_stdout":version.stdout[:1000],
            "runtime_version_stderr":version.stderr[:1000],"runtime_version_exit_code":version.returncode,"command":command,
            "resources":monitor.report(),"cap":cap,"source_baseline_run":baseline["run"],"source_sha256":receipt["sha256"],
            "completed":len(records),"expected":24,"truncated":sum(r["finish_reason"]=="length" for r in records),
            "server_stopped":process is None or process.poll() is not None,"qwen_used_as_labels_or_proxy":False,
            "new_learning_updates":0,"final_test_opened":False,"live_state_read_or_written":False,"error":error,"seconds":time.perf_counter()-start,
            "knowledge_scope_warning":"Neither model receives the pinned repository source; pretraining is not evidence Qwen knows these versioned facts. Prose fluency and unrestricted semantics are not automatically graded."}
    write_json(run/"result.json",result);write_json(ROOT/"artifacts/basic-knowledge-qwen-reference-v1.json",{
        "run":run.relative_to(ROOT).as_posix(),"decision":result["decision"],"summary":result["summary"],"completed":len(records),"truncated":result["truncated"],"server_stopped":result["server_stopped"],"error":error})
    (run/"report.md").write_text("# Independent Qwen knowledge reference\n\n"+json.dumps({k:v for k,v in result.items() if k!="rows"},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    publish(run);print(json.dumps({"run":run.name,"summary":result["summary"],"completed":len(records),"truncated":result["truncated"],"server_stopped":result["server_stopped"],"error":error},ensure_ascii=False),flush=True)
    return int(error is not None)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--phase",choices=("scarx","qwen"),required=True)
    args=parser.parse_args();raise SystemExit(scarx() if args.phase=="scarx" else qwen())
