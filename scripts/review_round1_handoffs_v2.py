"""Read-only branch handoff audit. No learner training or live-state access."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import subprocess
import sys
import time
import uuid

import psutil

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[2]
PARENT = PROJECT.parent
SPECS = {
    "grok": (PARENT / "scar-x-grok-round1", "4b93593df4368011f7ec2066ce4e3a6478a6dcb6"),
    "mtb_deepseek_session": (PROJECT / "local/worktrees/antigravity-scarx-neural-round1-20261111", "5e39c8f8fb3f316c2713d1c2f6d5edea0598882f"),
    "deepseek": (PARENT / "scar-x-deepseek-round1", "02be6e9a916df9faf61c6fd746280ba9ea225fc7"),
    "antigravity_gemini": (PROJECT / "local/worktrees/review-gemini-81641f7", "81641f773145f5c441b5ad3b0f5771f645a5cbd6"),
}


def git(path, *args):
    return subprocess.check_output(["git", "-C", str(path), *args], text=True, encoding="utf-8").strip()


def pinned_json(label, name):
    path, sha = SPECS[label]
    raw = subprocess.check_output(["git", "-C", str(path), "show", f"{sha}:{name}"])
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def child(run, label, path, args):
    env = dict(os.environ, PYTHONPATH=str(path / "src"), PYTHONDONTWRITEBYTECODE="1",
               PYTHONUTF8="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    log = run / f"{label}.log"
    started = time.monotonic()
    peak_rss = peak_vms = 0
    with log.open("w", encoding="utf-8") as output:
        p = subprocess.Popen([sys.executable, *args], cwd=path, env=env,
                             stdout=output, stderr=subprocess.STDOUT)
        while p.poll() is None:
            try:
                processes = [psutil.Process(p.pid)]
                processes += processes[0].children(recursive=True)
                readings = []
                for process in processes:
                    try:
                        readings.append(process.memory_info())
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                rss = sum(x.rss for x in readings)
                vms = sum(x.vms for x in readings)
                peak_rss, peak_vms = max(peak_rss, rss), max(peak_vms, vms)
                if max(rss, vms) > 2_000_000_000 or time.monotonic() - started > 180:
                    for process in reversed(processes):
                        try:
                            process.kill()
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                    raise RuntimeError(f"Review child exceeded memory/time budget: {label}")
            except psutil.NoSuchProcess:
                pass
            time.sleep(0.05)
        code = p.wait()
    record = {"returncode": code, "seconds": time.monotonic() - started,
              "sampled_peak_tree_rss_bytes": peak_rss, "sampled_peak_tree_commit_or_vms_bytes": peak_vms,
              "sample_interval_seconds": 0.05, "log": str(log.relative_to(ROOT)),
              "command": [sys.executable, *args], "cwd": str(path)}
    if code:
        raise RuntimeError(f"Review check failed: {label}; see {log}")
    if args[0] == "-c":
        record["output"] = json.loads(log.read_text(encoding="utf-8").splitlines()[-1])
    return record


def main():
    now = datetime.now(timezone.utc)
    run = ROOT / "runs" / (now.strftime("%Y%m%dT%H%M%SZ") + "-round1-handoff-review-v2-" + uuid.uuid4().hex[:6])
    run.mkdir(parents=True, exist_ok=False)
    result = {"scope": "Read-only source/results review plus bounded engineering checks; no training or live-state access",
              "created_utc": now.isoformat(), "run_dir": str(run.relative_to(ROOT)),
              "ponytail_available": False, "branches": {}, "checks": {}}
    for label, (path, sha) in SPECS.items():
        actual = git(path, "rev-parse", "HEAD")
        if actual != sha:
            raise RuntimeError(f"Branch moved before review: {label}: {actual}")
        result["branches"][label] = {"commit": sha, "worktree": str(path)}

    mtb_code = '''import json, numpy as np
from scarx.neural.core import Config, SpikingCore
from scarx.evaluation.multi_timescale import MTCore
c=Config(inputs=8,hidden=48,outputs=2,fan_in=8,plastic_rows=12,seed=17)
a=MTCore(c,deterministic=True); b=MTCore.loads(a.dumps())
before={"original_deterministic":a.deterministic,"restored_deterministic":b.deterministic}
for _ in range(4): a.step(np.zeros(8,np.float32)); b.step(np.zeros(8,np.float32))
before["continuation_spikes_equal"]=bool(np.array_equal(a.spikes,b.spikes))
before["continuation_rng_equal"]=a.rng.bit_generator.state==b.rng.bit_generator.state
c=Config(inputs=8,hidden=48,outputs=2,fan_in=8,plastic_rows=12,threshold=.5,feature_decay=0.)
m=SpikingCore(c);m.w_in.fill(0);m.w_rec.fill(0);m.w_out.fill(0);m.bias.fill(0)
m.w_in[0,:2]=1.;m.w_in[1,:2]=1./3.;m.w_out[1,0]=2.;m.w_out[1,1]=-4.;m.bias[1]=-1.
rows=[]
for x,y in [(0,0),(0,1),(1,0),(1,1)]:
 v=np.zeros(8,np.float32);v[:2]=[x,y];p=m.forward([v]);rows.append({"input":[x,y],"prediction":int(p.argmax()),"target":x^y})
print(json.dumps({"mtb_checkpoint_mode":before,"threshold_xor_counterexample":rows,"xor_scope":"Hand-set one-tick threshold circuit only: refutes additive-current-implies-linear argument, not evidence of learning or strict-binding capacity."}))
'''
    result["checks"]["mtb_mode_and_capacity_argument"] = child(run, "mtb-mode", SPECS["mtb_deepseek_session"][0], ["-c", mtb_code])

    ds_code = '''import json, pathlib, runpy, sys
p=pathlib.Path.cwd();sys.path.insert(0,str(p/'scripts'))
d=json.loads((p/'runs/20261110T203340Z-deepseek-pair-binding-screen-v1-9da1fc/result.json').read_text())
ns=runpy.run_path(str(p/'scripts/run_deepseek_pair_binding.py'))
s=ns['summarise'](d['trials'],d['summary']['locked_readout_lr'])
print(json.dumps({"stored_gates":d['summary']['gates'],"stored_unknown":d['summary']['gates_unknown'],"recomputed_gates":s['gates'],"recomputed_unknown":s['gates_unknown'],"stored_screen_pass":d['summary']['screen_pass'],"recomputed_screen_pass":s['screen_pass'],"means":{k:v['direct']['mean'] for k,v in s['per_task']['pair_binding'].items()},"diagnostic_script_tracked":False}))
'''
    result["checks"]["deepseek_summary_only"] = child(run, "deepseek-summary", SPECS["deepseek"][0], ["-c", ds_code])
    result["checks"]["gemini_engineering_tests"] = child(run, "gemini-p0-tests", SPECS["antigravity_gemini"][0],
        ["-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/test_dc_alif_p0.py",
         "tests/test_readout_eligibility.py", "tests/test_readout_usefulness_p0.py", "--junitxml=" + str(run / "gemini-p0-junit.xml")])

    for label, name in [
        ("mtb_deepseek_session", "runs/20261110T194509Z-collab-mtb-screen-v1-de8c62/result.json"),
        ("deepseek", "runs/20261110T203340Z-deepseek-pair-binding-screen-v1-9da1fc/result.json"),
        ("antigravity_gemini", "artifacts/collab/scarx-neural-round1-20261111/antigravity/dc-alif-round1-latest.json")]:
        d, digest = pinned_json(label, name)
        result["branches"][label]["result"] = name
        result["branches"][label]["result_sha256"] = digest
        if label == "antigravity_gemini":
            result["branches"][label]["p1_confirmation"] = d["p1_confirmation"]
            result["branches"][label]["p2_retention_summary"] = d["p2_retention"]["by_condition"]
            result["branches"][label]["resources"] = d["resources"]
        else:
            result["branches"][label]["screen_pass"] = d["summary"]["screen_pass"]
            result["branches"][label]["trials"] = len(d["trials"])
    result["production_learner_integrated"] = False
    result["main_before_after"] = git(PROJECT, "rev-parse", "main")
    result["other_agent_source_edited"] = False
    (run / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run_dir": str(run), "checks": {k:{a:b for a,b in v.items() if a not in ("command","output")} for k,v in result["checks"].items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
