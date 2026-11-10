"""Short CPU/native/retrieval smoke measurements; no language quality claim."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scarx import Config, SpikingCore
from scarx.evaluation.runs import new_run, write_json
from scarx.runtime.resources import ResourceMonitor
from scarx.runtime.native import NativeCore
from scarx.interfaces.text import TextAgent
from scarx.memory.store import EpisodeStore

parser = argparse.ArgumentParser()
parser.add_argument("--seconds",type=float,default=10)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
run = new_run(root,"runtime-smoke-v1",{"seconds":args.seconds,"corpus_byte_tokens":1_000_000,"model":"random-64-ALIF"})
results = {}
with ResourceMonitor() as monitor:
    for backend in ("numpy","native"):
        config = Config(inputs=256,outputs=256)
        model = SpikingCore(config) if backend=="numpy" else NativeCore(config,root/"build"/"scarx_cpu.dll")
        agent = TextAgent(model,None)
        generation=[]; prefill=[]; chars=[]
        for _ in range(10):
            t = time.perf_counter()
            agent.generate("a"*512,1)
            prefill.append(time.perf_counter()-t)
            t=time.perf_counter()
            out=agent.generate("a"*512,64)
            generation.append(time.perf_counter()-t)
            chars.append(len(out["text"]))
        results[backend]={"warm_prefill_plus_first_byte_p95_seconds":float(np.quantile(prefill,.95)),
                          "end_to_end_64byte_generation_p50_seconds":float(np.median(generation)),
                          "bytes_per_second_including_prefill":64/float(np.median(generation)),
                          "display_characters_per_second":float(np.mean(chars))/float(np.median(generation)),
                          "valid_utf8":out["valid_utf8"],"quality":"random weights; meaningless throughput only"}
    store=EpisodeStore(run/"episodes.sqlite")
    t=time.perf_counter(); keys=[]
    for i in range(1000):
        tag=f"record{i:06d}"
        text=tag+" "+("neutral "*125)
        text=text[:1000].ljust(1000," ")
        keys.append(store.observe(text,"synthetic:runtime-smoke-v1"))
    ingest=time.perf_counter()-t
    hits=[]; latencies=[]
    for i in range(0,1000,37):
        t=time.perf_counter(); rows=store.search(f"record{i:06d}"); latencies.append(time.perf_counter()-t)
        hits.append(any(r["id"]==keys[i] for r in rows))
    results["retrieval"]={"utf8_bytes_or_byte_tokens":1_000_000,"ingest_seconds":ingest,"queries":len(hits),
                          "exact_identifier_recall":sum(hits)/len(hits),"p95_seconds":float(np.quantile(latencies,.95)),
                          "scope":"ASCII literal identifier lookup; not semantic, multihop, native-context or language-token evidence"}
    store.close()
    m=SpikingCore(Config(inputs=5)); rng=np.random.default_rng(77); cycles=0
    start=time.perf_counter()
    while time.perf_counter()-start<args.seconds:
        x=rng.normal(0,.3,(8,5)).astype(np.float32)
        m.forward(x,track=True); m.learn(cycles%4)
        if cycles%32==0:
            m.consolidate(); m.save(run/"endurance.npz"); m=SpikingCore.load(run/"endurance.npz"); monitor.check()
        cycles+=1
    results["endurance"]={"seconds":time.perf_counter()-start,"cycles":cycles,"eight_hour_pass":args.seconds>=28800}
results["resources"]=monitor.report()
write_json(run/"result.json",results)
(run/"report.md").write_text("# Runtime smoke measurements\n\n```json\n"+json.dumps(results,indent=2)+"\n```\n",encoding="utf-8")
print(json.dumps({"run":str(run),"results":results},indent=2))
