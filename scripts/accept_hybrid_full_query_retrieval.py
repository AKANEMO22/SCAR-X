"""Actual SQLite before/after on owned data; candidate not applied to live source."""

import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sqlite3
import sys
import time
import tracemalloc
import traceback
import uuid
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from scarx.hybrid.store import Store, encode


def load_recipe():
    spec = importlib.util.spec_from_file_location("full_query_retrieval_acceptance", ROOT/"integrations/hermes/full_query_retrieval_patch.py")
    recipe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recipe)
    return recipe


def cases():
    rows = [{"id":"blue-pos-"+str(n),"query":"note "*(n-1)+"blue","expect":"blue","check_short_compat":n<=16}
            for n in (1,15,16,17,32,129)]
    rows += [{"id":"last-valid-keyword","query":"note "*1197+"blue","expect":"blue"},
             {"id":"Vietnamese-tail","query":"note "*80+"đèn màu xanh","expect":"vietnamese"},
             {"id":"punctuation-literal","query":"note "*40+'" OR NOT NEAR blue :)',"expect":"blue"},
             {"id":"no-hit","query":"unseenword "*30,"expect":None}]
    for i,text in enumerate(("blue","blue blue","blue OR pink","màu xanh","note filler0","!!!","alpha beta")):
        rows.append({"id":"short-"+str(i),"query":text,"check_short_compat":True})
    terms=[]
    for i in range(3000):
        trial=" ".join(terms+["word"+str(i)])
        if len(trial)>5940:
            break
        terms.append("word"+str(i))
    rows += [{"id":"many-unique-balanced-parser","query":" ".join(terms)+" blue","expect":"blue"},
             {"id":"stream-boundary","query":"note "*50+"blue","stream":"missing","expect":None},
             {"id":"time-boundary","query":"note "*50+"blue","stream":"main","before":0,"expect":None},
             {"id":"output-limit","query":"note "*30,"limit":1000,"max_rows":32}]
    assert all(len(row["query"])<=6000 for row in rows)
    return rows


def row_digest(store):
    evidence = [tuple(row) for row in store.db.execute("SELECT * FROM evidence ORDER BY id")]
    index = [tuple(row) for row in store.db.execute("SELECT id,text FROM evidence_fts ORDER BY id")]
    return hashlib.sha256(encode({"evidence":evidence,"index":index}).encode()).hexdigest()


def main():
    start=time.monotonic()
    name=time.strftime("%Y%m%dT%H%M%SZ",time.gmtime())+"-full-query-retrieval-p0-v1-"+uuid.uuid4().hex[:6]
    run=ROOT/"runs"/name
    data=ROOT/"local"/name
    run.mkdir();data.mkdir()
    locked=cases()
    (run/"locked-cases.json").write_text(encode(locked),encoding="utf-8")
    recipe=load_recipe()
    original=(ROOT/"src/scarx/hybrid/store.py").read_bytes()
    candidate=recipe.patch_bytes(original)
    (run/"candidate-store.py").write_bytes(candidate)
    scope={"__name__":"owned_full_query_SQLite_shadow"}
    exec(compile(candidate,"owned-candidate-store.py","exec"),scope)
    Candidate=scope["Store"]
    files=[Path(__file__),ROOT/"integrations/hermes/full_query_retrieval_patch.py",
           ROOT/"src/scarx/hybrid/store.py",ROOT/"experiments/hybrid-full-query-retrieval-p0-v1.md"]
    hashes={}
    for path in files:
        rel=path.relative_to(ROOT);dest=run/"source"/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(path.read_bytes());hashes[rel.as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
    (run/"manifest.json").write_text(encode({"run_id":name,"source_sha256":hashes,
        "candidate_sha256":hashlib.sha256(candidate).hexdigest(),
        "locked_cases_sha256":hashlib.sha256((run/"locked-cases.json").read_bytes()).hexdigest(),
        "no_live_source_activation":True,"new_model_or_neural_calls":0}),encoding="utf-8")
    fixture=Store(data/"fixture")
    records=[("blue","lamp color blue"),("pink","lamp color pink"),
             ("vietnamese","đèn màu xanh"),("other-vietnamese","đèn màu hồng")]
    records += [("filler-"+str(i),"note filler"+str(i)+" alpha beta gamma") for i in range(1024)]
    with fixture.db:
        for identifier,text in records:
            fixture.db.execute("INSERT INTO evidence VALUES(?,?,?,?,?,?,?,?,?,?,?)",(
                identifier,"main",0.,1.,text,"owned-retrieval-fixture","text","observed",None,0.,hashlib.sha256(text.encode()).hexdigest()))
            fixture.db.execute("INSERT INTO evidence_fts VALUES(?,?)",(identifier,text))
    for arm in ("base","candidate"):
        directory=data/arm;directory.mkdir()
        with sqlite3.connect(directory/"hybrid.sqlite") as target:
            fixture.db.backup(target)
    fixture_digest=row_digest(fixture);fixture.close()
    baseline=Store(data/"base");shadow=Candidate(data/"candidate")
    report={"run_id":name,"rows":[],"failures":[],"fixture_records":len(records),
            "fixture_digest":fixture_digest,"neural_calls":0,"live_source_changed":False,
            "Python_process_RSS_before":psutil.Process().memory_info().rss}
    try:
        assert row_digest(baseline)==row_digest(shadow)==fixture_digest
        for case in locked:
            result={"case_id":case["id"],"input_characters":len(case["query"]),
                    "input_lexical_words":len(re.findall(r"[^\W_]+",case["query"],re.UNICODE))}
            for arm,store in (("base",baseline),("candidate",shadow)):
                statements=[]
                store.db.set_trace_callback(statements.append)
                tracemalloc.start()
                begun=time.monotonic()
                rows=store.search(case["query"],stream=case.get("stream","main"),before=case.get("before"),limit=case.get("limit",12))
                elapsed=time.monotonic()-begun
                _,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
                store.db.set_trace_callback(None)
                result[arm]={"ids":[row["id"] for row in rows],"seconds":elapsed,
                             "peak_traced_Python_bytes":peak,"SQL_trace":statements}
            if case.get("check_short_compat"):
                assert result["base"]["ids"]==result["candidate"]["ids"],"Short-query order changed: "+case["id"]
            if "expect" in case:
                if case["expect"] is None:
                    assert result["candidate"]["ids"]==[],"Unexpected filtered/no-hit rows"
                else:
                    assert case["expect"] in result["candidate"]["ids"],"Expected owned source missing"
            assert len(result["candidate"]["ids"])<=case.get("max_rows",32)
            assert result["candidate"]["seconds"]<=2
            assert result["candidate"]["peak_traced_Python_bytes"]<=10_000_000
            report["rows"].append(result)
        for invalid in (None,True,[],"x"*6001):
            try:
                shadow.search(invalid)
            except ValueError:
                continue
            raise AssertionError("Malformed/out-of-bound search was accepted")
        assert row_digest(baseline)==row_digest(shadow)==fixture_digest
        assert (ROOT/"src/scarx/hybrid/store.py").read_bytes()==original
        assert time.monotonic()-start<=30
        report.update(gate_pass=True,verdict="FULL_QUERY_SQLITE_SHADOW_CONTRACT_PASS_NOT_LIVE_OR_NEURAL_GAIN")
    except Exception:
        report["failures"].append(traceback.format_exc())
        report.update(gate_pass=False,verdict="FULL_QUERY_SQLITE_SHADOW_INCOMPLETE_OR_FAILED")
    finally:
        baseline.close();shadow.close()
        if tracemalloc.is_tracing():tracemalloc.stop()
    report["seconds"]=time.monotonic()-start
    report["Python_process_RSS_after"]=psutil.Process().memory_info().rss
    with (run/"result.json").open("x",encoding="utf-8") as out:out.write(encode(report))
    live=Store(ROOT/"local/hybrid-v1")
    try:
        live.event("full_query_retrieval.shadow_P0_completed",{k:v for k,v in report.items() if k!="rows"},"orchestrator")
    finally:live.close()
    print(encode({k:report[k] for k in ("run_id","verdict","gate_pass","failures","seconds")}))


if __name__=="__main__":
    main()
