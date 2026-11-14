"""Independent literal-term coverage and readonly actual SQL reexecution."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from scarx.hybrid.store import encode,Store


def main(run):
    run=run.resolve()
    assert run.is_relative_to(ROOT/"runs")
    read=lambda name:json.loads((run/name).read_text(encoding="utf-8"))
    manifest,cases,result=map(read,("manifest.json","locked-cases.json","result.json"))
    assert result["gate_pass"] and not result["failures"] and result["seconds"]<=30
    assert len(cases)==len(result["rows"])
    assert hashlib.sha256((run/"locked-cases.json").read_bytes()).hexdigest()==manifest["locked_cases_sha256"]
    assert hashlib.sha256((run/"candidate-store.py").read_bytes()).hexdigest()==manifest["candidate_sha256"]
    for rel,digest in manifest["source_sha256"].items():
        assert hashlib.sha256((run/"source"/rel).read_bytes()).hexdigest()==digest
    databases={}
    digests={}
    maxima={"base_seconds":0.,"candidate_seconds":0.,"candidate_peak_python_bytes":0,
            "candidate_MATCH_characters":0,"candidate_expression_max_depth":0}
    short_equal=source_hits=0
    try:
        for arm in ("base","candidate"):
            path=(ROOT/"local"/run.name/arm/"hybrid.sqlite").resolve()
            db=sqlite3.connect(path.as_uri()+"?mode=ro",uri=True)
            db.row_factory=sqlite3.Row
            databases[arm]=db
            content={"evidence":[tuple(row) for row in db.execute("SELECT * FROM evidence ORDER BY id")],
                     "index":[tuple(row) for row in db.execute("SELECT id,text FROM evidence_fts ORDER BY id")]}
            digests[arm]=hashlib.sha256(encode(content).encode()).hexdigest()
            assert digests[arm]==result["fixture_digest"]
        for case,row in zip(cases,result["rows"]):
            assert row["case_id"]==case["id"] and row["input_characters"]==len(case["query"])<=6000
            words=re.findall(r"[^\W_]+",case["query"],re.UNICODE)
            assert len(words)==row["input_lexical_words"]
            for arm in ("base","candidate"):
                actual=row[arm]
                maxima[arm+"_seconds"]=max(maxima[arm+"_seconds"],actual["seconds"])
                statements=[statement for statement in actual["SQL_trace"]
                            if not statement.lstrip().startswith("--")]
                if not words:
                    assert not statements and actual["ids"]==[]
                    continue
                assert len(statements)==1 and statements[0].startswith("SELECT e.* FROM evidence_fts f JOIN evidence e ON e.id=f.id WHERE evidence_fts MATCH ")
                # Database is readonly and statement came from a bounded owned
                # fixture. No command outside this exact SELECT is executed.
                replay=[record["id"] for record in databases[arm].execute(statements[0])]
                assert replay==actual["ids"]
                match=re.search(r"evidence_fts MATCH '([^']*)'",statements[0]).group(1)
                terms=re.findall(r'"([^\"]+)"',match)
                if arm=="base":
                    assert terms==words[:16]
                elif len(words)<=16:
                    assert terms==words
                else:
                    assert terms==list(dict.fromkeys(word.lower() for word in words))
                    depth=maximum=0
                    for char in match:
                        depth+=int(char=="(")-int(char==")")
                        assert depth>=0
                        maximum=max(maximum,depth)
                    assert depth==0 and maximum<=math.ceil(math.log2(len(terms)))
                    maxima["candidate_expression_max_depth"]=max(maxima["candidate_expression_max_depth"],maximum)
                if arm=="candidate":
                    maxima["candidate_MATCH_characters"]=max(maxima["candidate_MATCH_characters"],len(match))
            if case.get("check_short_compat"):
                assert row["base"]["ids"]==row["candidate"]["ids"]
                short_equal+=1
            if "expect" in case:
                if case["expect"] is None:
                    assert row["candidate"]["ids"]==[]
                else:
                    assert case["expect"] in row["candidate"]["ids"]
                source_hits+=int(case["expect"] is not None)
            maxima["candidate_peak_python_bytes"]=max(maxima["candidate_peak_python_bytes"],row["candidate"]["peak_traced_Python_bytes"])
            assert len(row["candidate"]["ids"])<=32 and row["candidate"]["seconds"]<=2
            assert row["candidate"]["peak_traced_Python_bytes"]<=10_000_000
    finally:
        for db in databases.values():db.close()
    report={"run_id":run.name,"verdict":"SHADOW_SQLITE_QUERY_COVERAGE_AND_RESULTS_DATA_VERIFIED_NOT_LIVE_PROMOTION",
            "ordered_short_query_compatibility_cases":short_equal,"source_hit_cases":source_hits,
            "fixture_records":result["fixture_records"],"logical_cold_database_digest_equal":True,
            "all_distinct_long_query_terms_present":True,"maxima":maxima,
            "additional_neural_or_VLM_calls":0,"new_source_activation":False,
            "neural_improvement_or_general_retrieval_quality_PASS":False}
    with (run/"independent-data-audit-v1.json").open("x",encoding="utf-8") as output:
        output.write(encode(report))
    store=Store(ROOT/"local/hybrid-v1")
    try:store.event("full_query_retrieval.data_verified",report,"orchestrator")
    finally:store.close()
    print(encode(report))


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("run",type=Path)
    main(parser.parse_args().run)
