"""Verify pinned external source receipts and publish a mathematical source map."""
from pathlib import Path
import hashlib
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from scarx.evaluation.runs import new_run,write_json
from public_research_mirror import publish


def main():
    original=json.loads((ROOT/"artifacts/external-learning-sources-v2.json").read_text())
    supplement=ROOT/original["run"]/"supplementary-source-manifest.json"
    extra=json.loads(supplement.read_text())
    mapping=json.loads((ROOT/"configs/neural_method_audit_v2.json").read_text())
    names={x["method"] for x in original["sources"]}
    if names!={x["method"] for x in mapping["methods"]} or len(names)!=6:raise ValueError("Six source methods required")
    checked=[];errors=[]
    for record in [f for source in original["sources"] for f in source["files"]]+extra["files"]:
        if "local_cache" not in record:continue
        p=(ROOT/record["local_cache"]).resolve()
        if not p.is_relative_to(ROOT/"local/external-learning-sources"):raise ValueError("Source cache boundary")
        ok=p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==record["sha256"]
        checked.append({"url":record["url"],"sha256":record["sha256"],"cache_matches":ok})
        if not ok:errors.append(record["url"])
    c=mapping["cost_example"];p=c["hidden"]*(c["inputs"]+c["fan_in"]);po=c["outputs"]*c["hidden"]+c["outputs"]
    costs={"example":c,"hidden_synapses":p,"output_parameters_including_bias":po,
           "three_streaming_eligibility_arrays_bytes":3*p*c["float_bytes"],
           "scarx35_six_eligibility_bank_optimizer_arrays_bytes":6*p*c["float_bytes"],
           "two_mstdpet_eligibility_arrays_bytes":2*p*c["float_bytes"],
           "readout_si_three_auxiliary_arrays_bytes":3*po*c["float_bytes"],
           "replay_input_vectors_only_bytes":c["replay_examples"]*c["sequence_ticks"]*c["inputs"]*c["float_bytes"],
           "scope":"Arithmetic estimates only; exclude weights/stable copies, allocator, snapshots, label metadata and framework runtime. Not peak RAM or an energy/speed measurement."}
    run=new_run(ROOT,"neural-method-map-v2",{"display_title":"6 nguồn SNN/continual learning · đối chiếu công thức và quyền nghiên cứu","observability_kind":"source_audit","stage":1,"model_scope":"Pinned source/hash/license/math mapping only; no code reuse, external execution or learner training."},data_description="Six public repository source receipts; no private user data.",data_files=["configs/neural_method_audit_v2.json"])
    revisions={s["method"]:{"repository":s["repository"],"revision":s["revision"],"license_spdx_from_api":s.get("license_spdx")} for s in original["sources"]}
    result={"decision":"SOURCE_MATH_MAP_READY_FOR_RESEARCH_REVIEW" if not errors else "SOURCE_HASH_AUDIT_FAILED","stage":1,"sources":revisions,"mapping":mapping,"source_files_verified":len(checked),"source_hash_checks":checked,"errors":errors,"cost_arithmetic":costs,"code_reused":False,"external_code_executed":False,"new_learner_implemented":False,"training_executed":False,"production_ready":False}
    write_json(run/"result.json",result)
    write_json(ROOT/"artifacts/neural-method-map-v2.json",{"run":run.relative_to(ROOT).as_posix(),"decision":result["decision"],"source_files_verified":len(checked),"revisions":revisions,"hypotheses":mapping["hypotheses"],"training_executed":False})
    lines=["# Pinned neural learning source map V2","","No external code was executed or reused. Mathematical differences are source-audited; learning utility remains unproven.","","| Method | Existing SCAR-X Support | Mathematical Difference | Evidence | Cost | Risks | Experiment Needed | Authorization |","|---|---|---|---|---|---|---|---|---|"]
    keys=["method","existing_scarx_support","mathematical_difference","evidence","cost","risks","experiment_needed","authorization"]
    for row in mapping["methods"]:
        lines.append("| "+" | ".join((", ".join(row[k]) if isinstance(row[k],list) else row[k]).replace("|","/") for k in keys)+" |")
    lines.extend(["","## Pinned repositories and license review",""])
    for row in mapping["methods"]:
        source=revisions[row["method"]]
        lines.append(f"- [{row['method']}]({source['repository']}/tree/{source['revision']}): `{source['revision']}`. {row['license_review']}")
    lines.extend(["","## Auxiliary memory arithmetic","","```json",json.dumps(costs,ensure_ascii=False,indent=2),"```","","This map does not reopen a STOP experiment or authorize P2. See configs/neural_method_audit_v2.json for hypothesis scopes."])
    (run/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    publish(run)
    print(json.dumps({"run":run.name,"decision":result["decision"],"source_files_verified":len(checked),"errors":errors}))


if __name__=="__main__":main()
