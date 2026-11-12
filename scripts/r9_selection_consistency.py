"""Audit stored R9 ranking/selection arithmetic only; no model execution."""
from pathlib import Path
from collections import defaultdict
import hashlib,json,subprocess,sys
ROOT=Path(__file__).resolve().parents[1];SHA='2c41fdaf6fc491845879d638ac80449caf597471'


def raw(name):return subprocess.check_output(['git','show',SHA+':artifacts/gemini-r9/'+name],cwd=ROOT)
def read(name):return json.loads(raw(name+'.json'))


def main():
    run=Path(sys.argv[1]).resolve();output=run/'selection-consistency.json'
    if run.parent!=ROOT/'runs' or output.exists():raise ValueError('Fresh owned appendix required')
    screened=[json.loads(line) for line in raw('dev_screen_results.jsonl').splitlines() if line.strip()]
    ranking=read('candidate_rankings');ablations=read('causal_ablations');stress=read('robustness_results');summary=read('search_summary')
    grouped=defaultdict(list)
    for row in screened:grouped[row['group']].append(row)
    shortlist=[]
    for group,rows in grouped.items():
        rows.sort(key=lambda r:r['composite_score'],reverse=True)
        shortlist+=rows[:4 if group.startswith(('H1','H5')) else 3]
    top16=[r['candidate_id'] for r in sorted(shortlist,key=lambda r:r['composite_score'],reverse=True)[:16]]
    causal=[];mismatches=[]
    for cid in top16:
        data=ablations[cid];passed=data['causal_revert']['mean']>=.05 and data['learned_candidate']['mean']-data['shuffled_signal']['mean']>=.02
        if passed:causal.append(cid)
        if passed!=(data['causal_verdict']=='CONFIRMED_CAUSAL'):mismatches.append(cid)
    top5=(causal+[cid for cid in top16 if cid not in causal])[:5]
    def priority(cid):return (int(cid in causal),stress[cid]['heavy_distractor']['mean']-stress[cid]['reversed_stream']['mean'])
    top2=sorted(top5,key=priority,reverse=True)[:2]
    result={'peer_sha':SHA,'top16_matches_raw_ranking':top16==ranking['overall_top16'],'causal_verdict_mismatches':mismatches,
        'top5_matches_raw_summary':top5==summary['top5_candidates'],'top2_matches_raw_summary':top2==summary['confirmed_candidates'],
        'selected_candidates':top2,'causal_stage_passed_candidates':causal,'all_primary_json_artifacts_read':True,
        'source_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'new_model_or_training_calls':0,
        'scope':'Arithmetic consistency is not source/cache freshness, raw-prediction replication or confirmation independence'}
    output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))


if __name__=='__main__':main()
