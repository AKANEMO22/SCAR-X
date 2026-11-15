"""Saved-prediction statistics; conditional on the three fitted seed models."""
import numpy as np


def contrasts(records, root, prefix="dev_p_u192", repeats=10000):
    from pathlib import Path
    arrays={};labels={};metadata={}
    for r in records:
        with np.load(Path(root)/r["relative_directory"]/"predictions.npz",allow_pickle=False) as z:
            arrays[(r["seed"],r["sign"],r["arm"])]=(z[prefix].argmax(axis=1)==z["dev_y"]).astype(float)
            labels[r["seed"]]=z["dev_y"]
            metadata[r["seed"]]=list(zip(z["dev_order"].tolist(),z["dev_gap"].tolist(),z["dev_clock"].tolist()))
    seeds=sorted(labels);rng=np.random.default_rng(2026111609)
    boot={arm:np.zeros(repeats) for arm in ("label","shuffled","readout")}
    for seed in seeds:
        strata={k:[i for i,v in enumerate(metadata[seed]) if v==k] for k in set(metadata[seed])}
        sampled=np.empty((repeats,len(labels[seed])),np.int64);offset=0
        for key in sorted(strata):
            idx=np.asarray(strata[key]);n=len(idx)
            sampled[:,offset:offset+n]=idx[rng.integers(0,n,size=(repeats,n))];offset+=n
        for arm in boot:
            diffs=np.stack([arrays[(seed,sign,"teacher")]-arrays[(seed,sign,arm)] for sign in (1,-1)]).mean(axis=0)
            boot[arm]+=diffs[sampled].mean(axis=1)/len(seeds)
    result={}
    for arm,values in boot.items():
        effects=[float((arrays[(seed,sign,"teacher")]-arrays[(seed,sign,arm)]).mean()) for seed in seeds for sign in (1,-1)]
        result["teacher-minus-"+arm]={"gain":float(np.mean(effects)),"per_seed_sign_gain":effects,
            "CI95_input_conditional":np.quantile(values,[.025,.975]).tolist(),
            "CI98_333_input_conditional_Bonferroni3":np.quantile(values,[.05/6,1-.05/6]).tolist()}
    return result


def gate(records, statistics):
    teacher=[r for r in records if r["arm"]=="teacher"]
    acquisition=all(r["final_dev"]["accuracy"]>=.65 and r["final_dev"]["min_recall"]>=.65 for r in teacher)
    old=all(r["final_old"]["accuracy"]>=r["curves"][0]["old"]["accuracy"]-.03 for r in teacher)
    revert=all(r["final_dev"]["accuracy"]-r["hidden_revert"]["accuracy"]>=.05-1e-12 for r in teacher)
    gain=all(min(s["per_seed_sign_gain"])>=.05-1e-12 for s in statistics.values())
    ci=all(s["CI98_333_input_conditional_Bonferroni3"][0]>0 for s in statistics.values())
    checks={"acquisition_and_minclass":acquisition,"old_valid_retained":old,"hidden_revert":revert,
            "gain_each_seed_and_sign_vs_all_controls":gain,"conditional_adjusted_CI_lower_positive":ci}
    return {"pilot_gate_pass":all(checks.values()),"checks":checks,
            "population_seed_CI_status":"INCONCLUSIVE_N3_NOT_PROJECT_QUALIFIED","twenty_stage_retention":"NOT_MEASURED"}
