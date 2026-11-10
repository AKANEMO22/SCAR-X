from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from ..neural.core import Config, SpikingCore
from ..runtime.resources import ResourceMonitor
from .tasks import episodes, accuracy
from .runs import new_run, write_json, cluster_interval


def train(model, samples, mode):
    loss = []
    for x,y in samples:
        model.forward(x,track=True)
        loss.append(model.learn(y,mode="readout" if mode=="rate" else mode))
    return float(np.mean(loss))


def continual(config, settings, mode, seed, monitor, deadline):
    stages = settings["continual_stages"]
    model = SpikingCore(Config(**{**asdict(config),"inputs":4+stages}))
    rng = np.random.default_rng(3000+seed)
    reservoir = []
    seen = 0
    scores = []
    test = [list(episodes(6000+seed*100+j,settings["stage_test_examples"],contexts=stages,context=j)) for j in range(stages)]
    for stage in range(stages):
        if time.monotonic() > deadline:
            raise TimeoutError("Registered wall-time budget exceeded")
        fresh = list(episodes(4000+seed*100+stage,settings["stage_train_examples"],contexts=stages,context=stage))
        replay = [reservoir[i] for i in rng.choice(len(reservoir),min(len(reservoir),settings["replay_per_stage"]),replace=False)] if reservoir else []
        mixed = fresh+replay
        rng.shuffle(mixed)
        train(model,mixed,mode)
        if mode != "frozen":
            model.consolidate()
        for sample in fresh:
            seen += 1
            if len(reservoir) < settings["replay_capacity"]:
                reservoir.append(sample)
            else:
                j = int(rng.integers(seen))
                if j < len(reservoir):
                    reservoir[j] = sample
        scores.append([float(accuracy(model,test[j]).mean()) for j in range(stage+1)])
        monitor.check()
    final = scores[-1]
    forgetting = [max(scores[s][j] for s in range(j,stages))-final[j] for j in range(stages)]
    return {"matrix":scores,"final_mean":float(np.mean(final)),"forgetting_mean":float(np.mean(forgetting)),
            "forgetting_worst":float(max(forgetting)),"per_task_forgetting":forgetting,"replay_samples":len(reservoir)}


def benchmark(root, settings):
    run = new_run(root,settings["protocol"],settings)
    deadline = time.monotonic()+settings["max_seconds"]
    results = []
    try:
        with ResourceMonitor(settings["ram_limit"]) as monitor:
            for seed in settings["seeds"]:
                for mode in settings["modes"]:
                    config = Config(inputs=5,hidden=settings["hidden"],outputs=4,fan_in=settings["fan_in"],
                                    plastic_rows=settings["plastic_rows"],seed=seed,neuron="rate" if mode=="rate" else "alif")
                    model = SpikingCore(config)
                    initial_hidden = model.w_in.copy()
                    samples = list(episodes(2000+seed,settings["test_examples"]))
                    before = float(accuracy(model,samples).mean())
                    loss = train(model,episodes(1000+seed,settings["train_examples"]),mode)
                    predictions = accuracy(model,samples)
                    delayed = float(accuracy(model,list(episodes(2500+seed,settings["test_examples"],delay=8))).mean())
                    checkpoint = run/f"{seed}-{mode}.npz"
                    model.save(checkpoint)
                    # Independent process reload: no recurrent state/retrieval can help.
                    code = "import json; from scarx import SpikingCore; from scarx.evaluation.tasks import episodes,accuracy; import sys; m=SpikingCore.load(sys.argv[1]); print(json.dumps(accuracy(m,list(episodes(int(sys.argv[2]),int(sys.argv[3])))).tolist()))"
                    cp = subprocess.run([sys.executable,"-c",code,str(checkpoint),str(2000+seed),str(settings["test_examples"])],capture_output=True,text=True,check=True,timeout=60)
                    restart = bool(np.array_equal(predictions,np.asarray(json.loads(cp.stdout))))
                    hidden_l2 = float(np.linalg.norm(model.w_in-initial_hidden))
                    train(model,episodes(2600+seed,settings["train_examples"],reverse=True),mode)
                    reversal = float(accuracy(model,list(episodes(2700+seed,settings["test_examples"],reverse=True))).mean())
                    old_rule = float(accuracy(model,samples).mean())
                    result = {"seed":seed,"mode":mode,"initial_accuracy":before,"accuracy":float(predictions.mean()),
                              "delayed_accuracy":delayed,"train_mean_loss":loss,"restart_equal":restart,
                              "hidden_weight_l2_change":hidden_l2,"neural_array_bytes":model.memory_bytes(),
                              "reversal_accuracy":reversal,"old_rule_after_reversal":old_rule,
                              "correct":predictions.astype(int).tolist(),
                              "continual":continual(config,settings,mode,seed,monitor,deadline)}
                    results.append(result)
                    with (run/"trials.jsonl").open("a",encoding="utf-8") as f:
                        f.write(json.dumps(result,allow_nan=False)+"\n")
                    print(f"seed={seed} mode={mode} accuracy={result['accuracy']:.3f} forgetting={result['continual']['forgetting_mean']:.3f}",flush=True)
                    monitor.check()
        lookup = {(r["seed"],r["mode"]):r for r in results}
        gain = cluster_interval([lookup[s,"full"]["accuracy"]-lookup[s,"frozen"]["accuracy"] for s in settings["seeds"]])
        hidden_gain = cluster_interval([lookup[s,"full"]["accuracy"]-lookup[s,"readout"]["accuracy"] for s in settings["seeds"]])
        full = [r for r in results if r["mode"]=="full"]
        gates = {"learns_vs_frozen":gain["lower95"] >= settings["gain_required"],
                 "hidden_plasticity_contribution":hidden_gain["lower95"] > 0,
                 "delayed_above_chance":all(r["delayed_accuracy"] > .25 for r in full),
                 "retention":all(r["continual"]["forgetting_mean"] <= settings["forgetting_mean_max"] and r["continual"]["forgetting_worst"] <= settings["forgetting_group_max"] for r in full),
                 "restart_parity":all(r["restart_equal"] for r in results),
                 "observed_memory":not monitor.exceeded}
        summary = {"status":"complete","trials":results,"gain_vs_frozen":gain,"gain_vs_readout":hidden_gain,
                   "gates":gates,"expand_language":all(gates.values()),"resources":monitor.report(),
                   "verdict":"GO_SMALL_LANGUAGE" if all(gates.values()) else "NO_GO_SCALE_PIVOT_LOCAL_RULE",
                   "not_evaluated":["pretrained low-RAM LLM", "commercial parity", "8h endurance", "iGPU", "energy", "10M-token language training"],
                   "warning":"Synthetic recall only. No claim of semantic understanding. Backprop control is a separate run."}
        write_json(run/"result.json",summary)
        lines = ["# Phase 2 measured result", "",f"Verdict: **{summary['verdict']}**", "", "| Seed | Condition | Recall | Delay | Mean forgetting | Worst forgetting |", "|---|---|---:|---:|---:|---:|"]
        lines += [f"| {r['seed']} | {r['mode']} | {r['accuracy']:.3f} | {r['delayed_accuracy']:.3f} | {r['continual']['forgetting_mean']:.3f} | {r['continual']['forgetting_worst']:.3f} |" for r in results]
        lines += ["", "## Gates", ""]+[f"- {k}: {v}" for k,v in gates.items()]
        lines += ["",f"Full − frozen: {gain}",f"Full − readout-only: {hidden_gain}","",f"Resources: {summary['resources']}","",summary["warning"],"", "Unmeasured: "+", ".join(summary["not_evaluated"])]
        (run/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
        return run,summary
    except BaseException as exc:
        write_json(run/"failure.json",{"type":type(exc).__name__,"message":str(exc),"completed_trials":len(results)})
        (run/"report.md").write_text(f"# Incomplete run\n\n{type(exc).__name__}: {exc}\n",encoding="utf-8")
        raise

