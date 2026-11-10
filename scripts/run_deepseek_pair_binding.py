"""Registered runner — deepseek pair_binding branch.

Protocol: experiments/collab/scarx-neural-round1-20261111/deepseek/protocol-v1.md

  --phase screen        3 dev seeds, 3 tasks x 4 arms x 3 readout_lr = 108 trials
  --phase confirmation  only if the screen passed; 8 fresh seeds, locked lr

Every number lands under runs/<unique-run-id>/ with a full manifest.
"""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scarx.evaluation.pair_binding import order3_binding, pair_binding  # noqa: E402
from scarx.evaluation.phase21 import holm, interval, signflip  # noqa: E402
from scarx.evaluation.runs import new_run, write_json  # noqa: E402
from scarx.evaluation.tasks21 import representation, ridge_probe  # noqa: E402
from scarx.neural.core import Config  # noqa: E402
from scarx.neural.plasticity import PlasticCore  # noqa: E402
from scarx.runtime.resources import ResourceMonitor  # noqa: E402

TASKS = ("pair_binding", "strict_binding", "order")
ARMS = ("score", "frozen", "readout", "shuffle")
DEV_SEEDS = (41001, 41007, 41013)
CONF_SEEDS = (42001, 42007, 42013, 42019, 42023, 42029, 42031, 42037)
LR_GRID = (0.025, 0.05, 0.10)

CAL, TRAIN, FIT, TEST = 6_200_000, 6_300_000, 6_400_000, 6_500_000
N_TRAIN, N_CAL, N_FIT = 2048, 128, 128
N_TEST_DEV, N_TEST_CONF = 256, 512
GATE = .65
GAIN_GATE_PP = 5.0


def dataset(task, seed, count):
    if task == "pair_binding":
        return pair_binding(seed, count)
    if task == "strict_binding":
        return order3_binding(seed, count)
    return representation(task, seed, count)


def make_config(seed, readout_lr):
    return Config(inputs=8, hidden=48, outputs=2, fan_in=8, plastic_rows=12,
                  seed=seed, leak=.65, threshold=.5, feature_decay=.25,
                  adaptation_strength=.02, readout_lr=readout_lr,
                  hidden_lr=.002, anchor_strength=0.)


def build(arm, seed, lr):
    # PlasticCore rule names: 'readout' trains the head only; 'frozen' trains nothing.
    rule = {"score": "score", "frozen": "frozen",
            "readout": "readout", "shuffle": "score"}[arm]
    m = PlasticCore(make_config(seed, lr), rule, shuffle=(arm == "shuffle"))
    return m


def train(model, x, y):
    """One pass. Frozen still calls learn() so the head-loss is observed and the
    no-update path is exercised, but PlasticCore('frozen') returns early."""
    losses = []
    for seq, label in zip(x, y):
        model.forward(seq, track=True)
        losses.append(float(model.learn(int(label))))
    return losses


def evaluate(model, task, seed, phase):
    n_test = N_TEST_DEV if phase == "screen" else N_TEST_CONF
    fx, fy = dataset(task, seed + FIT, N_FIT)
    tx, ty = dataset(task, seed + TEST, n_test)
    preds = model.predict_many(tx, 903)
    direct = float(np.mean(preds == ty))
    probe = ridge_probe(model.features_many(fx, 902), fy, model.features_many(tx, 903), ty)
    init = PlasticCore(make_config(seed, model.config.readout_lr), "frozen")
    probe_init = ridge_probe(init.features_many(fx, 902), fy, init.features_many(tx, 903), ty)
    return {"direct": direct, "probe": probe, "probe_initial": probe_init,
            "probe_gain": probe - probe_init, "n_test": int(len(ty))}


def trial(job):
    phase, task, seed, arm, lr = job
    t0 = time.perf_counter()
    with ResourceMonitor(interval=.1) as mon:
        model = build(arm, seed, lr)
        x, y = dataset(task, seed + TRAIN, N_TRAIN)
        hidden_before = model.w_in.copy()
        losses = train(model, x, y)
        hidden_moved = not np.array_equal(model.w_in, hidden_before)
        sc = evaluate(model, task, seed, phase)
        mon.check()
    row = {"phase": phase, "task": task, "seed": seed, "arm": arm, "readout_lr": lr,
           "direct": sc["direct"], "probe": sc["probe"],
           "probe_initial": sc["probe_initial"], "probe_gain": sc["probe_gain"],
           "n_test": sc["n_test"], "hidden_moved": bool(hidden_moved),
           "train_seconds": time.perf_counter() - t0,
           "mean_loss": float(np.mean(losses)) if losses else None,
           "resources": mon.report()}
    print(json.dumps({k: row[k] for k in ("task", "arm", "seed", "readout_lr",
                                          "direct", "probe_gain", "hidden_moved")}), flush=True)
    return row


def select_lr(rows):
    locked, detail = {}, {}
    for arm in ARMS:
        arm_rows = [r for r in rows if r["arm"] == arm]
        if not arm_rows:
            continue
        scores = {}
        for lr in sorted({r["readout_lr"] for r in arm_rows}):
            scores[lr] = float(np.mean([r["direct"] for r in arm_rows if r["readout_lr"] == lr]))
        best = max(scores.items(), key=lambda kv: (round(kv[1], 6), -kv[0]))[0]
        locked[arm] = best
        detail[arm] = scores
    return locked, detail


def paired(rows, a, b, metric, task):
    by = {(r["task"], r["seed"], r["arm"]): r for r in rows}
    d = []
    for seed in sorted({r["seed"] for r in rows}):
        ra, rb = by.get((task, seed, a)), by.get((task, seed, b))
        if ra and rb:
            d.append(ra[metric] - rb[metric])
    return interval(d) if d else None


def summarise(rows, locked):
    sel = [r for r in rows if r["readout_lr"] == locked.get(r["arm"])]
    out = {"locked_readout_lr": locked, "scored_rows": len(sel),
           "per_task": {}, "contrasts": {}}

    for task in TASKS:
        out["per_task"][task] = {}
        for arm in ARMS:
            vals = [r["direct"] for r in sel if r["task"] == task and r["arm"] == arm]
            gains = [r["probe_gain"] for r in sel if r["task"] == task and r["arm"] == arm]
            if vals:
                out["per_task"][task][arm] = {
                    "direct": interval(vals), "probe_gain": interval(gains)}
        out["contrasts"][task] = {
            "score_minus_frozen": paired(sel, "score", "frozen", "direct", task),
            "score_minus_readout": paired(sel, "score", "readout", "direct", task),
            "score_minus_shuffle": paired(sel, "score", "shuffle", "direct", task),
            "gain_score_minus_frozen": paired(sel, "score", "frozen", "probe_gain", task),
        }

    # frozen arms must not move the hidden
    out["frozen_hidden_untouched"] = all(
        not r["hidden_moved"] for r in sel if r["arm"] == "frozen")

    def lo(task, key):
        c = out["contrasts"][task][key]
        return None if c is None else c["lower95"]

    pb = "pair_binding"
    c1_ok = None  # established in P0, not from learning rows
    frozen_direct = out["per_task"][pb]["frozen"]["direct"]["mean"]
    out["gates"] = {
        "C1_task_valid": c1_ok,  # see P0 receipt: order-1 <= .60, order-2 == 1.00
        "C2_frozen_below_gate": bool(frozen_direct < GATE),
        "C3_learning_measurable": (None if lo(pb, "score_minus_frozen") is None
                                   else bool(lo(pb, "score_minus_frozen") > 0)),
        "score_beats_readout": (None if lo(pb, "score_minus_readout") is None
                                else bool(lo(pb, "score_minus_readout") > 0)),
        "score_beats_shuffle": (None if lo(pb, "score_minus_shuffle") is None
                                else bool(lo(pb, "score_minus_shuffle") > 0)),
        "frozen_hidden_untouched": out["frozen_hidden_untouched"],
    }
    known = [v for v in out["gates"].values() if v is not None]
    out["gates_unknown"] = [k for k, v in out["gates"].items() if v is None]
    # C1 is proven in P0; treat it as satisfied only if the P0 receipt says so.
    p0 = Path(__file__).resolve().parents[1] / "artifacts" / "deepseek-r1" / "p0-validation.json"
    p0_ok = p0.exists() and json.loads(p0.read_text(encoding="utf-8")).get("verdict") == "P0_PASS"
    out["gates"]["C1_task_valid"] = bool(p0_ok)
    out["screen_pass"] = bool(all(v is True for v in out["gates"].values()))
    out["significance"] = {t: signflip([r["direct"] for r in sel
                                        if r["task"] == t and r["arm"] == "score"])
                           for t in TASKS}
    return out


def render(result):
    s = result["summary"]
    L = [f"# deepseek {result['phase']} run", "",
         f"Scope: {result['scope']}", "",
         "## Acquisition (direct %, mean of dev seeds)", "",
         "| arm | " + " | ".join(TASKS) + " |",
         "|---|" + "---|" * len(TASKS)]
    for arm in ARMS:
        cells = []
        for t in TASKS:
            v = s["per_task"].get(t, {}).get(arm, {}).get("direct")
            cells.append("—" if v is None else f"{v['mean']*100:.2f}")
        L.append(f"| {arm} | " + " | ".join(cells) + " |")
    L += ["", "## Contrasts per task (pp)", "", "| task | contrast | mean [95% CI] |", "|---|---|---|"]
    for t in TASKS:
        for k, c in s["contrasts"][t].items():
            if c:
                L.append(f"| {t} | {k} | {c['mean']*100:+.2f} "
                         f"[{c['lower95']*100:+.2f}, {c['upper95']*100:+.2f}] |")
            else:
                L.append(f"| {t} | {k} | — |")
    L += ["", "## Gates", "", "```json", json.dumps(s["gates"], indent=2), "```",
          "", f"**screen_pass: {s['screen_pass']}**", "",
          f"unknown gates: {s['gates_unknown']}", ""]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("screen", "confirmation"), default="screen")
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    args = ap.parse_args()
    root, phase = Path(args.root), args.phase
    seeds = DEV_SEEDS if phase == "screen" else CONF_SEEDS

    config = {"phase": phase, "tasks": list(TASKS), "arms": list(ARMS),
              "seeds": list(seeds), "lr_grid": list(LR_GRID),
              "n_train": N_TRAIN, "n_test": N_TEST_DEV if phase == "screen" else N_TEST_CONF,
              "gate_accuracy": GATE, "gain_gate_pp": GAIN_GATE_PP,
              "interval": "seed bootstrap 10000, rng 2101 (phase21.interval)"}
    run = new_run(root, f"deepseek-pair-binding-{phase}-v1", config, data_description=(
        "Repository-authored deterministic generators: pair_binding (this branch), "
        "order3_binding (existing strict_binding), tasks21.representation('order'). "
        "No downloaded/private/model-generated data."))
    print(f"RUN {run}", flush=True)

    jobs = list(itertools.product((phase,), TASKS, seeds, ARMS, LR_GRID))
    locked, lr_detail = {}, {}
    if phase == "confirmation":
        lock = json.loads((root / "artifacts" / "deepseek-r1" / "locked-readout-lr.json").read_text())
        locked = lock["per_arm"]
        jobs = [j for j in jobs if j[4] == locked[j[3]]]
    print(f"{len(jobs)} trials", flush=True)

    rows = []
    with ResourceMonitor(interval=.2) as mon:
        for i, job in enumerate(jobs, 1):
            rows.append(trial(job))
            if i % 10 == 0:
                mon.check()
                write_json(run / "progress.json", rows)
                print(f"  {i}/{len(jobs)}", flush=True)

    if phase == "screen" and not locked:
        locked, lr_detail = select_lr(rows)
    summary = summarise(rows, locked)
    result = {"protocol": f"deepseek-pair-binding-{phase}-v1", "phase": phase,
              "config": config, "trials": rows, "summary": summary,
              "lr_selection": lr_detail, "resources": mon.report(),
              "scope": "Registered synthetic tasks at H=48. No chat/vision/reasoning claim."}
    write_json(run / "result.json", result)
    (run / "report.md").write_text(render(result), encoding="utf-8")
    if phase == "screen":
        lp = root / "artifacts" / "deepseek-r1" / "locked-readout-lr.json"
        lp.parent.mkdir(parents=True, exist_ok=True)
        write_json(lp, {"per_arm": locked, "selected_on": "development only",
                        "screen_run": run.name, "screen_pass": summary["screen_pass"]})
    print(f"RESULT {run}", flush=True)
    print("GATES:", json.dumps(summary["gates"]))
    print("screen_pass:", summary["screen_pass"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
