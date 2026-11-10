"""Registered experiment runner for the MTB multi-timescale readout study.

Protocol: experiments/collab/scarx-neural-round1-20261111/antigravity/protocol-v1.md

Two phases:
  --phase screen        dev only, 3 fresh seeds, 11 conditions x 3 tasks
  --phase confirmation  only if the screen passed; 8 fresh seeds, no tuning

Every number is written under runs/<unique-run-id>/ with the full manifest.
No shared latest.json; prior runs are never rewritten.
"""
import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scarx.evaluation.binding_audit21 import strict_binding  # noqa: E402
from scarx.evaluation.multi_timescale import (  # noqa: E402
    DECAY_BANK, MTCore, make_dimension_matched_projection,
)
from scarx.evaluation.phase21 import holm, interval, signflip  # noqa: E402
from scarx.evaluation.runs import new_run, write_json  # noqa: E402
from scarx.evaluation.tasks21 import representation, ridge_probe  # noqa: E402
from scarx.neural.core import Config  # noqa: E402
from scarx.runtime.resources import ResourceMonitor  # noqa: E402

TASKS = ("order", "xor", "strict_binding")
DEV_SEEDS = (21001, 21007, 21013)
CONFIRM_SEEDS = (22003, 22009, 22013, 22019, 22021, 22027, 22031, 22037)

# protocol section 5: identical split offsets to scripts/run_p1_paired_score.py
CAL, TRAIN, FIT, TEST, OOD = 6_200_000, 6_300_000, 6_400_000, 6_500_000, 6_600_000
N_TRAIN = 2048
N_CAL = 128
N_FIT = 128
N_TEST_DEV = 256
N_TEST_CONF = 512
LEAK, THRESH, FEAT_DECAY, ADAPT = .65, .5, .25, .02
HIDDEN, FAN_IN, ROWS = 48, 8, 12
READOUT_LR_GRID = (0.025, 0.05, 0.10)


def dataset(task, seed, count, ood=False):
    if task == "strict_binding":
        return (None, None) if ood else strict_binding(seed, count)
    return representation(task, seed, count, ood)


def make_config(seed, readout_lr):
    return Config(inputs=8, hidden=HIDDEN, outputs=2, fan_in=FAN_IN, plastic_rows=ROWS,
                  seed=seed, leak=LEAK, threshold=THRESH, feature_decay=FEAT_DECAY,
                  adaptation_strength=ADAPT, readout_lr=readout_lr,
                  hidden_lr=.002, anchor_strength=0.)


def build(condition, seed, readout_lr):
    """Construct one arm. Only the readout basis / hidden-learning switch varies."""
    cfg = make_config(seed, readout_lr)
    wide = len(DECAY_BANK) * HIDDEN
    if condition == "prod":
        return MTCore(cfg, decays=(FEAT_DECAY,), rule="score")
    if condition == "mtb":
        return MTCore(cfg, decays=DECAY_BANK, rule="score")
    if condition == "prod_x240":
        return MTCore(cfg, decays=(FEAT_DECAY,), rule="score",
                      readout_projection=make_dimension_matched_projection(seed, HIDDEN, wide))
    if condition == "mtb_one_decay":
        return MTCore(cfg, decays=(FEAT_DECAY,) * len(DECAY_BANK), rule="score")
    if condition == "prod_frozen":
        return MTCore(cfg, decays=(FEAT_DECAY,), rule="frozen")
    if condition == "mtb_frozen":
        return MTCore(cfg, decays=DECAY_BANK, rule="frozen")
    if condition == "prod_readout":
        return MTCore(cfg, decays=(FEAT_DECAY,), rule="readout")
    if condition == "mtb_readout":
        return MTCore(cfg, decays=DECAY_BANK, rule="readout")
    if condition == "mtb_shuffle":
        return MTCore(cfg, decays=DECAY_BANK, rule="score", shuffle=True)
    if condition == "mtb_only_fast":
        return MTCore(cfg, decays=(.10,), rule="score")
    if condition == "mtb_only_slow":
        return MTCore(cfg, decays=(.90,), rule="score")
    raise ValueError(condition)


CONDITIONS = ("prod", "mtb", "prod_x240", "mtb_one_decay", "prod_frozen",
              "mtb_frozen", "prod_readout", "mtb_readout", "mtb_shuffle",
              "mtb_only_fast", "mtb_only_slow")


def train(model, x, y):
    losses = []
    for seq, label in zip(x, y):
        model.forward(seq, track=True)
        losses.append(model.learn(int(label)))
    return losses


def evaluate(model, task, seed, phase):
    """Score on the registered split. Ridge probe refits a head from scratch."""
    n_test = N_TEST_DEV if phase == "screen" else N_TEST_CONF
    fx, fy = dataset(task, seed + FIT, N_FIT)
    tx, ty = dataset(task, seed + TEST, n_test)

    preds = model.predict_many(tx, 903)
    direct = float(np.mean(preds == ty))

    # hidden representation: same fit/test split, same algorithm, same budget
    feats = model.features_many(fx, 902)
    tfeats = model.features_many(tx, 903)
    probe = ridge_probe(feats, fy, tfeats, ty)

    row = {"direct": direct, "probe": probe,
           "n_test": int(len(ty)), "n_fit": int(len(fy))}
    # initial-hidden probe for the gain: a freshly seeded model, same config
    init = MTCore(make_config(seed, model.config.readout_lr), decays=model.decays,
                  rule=model.rule)

    probe_init = ridge_probe(init.features_many(fx, 902), fy, init.features_many(tx, 903), ty)
    row["probe_initial"] = probe_init
    row["probe_gain"] = probe - probe_init
    return row


def trial(job):
    phase, task, seed, condition, readout_lr = job
    started = time.perf_counter()
    with ResourceMonitor(interval=.1) as monitor:
        model = build(condition, seed, readout_lr)
        x, y = dataset(task, seed + TRAIN, N_TRAIN)
        loss_before = float(model.probabilities()[1])
        losses = train(model, x, y)
        spike_hash = model.spike_hash(x[:32], 991)
        sc = evaluate(model, task, seed, phase)
        monitor.check()
    elapsed = time.perf_counter() - started
    row = {
        "phase": phase, "task": task, "seed": seed, "condition": condition,
        "readout_lr": readout_lr,
        "direct": sc["direct"], "probe": sc["probe"],
        "probe_initial": sc["probe_initial"], "probe_gain": sc["probe_gain"],
        "n_test": sc["n_test"],
        "mean_loss": float(np.mean(losses)),
        "first_loss": float(losses[0]), "last_loss": float(losses[-1]),
        "train_seconds": elapsed,
        "spike_hash": spike_hash,
        "readout_width": int(model.readout_projection.shape[1]
                             if model.readout_projection is not None else model.substrate_width),
        "updates": int(model.updates),
        "array_bytes": int(model.memory_bytes()),
        "resources": monitor.report(),
    }
    print(json.dumps({k: row[k] for k in ("task", "condition", "seed", "readout_lr",
                                          "direct", "probe_gain", "readout_width")}), flush=True)
    return row


def paired(rows, a, b, metric):
    """Paired seed-level contrast a-b, each arm at its OWN locked lr.

    Pairing on lr would be wrong here: the per-arm lr lock legitimately differs,
    so requiring a shared lr silently produced no pairs and a None contrast.
    What must match is the seed (the model initialisation), which is the actual
    unit of independence.
    """
    by = {}
    for r in rows:
        by[(r["task"], r["seed"], r["condition"])] = r
    diffs = []
    for task in TASKS:
        for seed in sorted({r["seed"] for r in rows}):
            ra, rb = by.get((task, seed, a)), by.get((task, seed, b))
            if ra and rb:
                diffs.append(ra[metric] - rb[metric])
    if not diffs:
        return None
    return interval(diffs)


def select_lr_per_arm(rows):
    """Dev-only hyperparameter selection: best mean direct accuracy per arm.

    Locked before confirmation; confirmation never re-tunes. Ties break toward
    the production value (0.025 is not in this grid; lowest lr wins) so the
    choice is deterministic and generous to the controls, not the candidate.
    """
    locked = {}
    detail = {}
    for cond in CONDITIONS:
        arm = [r for r in rows if r["condition"] == cond]
        if not arm:
            continue
        scores = {}
        for lr in sorted({r["readout_lr"] for r in arm}):
            vals = [r["direct"] for r in arm if r["readout_lr"] == lr]
            scores[lr] = float(np.mean(vals))
        best = max(scores.items(), key=lambda kv: (round(kv[1], 6), -kv[0]))[0]
        locked[cond] = best
        detail[cond] = scores
    return locked, detail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("screen", "confirmation"), default="screen")
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    args = ap.parse_args()
    root = Path(args.root)
    phase = args.phase
    seeds = DEV_SEEDS if phase == "screen" else CONFIRM_SEEDS

    protocol = f"collab-mtb-{phase}-v1"
    config = {
        "hypothesis": "MTB multi-timescale readout receptive field",
        "conditions": list(CONDITIONS),
        "tasks": list(TASKS),
        "seeds": list(seeds),
        "readout_lr_grid": list(READOUT_LR_GRID),
        "hidden": HIDDEN, "fan_in": FAN_IN, "plastic_rows": ROWS,
        "n_train": N_TRAIN, "n_cal": N_CAL, "n_fit": N_FIT,
        "n_test": N_TEST_DEV if phase == "screen" else N_TEST_CONF,
        "leak": LEAK, "threshold": THRESH, "feature_decay": FEAT_DECAY,
        "adaptation_strength": ADAPT,
        "decay_bank": list(DECAY_BANK),
        "split_offsets": {"cal": CAL, "train": TRAIN, "fit": FIT, "test": TEST, "ood": OOD},
        "gate_acquisition": .65, "gate_probe_gain_pp": 5.0,
        "phase": phase,
    }
    run = new_run(root, protocol, config, data_description=(
        "Repository-authored synthetic generators only: "
        "tasks21.representation(order,xor) and binding_audit21.strict_binding. "
        f"Split seed offsets {config['split_offsets']}. No downloaded, private or "
        "model-generated data; no Qwen output used as a label."))
    print(f"RUN {run}", flush=True)

    jobs = list(itertools.product((phase,), TASKS, seeds, CONDITIONS, READOUT_LR_GRID))
    # Protocol §7: readout_lr is screened on dev over the full grid for EVERY arm
    # with the same rule; one winner per arm is locked before confirmation.
    locked, lr_detail = {}, {}
    if phase == "confirmation":
        lock = json.loads((root / "artifacts" / "antigravity-r1" /
                           "locked-readout-lr.json").read_text())
        locked = lock["per_arm"]
        jobs = [j for j in jobs if j[4] == locked[j[3]]]

    print(f"{len(jobs)} trials", flush=True)
    rows = []
    with ResourceMonitor(interval=.2) as monitor:
        for i, job in enumerate(jobs, 1):
            rows.append(trial(job))
            if i % 10 == 0:
                monitor.check()
                write_json(run / "progress.json", rows)
                print(f"  {i}/{len(jobs)} done", flush=True)

    # In the screen phase the lr is chosen HERE, on development only, from the
    # completed trials; confirmation reads the lock file and never re-tunes.
    if phase == "screen" and not locked:
        locked, lr_detail = select_lr_per_arm(rows)
    summary = summarise(rows, phase, locked)
    result = {
        "protocol": protocol, "phase": phase, "config": config,
        "trials": rows, "summary": summary,
        "lr_selection": lr_detail,
        "resources": monitor.report(),
        "scope": "Registered synthetic tasks at H=48. No chat/vision/reasoning claim.",
    }
    write_json(run / "result.json", result)
    (run / "report.md").write_text(render_report(result), encoding="utf-8")
    if phase == "screen":
        # Lock the dev-selected lr per arm BEFORE confirmation runs. Confirmation
        # reads this file and never re-tunes.
        lock_path = root / "artifacts" / "antigravity-r1" / "locked-readout-lr.json"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(lock_path, {
            "per_arm": locked, "selected_on": "development seeds only",
            "selection_metric": "mean dev_direct across 3 tasks x 3 dev seeds",
            "screen_run": run.name, "screen_pass": summary["screen_pass"],
            "note": "Written before any confirmation trial. Never re-tuned after.",
        })
        print(f"LOCKED lr per arm -> {lock_path}", flush=True)
    print(f"RESULT {run}", flush=True)
    return 0


def summarise(rows, phase, locked):
    """Score each arm at its own locked lr; contrasts pair on the same lr."""
    sel = [r for r in rows if r["readout_lr"] == locked.get(r["condition"])]
    by = {}
    for r in sel:
        by.setdefault((r["task"], r["condition"]), []).append(r)

    def stat(task, cond, key):
        vals = [r[key] for r in by.get((task, cond), [])]
        return interval(vals) if vals else None

    out = {"per_task": {}, "contrasts": {}, "spike_identity": {},
           "locked_readout_lr": locked, "scored_rows": len(sel)}
    for task in TASKS:
        out["per_task"][task] = {
            cond: {
                "direct": stat(task, cond, "direct"),
                "probe_gain": stat(task, cond, "probe_gain"),
                "probe": stat(task, cond, "probe"),
            }
            for cond in CONDITIONS if (task, cond) in by
        }

    for name, (a, b) in {
        "mtb_minus_prod_x240": ("mtb", "prod_x240"),
        "mtb_minus_mtb_one_decay": ("mtb", "mtb_one_decay"),
        "mtb_minus_prod": ("mtb", "prod"),
        "mtb_minus_mtb_shuffle": ("mtb", "mtb_shuffle"),
        "mtb_minus_mtb_frozen": ("mtb", "mtb_frozen"),
        "mtb_minus_mtb_readout": ("mtb", "mtb_readout"),
        "mtb_frozen_minus_prod_frozen": ("mtb_frozen", "prod_frozen"),
        "prod_minus_prod_frozen": ("prod", "prod_frozen"),
        "mtb_minus_mtb_only_slow": ("mtb", "mtb_only_slow"),
    }.items():
        out["contrasts"][name] = {
            "direct": paired(sel, a, b, "direct"),
            "probe_gain": paired(sel, a, b, "probe_gain"),
        }

    for task in TASKS:
        hashes = {}
        for r in sel:
            if r["task"] != task:
                continue
            hashes.setdefault(r["seed"], set()).add(r["spike_hash"])
        out["spike_identity"][task] = {str(s): len(h) for s, h in hashes.items()}
    # NOTE: this hash is taken AFTER each arm trained, so arms whose hidden
    # plasticity differs (score vs frozen vs shuffle) are SUPPOSED to diverge
    # here. Identity is only expected among arms that share the hidden rule.
    # The genuine readout-only claim is proven in P0 on identical weights; here
    # we report divergence per rule-group instead of asserting one hash.
    out["spike_identity_all_equal"] = all(
        v == 1 for d in out["spike_identity"].values() for v in d.values())
    by_rule = {"learning_score": ("mtb", "prod", "prod_x240", "mtb_one_decay",
                                  "mtb_only_fast", "mtb_only_slow"),
               "frozen": ("prod_frozen", "mtb_frozen"),
               "readout_only": ("prod_readout", "mtb_readout"),
               "shuffled_signal": ("mtb_shuffle",)}
    out["spike_identity_within_rule_group"] = {}
    for group, arms in by_rule.items():
        ok = True
        for task in TASKS:
            for seed in sorted({r["seed"] for r in sel}):
                hs = {r["spike_hash"] for r in sel
                      if r["task"] == task and r["seed"] == seed and r["condition"] in arms}
                if len(hs) > 1:
                    ok = False
        out["spike_identity_within_rule_group"][group] = ok

    def lower(contrast, key="direct"):
        c = out["contrasts"][contrast][key]
        return None if c is None else c["lower95"]

    g5, g6, g7 = (lower("mtb_minus_prod_x240"), lower("mtb_minus_mtb_one_decay"),
                  lower("mtb_minus_mtb_shuffle"))
    out["screen"] = {
        "G5_mtb_beats_dim_matched": None if g5 is None else bool(g5 > 0),
        "G6_mtb_beats_single_timescale": None if g6 is None else bool(g6 > 0),
        "G7_signal_matters": None if g7 is None else bool(g7 > 0),
        "spike_identity_holds": out["spike_identity_all_equal"],
        # The readout-only attribution does not depend on spikes being equal
        # after training; it is proven in P0 on identical weights. What matters
        # here is that arms sharing a hidden rule produce identical spikes, i.e.
        # the readout change alone does not perturb the neuron.
        "spike_identity_within_rule_groups": out["spike_identity_within_rule_group"],        "mtb_frozen_beats_prod_frozen":
            (lambda c: None if c is None else bool(c["lower95"] > 0))(
                out["contrasts"]["mtb_frozen_minus_prod_frozen"]["direct"]),
    }
    # A gate whose contrast could not be computed is UNKNOWN, not passed. If any
    # gate is None the screen has not passed; `all()` over an empty/None set would
    # otherwise report a vacuous True, which is exactly the kind of silent success
    # this campaign forbids.
    required = list(out["screen"].values())
    out["screen_gates_unknown"] = [k for k, v in out["screen"].items() if v is None]
    out["screen_pass"] = bool(all(v is True for v in required))
    return out


def _fmt(interval_dict, scale=100.0, suffix="pp"):
    if interval_dict is None:
        return "—"
    return (f"{interval_dict['mean']*scale:+.2f}{suffix} "
            f"[{interval_dict['lower95']*scale:+.2f}, {interval_dict['upper95']*scale:+.2f}]")


def render_report(result):
    s = result["summary"]
    lines = [f"# MTB {result['phase']} run", "",
             f"Scope: {result['scope']}", "",
             "## Per-task acquisition (dev_direct, %)", "",
             "| condition | " + " | ".join(TASKS) + " |",
             "|---|" + "---|" * len(TASKS)]
    for cond in CONDITIONS:
        cells = []
        for task in TASKS:
            st = s["per_task"].get(task, {}).get(cond, {}).get("direct")
            cells.append("—" if st is None else f"{st['mean']*100:.2f}")
        lines.append(f"| {cond} | " + " | ".join(cells) + " |")
    lines += ["", "## Attribution contrasts (direct accuracy, pp)", "",
              "| contrast | mean [95% CI] |", "|---|---|"]
    for name, c in s["contrasts"].items():
        lines.append(f"| {name} | {_fmt(c['direct'])} |")
    lines += ["", "## Screen gates", "", "```json",
              json.dumps(s["screen"], indent=2), "```", "",
              f"**screen_pass: {s['screen_pass']}**", "",
              "Spike-train identity across arms: "
              f"{'HOLDS' if s['spike_identity_all_equal'] else 'VIOLATED'}", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
