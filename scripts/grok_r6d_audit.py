"""Recompute published numbers from pairs and from raw trial rows."""
import json
from pathlib import Path

import numpy as np

from scarx.evaluation import r5b_independent as gen
from scarx.evaluation import r5b_probe as probe
from scarx.evaluation.r6d_reference import both_correct, seed_interval

R6A = Path(r"C:\Users\hachimi\Downloads\model agent\scar-x-deepseek-r6\runs\20261111T072852Z-r6-confirmation-v1-d3a510\result.json")
R6B = Path(r"C:\Users\hachimi\Downloads\model agent\scar-x-grok-r6\runs\grok-r6-causal-v1\result.json")
OUT = Path(__file__).resolve().parents[1] / "runs" / "grok-r6d-audit-v1"


def replay_one_r5b_seed(seed=98001):
    train_x, train_y, *_ = gen.independent_query_set_v1(seed + 200_000, 1024)
    pair_x, pair_y, _va, _vb, _q, _pid, flips = gen.independent_paired_set_v1(seed + 400_000, 256)
    tick = gen.locate_query_tick(train_x[0])
    captured = probe.capture(seed, train_x, query_tick=tick)
    train_f = probe.bilinear_query_map(probe.state_vector(captured, "post"), gen.query_cue(train_x, tick=tick))
    pair_cap = probe.capture(seed, pair_x, query_tick=gen.locate_query_tick(pair_x[0]))
    pair_f = probe.bilinear_query_map(probe.state_vector(pair_cap, "post"), gen.query_cue(pair_x, tick=gen.locate_query_tick(pair_x[0])))
    head = probe.train_head(train_f, train_y, seed=seed)
    predicted = head.predict(pair_f)
    return {
        "seed": seed,
        "both_A_from_predictions": both_correct(predicted, pair_y, flips),
        "prediction_sha256": __import__("hashlib").sha256(np.ascontiguousarray(predicted).tobytes()).hexdigest(),
        "oracle_matches": bool(np.array_equal(gen.oracle_all(pair_x), pair_y)),
        "query_tick": tick,
    }


def recompute_r6a(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    trials = payload["trials"]
    def rows(arm):
        return [row for row in trials if row.get("arm") == arm]
    matched = rows("matched-16384")
    g1_values = [row["both_A"] for row in matched]
    g1 = float(np.mean(g1_values)) >= 0.75
    gain = []
    for seed in sorted({row["seed"] for row in trials if row.get("arm") == "budget-16384"}):
        big = [row["direct"] for row in trials if row["seed"] == seed and row["arm"] == "budget-16384"][0]
        small = [row["direct"] for row in trials if row["seed"] == seed and row["arm"] == "budget-1024"][0]
        gain.append(big - small)
    g2_interval = seed_interval(gain)
    g2 = g2_interval["mean"] >= 0.05 and g2_interval["lower95"] > 0
    stress = [row for row in trials if row.get("condition") == "stress"]
    g6_values = [(row["seed"], row["arm"], row["direct"]) for row in stress]
    g6 = all(row["direct"] > 0.7419 for row in stress) if stress else False
    worst = min(stress, key=lambda row: row["direct"]) if stress else None
    held = {}
    for row in trials:
        if row.get("condition") != "ladder":
            continue
        held.setdefault(row["seed"], set()).add(row["digests"]["held"])
    return {
        "G1_recomputed": g1,
        "G1_mean_both_A": float(np.mean(g1_values)),
        "G2_recomputed": g2,
        "G2_interval": g2_interval,
        "G4_in_json_not_recomputed": payload["summary"]["gates"].get("G4_random_labels_chance"),
        "G5_in_json_not_recomputed": payload["summary"]["gates"].get("G5_matched_update_reported"),
        "G6_recomputed": g6,
        "G6_json": payload["summary"]["gates"].get("G6_stress_above_ceiling"),
        "confirmation_pass_json": payload["summary"].get("confirmation_pass"),
        "stress_worst": {"seed": worst["seed"], "arm": worst["arm"], "direct": worst["direct"]} if worst else None,
        "heldout_digest_unique_per_seed": {str(seed): len(values) for seed, values in held.items()},
        "published_boolean_agreement": {
            "G1": g1 == payload["summary"]["gates"]["G1_bothA_16384_ge_075"],
            "G2": g2 == payload["summary"]["gates"]["G2_gain_over_1024_ge_5pp"],
            "G6": g6 == payload["summary"]["gates"]["G6_stress_above_ceiling"],
        },
    }


def main():
    OUT.mkdir(parents=True)
    replay = replay_one_r5b_seed()
    stored = json.loads(R6B.read_text(encoding="utf-8"))
    stored_row = [row for row in stored["reproduction"] if row["seed"] == 98001][0]
    r6a = recompute_r6a(R6A)
    result = {
        "verdict": "INCONCLUSIVE",
        "r6b_seed_98001": replay,
        "r6b_stored_both_A": stored_row["both_A"],
        "r6b_prediction_recompute_matches_stored": abs(replay["both_A_from_predictions"] - stored_row["both_A"]) < 1e-12,
        "r6a_gates_recomputed_from_trials": r6a,
    }
    # The audit does not award PASS. R6A confirmation_pass is false on recomputed G6.
    if r6a["G6_recomputed"] is False and r6a["published_boolean_agreement"]["G6"]:
        result["r6a_confirmation"] = "STOP_RECOMPUTED"
    else:
        result["r6a_confirmation"] = "INCONCLUSIVE"
    (OUT / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
