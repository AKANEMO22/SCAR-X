"""Diagnostics missing from the locked screen. Does not change its gates.

Same generators and split seeds. Adds the ridge reader on the stress tasks,
a label-shuffle null, a paired distractor, and a delay time-shift.
"""
import json
import os
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np

from scarx.evaluation.temporal_stress import (
    accuracy,
    delay_set,
    fit_reader,
    make_core,
    order_pairs,
    order_with_distractors,
    paired_flip,
    query_set,
    rollout,
    sim_xor,
)


def features_of(core, sequences, substrate):
    built, _state = rollout(core, sequences)
    return built[substrate]


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "grok-r3b-temporal-stress-v1-careful"
SEEDS = (94011, 94017, 94023)
SUBSTRATES = ("decay025", "spikesum", "bank5")


def ridge_weight(train_f, train_y):
    mean = train_f.mean(0)
    scale = train_f.std(0) + 1e-4
    design = np.column_stack(((train_f - mean) / scale, np.ones(len(train_f))))
    target = np.eye(2, dtype=np.float64)[np.asarray(train_y, np.int32)]
    gram = design.T @ design + 0.01 * np.eye(design.shape[1])
    solved = np.linalg.solve(gram, design.T @ target)
    return mean, scale, solved


def ridge_predict(mean, scale, solved, test_f):
    design = np.column_stack(((test_f - mean) / scale, np.ones(len(test_f))))
    return np.argmax(design @ solved, axis=1).astype(np.int32)


def ridge_accuracy(train_f, train_y, test_f, test_y):
    predicted = ridge_predict(*ridge_weight(train_f, train_y), test_f)
    return float(np.mean(predicted == np.asarray(test_y))), predicted


def order_block(core, seed):
    early, early_y, late, late_y = order_pairs(94102, 128)
    train_x = np.concatenate((early, late))
    train_y = np.concatenate((early_y, late_y))
    test_a, labels_a, test_b, labels_b = order_pairs(94104, 128)
    test_x = np.concatenate((test_a, test_b))
    test_y = np.concatenate((labels_a, labels_b))
    shuffled = np.random.default_rng(19).permutation(train_y)
    out = {}
    for substrate in SUBSTRATES:
        trained = features_of(core, train_x, substrate)
        first = features_of(core, test_a, substrate)
        second = features_of(core, test_b, substrate)
        held = np.concatenate((first, second))
        head = fit_reader("norm", trained, train_y, seed)
        score, _predicted = accuracy(head, held, test_y)
        true_ridge, ridge_hat = ridge_accuracy(trained, train_y, held, test_y)
        null_ridge, _null_hat = ridge_accuracy(trained, shuffled, held, test_y)
        out[substrate] = {
            "head_accuracy": score,
            "head_paired_flip": paired_flip(head, first, labels_a, second, labels_b),
            "ridge": true_ridge,
            "ridge_paired_flip": paired_flip_from(ridge_hat, labels_a, labels_b),
            "ridge_label_shuffle": null_ridge,
            "feature_l1": float(np.mean(np.abs(first - second))),
        }
    return out


def paired_flip_from(predicted, labels_a, labels_b):
    half = len(labels_a)
    left, right = predicted[:half], predicted[half:]
    return float(np.mean((left == labels_a) & (right == labels_b)))


def query_block(core):
    train_x, train_y, _ignored = query_set(94102, 256)
    test_x, test_y, first = query_set(94104, 256)
    shuffled = np.random.default_rng(23).permutation(train_y)
    out = {"first_key_policy": float(np.mean(first == test_y)), "label_balance": float(np.mean(test_y))}
    for substrate in SUBSTRATES:
        trained = features_of(core, train_x, substrate)
        held = features_of(core, test_x, substrate)
        head = fit_reader("norm", trained, train_y, 94011)
        score, _predicted = accuracy(head, held, test_y)
        true_ridge, _hat = ridge_accuracy(trained, train_y, held, test_y)
        null_ridge, _null = ridge_accuracy(trained, shuffled, held, test_y)
        out[substrate] = {"head": score, "ridge": true_ridge, "ridge_label_shuffle": null_ridge}
    return out


def delay_shift(core):
    test_x, test_y = delay_set(94104, 256, 6)
    moved = test_x.copy()
    pulse = moved[:, 1, :2].copy()
    moved[:, 1, :2] = 0
    moved[:, 8, :2] = pulse
    train_x, train_y = delay_set(94102, 256, 6)
    out = {}
    for substrate in ("decay025", "spikesum"):
        trained = features_of(core, train_x, substrate)
        head = fit_reader("norm", trained, train_y, 94011)
        original, _predicted = accuracy(head, features_of(core, test_x, substrate), test_y)
        shifted, _shifted_hat = accuracy(head, features_of(core, moved, substrate), test_y)
        ridge, _hat = ridge_accuracy(trained, train_y, features_of(core, test_x, substrate), test_y)
        out[substrate] = {"head_original": original, "head_pulse_moved_to_tick8": shifted, "ridge_original": ridge}
    return out


def paired_distractor(core):
    base, labels = order_with_distractors(94104, 128, 0)
    out = {}
    train_x, train_y = order_with_distractors(94102, 256, 0)
    for substrate in ("spikesum", "decay025"):
        head = fit_reader("norm", features_of(core, train_x, substrate), train_y, 94011)
        scores = {}
        for extras, tick, channel in ((0, None, None), (1, 4, 4), (2, 5, 5)):
            sequences = base.copy()
            if extras >= 1:
                sequences[:, 4, 4] += 1.5
            if extras == 2:
                sequences[:, 5, 5] += 1.5
            score, _predicted = accuracy(head, features_of(core, sequences, substrate), labels)
            scores[str(extras)] = score
        out[substrate] = scores
    return out


def random_label_null(core):
    train_x, train_y = sim_xor(94102, 256)
    test_x, test_y = sim_xor(94104, 256)
    trained = features_of(core, train_x, "spikesum")
    held = features_of(core, test_x, "spikesum")
    scores = []
    majority_hits = []
    for draw in range(32):
        shuffled = np.random.default_rng(100 + draw).permutation(train_y)
        head = fit_reader("norm", trained, shuffled, 94011)
        score, predicted = accuracy(head, held, test_y)
        scores.append(score)
        majority = int(np.round(np.mean(predicted)))
        majority_hits.append(float(np.mean(predicted == majority)))
    values = np.asarray(scores)
    return {
        "test_label_fraction_one": float(np.mean(test_y)),
        "majority_class_accuracy": float(max(np.mean(test_y), 1.0 - np.mean(test_y))),
        "shuffle_mean": float(values.mean()),
        "shuffle_min": float(values.min()),
        "shuffle_max": float(values.max()),
        "fraction_of_shuffles_at_or_above_0_60": float(np.mean(values >= 0.60)),
        "mean_prediction_majority_fraction": float(np.mean(majority_hits)),
        "original_gate_draw_is_one_cell": True,
    }


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    payload = {
        "does_not_change_screen_gates": True,
        "previous_result": "runs/grok-r3b-temporal-stress-v1/result.json",
        "order_by_seed": {},
    }
    for seed in SEEDS:
        core = make_core(seed)
        block = order_block(core, seed)
        payload["order_by_seed"][str(seed)] = block
        if seed == 94011:
            payload["query"] = query_block(core)
            payload["delay_time_shift"] = delay_shift(core)
            payload["paired_distractor"] = paired_distractor(core)
            payload["random_label_null"] = random_label_null(core)
    OUT.mkdir(parents=True)
    (OUT / "result.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "order_94011": payload["order_by_seed"]["94011"],
        "query": payload["query"],
        "delay_time_shift": payload["delay_time_shift"],
        "paired_distractor": payload["paired_distractor"],
        "random_label_null": payload["random_label_null"],
    }, indent=2))


if __name__ == "__main__":
    main()
