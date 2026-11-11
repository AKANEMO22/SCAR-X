"""Worker process for isolated DC-ALIF P1 trial, cold-process resume check, and P2 retention trial."""
from __future__ import annotations

import os

for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np

from direct_eligibility_common import CONFIRM, DEV, TASKS, dataset
from scarx.evaluation.dc_alif_eligibility import (
    DCALIFBPTTControl,
    DCALIFEligibility,
    dc_alif_finite_difference_check,
)
from scarx.evaluation.paired_retention import (
    component_intervention,
    mixed_stream,
    retention_metrics,
    ridge_multiclass,
)
from scarx.evaluation.tasks21 import context_task, ridge_probe
from scarx.learning.replay import ReplayBuffer
from scarx.neural.core import Config
from scarx.runtime.limits import apply_memory_limit
from scarx.runtime.resources import ResourceMonitor
from visual_word_control_common import atomic

METHODS = ("correct", "no_gate", "shuffled", "zero", "frozen", "bptt_control")
P2_SEEDS = (15217, 15227, 15241)
P2_CONDITIONS = ("dc_alif_reservoir_usefulness", "dc_alif_reservoir_unprotected", "dc_alif_frozen_reservoir")


def make_model(task: str, seed: int, method: str) -> DCALIFEligibility | DCALIFBPTTControl:
    if method not in METHODS:
        raise ValueError(f"Unregistered DC-ALIF method: {method}")
    c = Config(inputs=8, outputs=2, hidden=48, fan_in=8, plastic_rows=12, seed=seed, anchor_strength=0)
    cond = "correct" if method == "bptt_control" else method
    m = DCALIFEligibility(c, cond)
    cal, _ = dataset(task, seed + 1360000, 128)
    m.calibrate(cal)
    if method == "bptt_control":
        return DCALIFBPTTControl(m)
    return m


def load_model(method: str, payload: bytes) -> DCALIFEligibility | DCALIFBPTTControl:
    if method == "bptt_control":
        return DCALIFBPTTControl.loads(payload)
    return DCALIFEligibility.loads(payload)


def reference_model(model: DCALIFEligibility | DCALIFBPTTControl) -> DCALIFEligibility:
    return getattr(model, "reference", model)


def run_p1_trial(job: tuple[str, str, int, str, str, str]) -> dict[str, object]:
    phase, task, seed, method, runpath, artpath = job
    if (
        phase not in ("development", "confirmation")
        or task not in TASKS
        or method not in METHODS
        or seed not in (DEV if phase == "development" else CONFIRM)
    ):
        raise ValueError("Unregistered DC-ALIF P1 trial")
    run = Path(runpath)
    artifact_dir = Path(artpath) / "checkpoints"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    cap = apply_memory_limit()
    began = time.perf_counter()
    if method == "bptt_control":
        import torch

        torch.set_num_threads(1)

    identifier = f"{phase}-{task}-{seed}-{method}"
    m = make_model(task, seed, method)
    ref = reference_model(m)
    initial_payload = m.dumps()
    initial = {n: getattr(ref.core, n).copy() for n in ("w_in", "w_rec", "w_out")}

    train, labels = dataset(task, seed + 1370000, 1024)
    losses: list[float] = []
    ticks_before = ref.total_ticks

    with ResourceMonitor(interval=0.1) as monitor:
        for i, (seq, label) in enumerate(zip(train, labels), 1):
            losses.append(m.learn(seq, int(label)))
            if i % 256 == 0:
                monitor.check()
            if time.perf_counter() - began > 180:
                raise TimeoutError("Registered 180s DC-ALIF trial cap exceeded")

        train_seconds = time.perf_counter() - began
        training_ticks = ref.total_ticks - ticks_before
        assert training_ticks == 12288 and ref.core.updates == 1024 and ref.head.updates == 1024

        payload = m.dumps()
        checkpoint = artifact_dir / f"{identifier}.dc_alif.zip"
        checkpoint.write_bytes(payload)

        view = load_model(method, payload)
        v = reference_model(view)
        original = reference_model(load_model(method, initial_payload))

        fit, fy = dataset(task, seed + 1380000, 128)
        test_count = 256 if phase == "development" else 512
        test, ty = dataset(task, seed + 1390000, test_count)

        ff = view.features_many(fit)
        tf = view.features_many(test)
        probabilities = v.head.probabilities(tf, v.output_mask)
        pred = probabilities.argmax(1)

        probe = ridge_probe(ff, fy, tf, ty)
        initial_probe = ridge_probe(original.features_many(fit), fy, original.features_many(test), ty)

        rng = np.random.default_rng(seed + 1410000)
        permuted = np.stack([seq[rng.permutation(len(seq))] for seq in test])
        time_permuted = ridge_probe(ff, fy, view.features_many(permuted), ty)
        state_reset = ridge_probe(ff, fy, view.features_many(test, True), ty)

        ood, oy = dataset(task, seed + 1400000, 128, True)
        ood_probe = ridge_probe(ff, fy, view.features_many(ood), oy) if ood is not None else None
        bag = ridge_probe(fit.sum(1), fy, test.sum(1), ty)

        assert m.dumps() == payload
        duplicate = load_model(method, payload)
        loss_next = duplicate.learn(train[0], int(labels[0]))
        reply_next = duplicate.predict(test[0])
        draw_next = int(reference_model(duplicate).signal_rng.integers(10000))
        cold_expected = {
            "loss": loss_next,
            "prediction": reply_next,
            "signal_rng_draw": draw_next,
            "full_state_sha256": hashlib.sha256(duplicate.dumps()).hexdigest(),
        }

        row: dict[str, object] = {
            "phase": phase,
            "task": task,
            "seed": seed,
            "method": method,
            "direct": float(np.mean(pred == ty)),
            "probe": probe,
            "initial_probe": initial_probe,
            "probe_gain": probe - initial_probe,
            "probe_time_permuted": time_permuted,
            "probe_state_reset": state_reset,
            "ood_probe": ood_probe,
            "bag_only": bag,
            "experiences": 1024,
            "head_updates": ref.core.updates,
            "hidden_updates": ref.hidden_updates,
            "training_ticks": training_ticks,
            "array_bytes": ref.memory_bytes(),
            "checkpoint_bytes": len(payload),
            "hidden_delta_in": float(np.linalg.norm(ref.core.w_in - initial["w_in"])),
            "hidden_delta_rec": float(np.linalg.norm(ref.core.w_rec - initial["w_rec"])),
            "mean_training_loss": float(np.mean(losses)),
            "train_seconds": train_seconds,
            "seconds": time.perf_counter() - began,
            "test_inputs_sha256": hashlib.sha256(test.tobytes()).hexdigest(),
            "test_targets_sha256": hashlib.sha256(ty.astype(np.int32).tobytes()).hexdigest(),
            "checkpoint": checkpoint.relative_to(ROOT).as_posix(),
            "checkpoint_sha256": hashlib.sha256(payload).hexdigest(),
            "cold_expected": cold_expected,
            "core_weight_fingerprint": ref.core.fingerprint(),
            "head_sha256": hashlib.sha256(ref.head.dumps()).hexdigest(),
            "resources": monitor.report(),
            "cap": cap,
            "candidate_network_backward": False,
            "network_backward_in_this_trial": method == "bptt_control",
            "control_only": method == "bptt_control",
            "proposal30_approved": True,
        }
        if method == "bptt_control":
            assert isinstance(m, DCALIFBPTTControl)
            row.update(
                backward_passes=m.backward_passes,
                surrogate_forward_ticks=m.surrogate_forward_ticks,
                max_graph_state_error=m.max_graph_state_error,
                max_graph_feature_error=m.max_graph_feature_error,
            )
        monitor.check()
        atomic(run / f"{identifier}.json", {"record": row})
        return row


def run_p2_trial(job: tuple[int, str, str, str]) -> dict[str, object]:
    seed, condition, runpath, artpath = job
    if seed not in P2_SEEDS or condition not in P2_CONDITIONS:
        raise ValueError("Unregistered DC-ALIF P2 trial")
    run = Path(runpath)
    artifact_dir = Path(artpath) / "p2_checkpoints"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    cap = apply_memory_limit()
    began = time.perf_counter()

    frozen = "frozen" in condition
    usefulness_beta = 4.0 if "usefulness" in condition else 0.0
    cond_mode = "frozen" if frozen else "correct"

    c = Config(inputs=24, outputs=4, hidden=48, fan_in=8, plastic_rows=12, seed=seed, anchor_strength=0)
    m = DCALIFEligibility(c, cond_mode, usefulness_beta=usefulness_beta)
    calibration, _ = context_task(seed + 7200000, 128, 0, 20)
    m.calibrate(calibration)

    x, y, groups = mixed_stream(seed)
    stream_sha = hashlib.sha256(x.tobytes() + y.tobytes()).hexdigest()
    tests = [context_task(seed + 7500000 + 131 * j, 128, j, 20) for j in range(20)]
    buffer = ReplayBuffer(128, 12, 24, "reservoir", seed + 7700000)

    losses: list[float] = []
    scores: list[list[float]] = []
    interventions: list[dict[str, float]] = []
    stored_features: list[np.ndarray] | None = None
    previous_head = (m.head.weights.copy(), m.head.bias.copy())
    head_scale = np.maximum(np.sqrt(m.head.m2 / (m.head.count - 1)), 0.05).astype(np.float32)
    head_mean = m.head.mean.astype(np.float32)

    with ResourceMonitor(interval=0.2) as monitor:
        for i, (sequence, label) in enumerate(zip(x, y)):
            losses.append(m.learn(sequence, int(label)))
            buffer.add(sequence, int(label), 0)
            if (i + 1) % 2 == 0:
                item = buffer.sample(1)
                old_x, old_y = item[0] if item else (sequence, int(label))
                losses.append(m.learn(old_x, int(old_y)))
            if (i + 1) % 192 == 0:
                window = i // 192
                new_features = [m.features_many(tx) for tx, _ in tests]
                window_scores = [
                    float(np.mean(m.head.probabilities(f, m.output_mask).argmax(1) == ty))
                    for f, (_, ty) in zip(new_features, tests)
                ]
                scores.append(window_scores)
                current_head = (m.head.weights.copy(), m.head.bias.copy())
                if window and stored_features is not None:
                    labels_prev = np.concatenate([tests[j][1] for j in range(window)])
                    interventions.append(
                        {
                            "window": window + 1,
                            **component_intervention(
                                np.concatenate(stored_features[:window]),
                                np.concatenate(new_features[:window]),
                                labels_prev,
                                previous_head,
                                current_head,
                                head_mean,
                                head_scale,
                            ),
                        }
                    )
                stored_features = new_features
                previous_head = current_head
                monitor.check()
                if time.perf_counter() - began > 180:
                    raise TimeoutError("Registered 180s DC-ALIF P2 trial limit exceeded")

        assert stored_features is not None
        fit_feats = m.features_many(x[:1024])
        probe, _ = ridge_multiclass(
            fit_feats,
            y[:1024],
            np.concatenate(stored_features),
            np.concatenate([t[1] for t in tests]),
        )
        ckpt_path = artifact_dir / f"p2-{seed}-{condition}.dc_alif.zip"
        payload = m.dumps()
        ckpt_path.write_bytes(payload)

        # Verify exact resume with ReplayBuffer
        a = DCALIFEligibility.loads(payload)
        b = DCALIFEligibility.loads(ckpt_path.read_bytes())
        assert a.learn(x[0], int(y[0])) == b.learn(x[0], int(y[0])) and a.dumps() == b.dumps()

        metrics = retention_metrics(scores)
        row: dict[str, object] = {
            "seed": seed,
            "condition": condition,
            "usefulness_beta": usefulness_beta,
            "stream_sha256": stream_sha,
            "joint_refit_probe": probe,
            "updates": m.core.updates,
            "hidden_updates": m.hidden_updates,
            "mean_loss": float(np.mean(losses)),
            "seconds": time.perf_counter() - began,
            "mean_head_only_drop": float(np.mean([v["head_only_drop"] for v in interventions])) if interventions else 0.0,
            "mean_hidden_only_drop": float(np.mean([v["hidden_only_drop"] for v in interventions])) if interventions else 0.0,
            "mean_interaction_drop": float(np.mean([v["interaction_drop"] for v in interventions])) if interventions else 0.0,
            "array_bytes": m.memory_bytes(),
            "checkpoint": ckpt_path.relative_to(ROOT).as_posix(),
            "checkpoint_sha256": hashlib.sha256(payload).hexdigest(),
            "resume_exact": True,
            "resources": monitor.report(),
            "cap": cap,
            **metrics,
        }
        atomic(run / f"p2-{seed}-{condition}.json", {"record": row})
        return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    args = parser.parse_args()
    path = Path(args.job).resolve()
    job = json.loads(path.read_text(encoding="utf-8"))
    mode = job.get("mode", "p1")
    if mode == "cold":
        apply_memory_limit()
        phase, task, seed, method, runpath, artpath = job["trial"]
        if method == "bptt_control":
            import torch

            torch.set_num_threads(1)
        checkpoint = (ROOT / job["checkpoint"]).resolve()
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != job["sha256"]:
            raise ValueError("Cold checkpoint SHA256 mismatch")
        m = load_model(method, checkpoint.read_bytes())
        train, y = dataset(task, seed + 1370000, 1024)
        test, _ = dataset(task, seed + 1390000, 256 if phase == "development" else 512)
        loss = m.learn(train[0], int(y[0]))
        reply = m.predict(test[0])
        draw = int(reference_model(m).signal_rng.integers(10000))
        run = Path(runpath).resolve()
        atomic(
            run / f"cold-{phase}-{task}-{seed}-{method}.json",
            {
                "loss": loss,
                "prediction": reply,
                "signal_rng_draw": draw,
                "full_state_sha256": hashlib.sha256(m.dumps()).hexdigest(),
            },
        )
        return
    if mode == "p2":
        row = run_p2_trial(tuple(job["trial"]))
        print(
            json.dumps(
                {
                    "mode": "p2",
                    "seed": row["seed"],
                    "condition": row["condition"],
                    "acquisition": row["acquisition"],
                    "final": row["final"],
                    "mean_forgetting": row["mean_forgetting"],
                }
            ),
            flush=True,
        )
        return

    row = run_p1_trial(tuple(job["trial"]))
    print(
        json.dumps(
            {
                "mode": "p1",
                "phase": row["phase"],
                "task": row["task"],
                "seed": row["seed"],
                "method": row["method"],
                "direct": row["direct"],
                "probe": row["probe"],
                "probe_gain": row["probe_gain"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
