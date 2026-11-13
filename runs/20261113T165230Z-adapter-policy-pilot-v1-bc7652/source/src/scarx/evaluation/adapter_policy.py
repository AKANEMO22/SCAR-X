"""Isolated ALIF lifecycle teaching; never relabel the serving action head."""

import hashlib
import json
import math
import re
import numpy as np
from ..hybrid.learner import LocalALIF, features

ACTIONS = ("base", "memory", "repair", "verify", "reuse_adapter", "propose_adapter")
FIELDS = (
    "pipeline_fault",
    "verified",
    "ephemeral",
    "memory_available",
    "base_accuracy",
    "adapter_present",
    "compatible",
    "qualified",
    "adapter_gain_pp",
    "headroom_mb",
    "failure_count",
    "verified_examples",
    "training_affordable",
    "idle",
    "regression_pp",
    "provenance",
)
SCALES = (1, 1, 1, 1, 1, 1, 1, 1, 10, 300, 16, 128, 1, 1, 10, 1)


def validate_state(state):
    if set(state) != set(FIELDS):
        raise ValueError("Incomplete/unknown policy state")
    for k, v in state.items():
        if type(v) not in (float, int, bool) or not math.isfinite(v):
            raise ValueError("Nonfinite/nonnumeric state")
    for k in set(FIELDS) - {
        "base_accuracy",
        "adapter_gain_pp",
        "headroom_mb",
        "failure_count",
        "verified_examples",
        "regression_pp",
    }:
        if state[k] not in (0, 1):
            raise ValueError("Expected measured binary flag")
    if not 0 <= state["base_accuracy"] <= 1:
        raise ValueError("Invalid accuracy")
    if any(
        state[k] < 0
        for k in ("headroom_mb", "failure_count", "verified_examples", "regression_pp")
    ):
        raise ValueError("Negative measurement")


def oracle(s):
    """Synthetic workflow contract, NOT a learned claim about real LoRA utility."""
    validate_state(s)
    if s["pipeline_fault"]:
        return "repair"
    if not s["verified"]:
        return "verify"
    if s["ephemeral"] and s["memory_available"]:
        return "memory"
    if s["base_accuracy"] >= 0.85:
        return "base"
    if (
        s["adapter_present"]
        and s["compatible"]
        and s["qualified"]
        and s["provenance"]
        and s["adapter_gain_pp"] >= 5
        and s["headroom_mb"] >= 150
        and s["regression_pp"] <= 3
    ):
        return "reuse_adapter"
    if (
        not s["ephemeral"]
        and s["failure_count"] >= 8
        and s["verified_examples"] >= 64
        and s["training_affordable"]
        and s["idle"]
        and s["headroom_mb"] >= 150
    ):
        return "propose_adapter"
    return "verify"


def sequence(prompt, state):
    validate_state(state)
    return features(prompt, [state[k] / scale for k, scale in zip(FIELDS, SCALES)])


def family_splits(wordings):
    """Group near-duplicate wording BEFORE training/splitting; no model scores."""
    stop = set(
        "a an the this that these those to of and in on for with our your my me i please use using into from is are be as by it its at each".split()
    )
    assignment = {}
    for group in ACTIONS:
        values = sorted(
            [w for w in wordings if w["id"].rsplit("-", 1)[0] == group],
            key=lambda w: w["id"],
        )
        tokens = [set(re.findall(r"[a-z]+", w["en"].lower())) - stop for w in values]
        parents = list(range(len(values)))

        def find(i):
            while parents[i] != i:
                i = parents[i]
            return i

        for i in range(len(values)):
            for j in range(i):
                overlap = len(tokens[i] & tokens[j])
                union = len(tokens[i] | tokens[j])
                if overlap >= 4 and (
                    overlap / max(1, union) >= 0.4
                    or overlap / max(1, min(len(tokens[i]), len(tokens[j]))) >= 0.7
                ):
                    parents[find(i)] = find(j)
        clusters = {}
        for i, w in enumerate(values):
            clusters.setdefault(find(i), []).append(w)
        ordered = sorted(
            clusters.values(),
            key=lambda ws: hashlib.sha256(
                " ".join(sorted(w["en"] for w in ws)).encode()
            ).hexdigest(),
        )
        if len(ordered) < 3:
            raise ValueError("Insufficient independent lexical clusters for " + group)
        ntest = max(1, len(ordered) // 4)
        for i, cluster in enumerate(ordered):
            split = "test" if i < ntest else "dev" if i == ntest else "train"
            family = (
                group
                + ":"
                + hashlib.sha256(
                    " ".join(sorted(w["id"] for w in cluster)).encode()
                ).hexdigest()[:12]
            )
            for w in cluster:
                assignment[w["id"]] = (family, split)
    return assignment


def generated_scenarios(wordings):
    """Split whole scenario families, then vary measurements within each family."""
    rows = []
    splits = family_splits(wordings)
    for word in wordings:
        group, index = word["id"].rsplit("-", 1)
        index = int(index)
        if group not in ACTIONS or not 0 <= index < 16:
            raise ValueError("Unexpected family")
        family, split = splits[word["id"]]
        for variant in range(4):
            rng = np.random.default_rng(
                930001 + index * 70 + ACTIONS.index(group) * 3000 + variant
            )
            s = dict(zip(FIELDS, (0, 1, 0, 0, 0.35, 0, 1, 0, 0, 300, 0, 0, 1, 1, 0, 1)))
            s["headroom_mb"] = int(rng.integers(160, 501))
            if group == "base":
                s["base_accuracy"] = float(rng.uniform(0.88, 0.99))
            if group == "memory":
                s.update(ephemeral=1, memory_available=1)
            if group == "repair":
                s.update(pipeline_fault=1, verified=variant % 2)
            if group == "verify":
                if variant == 0:
                    s["verified"] = 0
                elif variant == 1:
                    s.update(failure_count=3, verified_examples=100)
                elif variant == 2:
                    s.update(failure_count=10, verified_examples=18)
                else:
                    s.update(failure_count=10, verified_examples=100, headroom_mb=90)
            if group == "reuse_adapter":
                s.update(
                    adapter_present=1,
                    qualified=1,
                    adapter_gain_pp=int(rng.integers(5, 13)),
                    regression_pp=int(rng.integers(0, 4)),
                )
            if group == "propose_adapter":
                s.update(
                    failure_count=int(rng.integers(8, 17)),
                    verified_examples=int(rng.integers(64, 160)),
                )
            for language in ("vi", "en"):
                rows.append(
                    {
                        "id": word["id"] + f"-{variant}-{language}",
                        "family": family,
                        "split": split,
                        "prompt": word[language],
                        "state": s.copy(),
                        "target": oracle(s),
                        "label_source": "synthetic-lifecycle-contract-v1; not real adapter outcome",
                        "teacher_model": "gemini-3.8-flash-high",
                    }
                )
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate row")
    counterfactuals = []
    for row in rows:
        if (
            row["target"] in ("base", "memory", "reuse_adapter", "propose_adapter")
            and "-0-" in row["id"]
        ):
            for kind, changes in (
                ("fault", {"pipeline_fault": 1}),
                ("unverified", {"verified": 0}),
            ):
                changed = {**row["state"], **changes}
                counterfactuals.append(
                    {
                        **row,
                        "id": row["id"] + "-" + kind,
                        "state": changed,
                        "target": oracle(changed),
                        "counterfactual": True,
                    }
                )
    rows += counterfactuals
    return rows


class LifecycleALIF:
    def __init__(self, seed, mode="learned"):
        self.model = LocalALIF(seed, mode)

    def predict(self, row):
        p = self.model.forward(sequence(row["prompt"], row["state"]))[0]
        return ACTIONS[int(np.argmax(p))], p.tolist()

    def learn(self, row):
        return self.model.learn(
            sequence(row["prompt"], row["state"]),
            ACTIONS.index(row["target"]),
            evidence={
                "kind": "machine_verifier",
                "id": "synthetic-contract:" + row["id"],
            },
            remember=False,
        )

    def dumps(self):
        return self.model.dumps(
            {
                "purpose": "adapter-lifecycle-v1",
                "actions": ACTIONS,
                "schema": FIELDS,
                "not_serving_checkpoint": True,
            }
        )

    @classmethod
    def loads(cls, data):
        model, state = LocalALIF.loads(data)
        if (
            state.get("purpose") != "adapter-lifecycle-v1"
            or state.get("actions") != list(ACTIONS)
            or state.get("schema") != list(FIELDS)
        ):
            raise ValueError("Wrong controller checkpoint purpose")
        obj = cls.__new__(cls)
        obj.model = model
        return obj


def score_rows(model, rows):
    output = []
    for row in rows:
        action, p = model.predict(row)
        output.append(
            {
                "id": row["id"],
                "family": row["family"],
                "target": row["target"],
                "prediction": action,
                "probabilities": p,
                "correct": action == row["target"],
                "unsafe": action in ("reuse_adapter", "propose_adapter")
                and action != row["target"],
            }
        )
    return output
