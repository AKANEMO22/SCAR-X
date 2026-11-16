"""RC-OPERATOR v1 pure data/oracle/codec/episodic controls; no neural imports.

The oracle is exposed only for data generation/control. State feedback has no
operator argument and cannot perform a world transition. Fixed NumPy RNG is
used only because the registered protocol specifies SeedSequence.
"""
from collections import Counter, defaultdict
import hashlib
from itertools import combinations, product
import json

import numpy as np

DATA_SEED = 261180
MODEL_SEEDS = (261181, 261187, 261191)
MOVE = tuple(("MOVE", i, -1, l) for i in range(5) for l in range(4))
SWAP = tuple(("SWAP", i, j, -1) for i, j in combinations(range(5), 2))
COMMANDS = MOVE + SWAP
QUOTAS = {("MOVE", True): 224, ("MOVE", False): 32,
          ("SWAP", True): 224, ("SWAP", False): 32}


class DataStop(RuntimeError):
    def __init__(self, message, evidence=None):
        super().__init__(message)
        self.evidence = evidence or {}


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical_json(value)).hexdigest()


def state_id(state):
    return "".join(map(str, state))


def command_id(command):
    return ":".join(map(str, command))


def rank(label, value):
    # partition payload is exactly the five-digit canonical state required by
    # the protocol. Other payloads use this same explicit immutable convention.
    return hashlib.sha256(f"rcop-v1|{DATA_SEED}|{label}|{value}".encode()).hexdigest()


def checked_state(state):
    result = tuple(state)
    if len(result) != 5 or any(type(v) is not int or not 0 <= v < 4 for v in result):
        raise ValueError("Five categorical location integers required")
    return result


def checked_command(command):
    result = tuple(command)
    if result not in COMMANDS:
        raise ValueError("Unregistered primitive command")
    return result


def oracle(state, command):
    state, command = checked_state(state), checked_command(command)
    op, first, second, destination = command
    result = list(state)
    if op == "MOVE":
        result[first] = destination
    else:
        result[first], result[second] = result[second], result[first]
    return tuple(result)


def encode_state(state):
    state = checked_state(state)
    return tuple(float(value == state[o]) for o in range(5) for value in range(4))


def decode_state(logits):
    if len(logits) != 20 or not all(np.isfinite(v) for v in logits):
        raise ValueError("Twenty finite logits required")
    return tuple(max(range(4), key=lambda l: float(logits[4 * o + l])) for o in range(5))


def identity_feedback(logits):
    """No command/oracle argument, copy mask, residual, edit or transition."""
    return encode_state(decode_state(logits))


def encode_input(state, command):
    op, first, second, destination = checked_command(command)
    vector = list(encode_state(state)) + [float(op == "MOVE"), float(op == "SWAP")]
    vector += [float(first == o) for o in range(5)]
    vector += [float(second == o) for o in range(5)]
    vector += [float(destination == l) for l in range(4)]
    return tuple(vector + [0.0] * 28)


def occupancy(state):
    return max(Counter(state).values())


def orbit_key(state):
    return min(tuple((v + k) % 4 for v in state) for k in range(4))


def orbit_states(key):
    return tuple(sorted(tuple((v + k) % 4 for v in key) for k in range(4)))


def record(state, command, stratum="TRAIN", noop=False):
    target = oracle(state, command)
    payload = f"{state_id(state)}|{command_id(command)}|{state_id(target)}"
    return {"id": rank("record-id", payload), "state": tuple(state),
            "command": tuple(command), "target": target, "stratum": stratum,
            "changed": target != tuple(state), "noop_diagnostic": noop,
            "_rank": rank("train-record", payload)}


def make_partitions(check):
    states = tuple(product(range(4), repeat=5))
    keys = sorted({orbit_key(s) for s in states})
    ordinary = [k for k in keys if occupancy(k) <= 3]
    shift = [k for k in keys if occupancy(k) >= 4]
    four = [k for k in shift if occupancy(k) == 4]
    ordinary.sort(key=lambda k: rank("partition", state_id(k)))
    train = sorted(s for key in ordinary[:128] for s in orbit_states(key))
    remaining = ordinary[128:]
    iid_keys = sorted(remaining, key=lambda k: rank("iid-eval", state_id(k)))[:16]
    shift_keys = sorted(four, key=lambda k: rank("shift-eval", state_id(k)))[:8]
    result = {"train_states": train,
              "iid_dev_states": sorted(s for k in remaining for s in orbit_states(k)),
              "shift_states": sorted(s for k in shift for s in orbit_states(k)),
              "iid_eval_states": sorted(s for k in iid_keys for s in orbit_states(k)),
              "shift_eval_states": sorted(s for k in shift_keys for s in orbit_states(k)),
              "all_five_unmeasured": sorted(s for s in states if occupancy(s) == 5)}
    counts = (len(states), len(keys), len(ordinary), len(shift), len(four),
              len(train), len(result["iid_dev_states"]), len(result["shift_states"]))
    if counts != (1024, 256, 240, 16, 15, 512, 448, 64):
        raise DataStop(f"Partition arithmetic mismatch: {counts}")
    check()
    return result


def make_train(partitions, check):
    train_set = set(partitions["train_states"])
    pool = []
    for index, state in enumerate(partitions["train_states"]):
        if index % 32 == 0:
            check()
        for command in COMMANDS:
            item = record(state, command)
            if item["target"] in train_set:
                pool.append(item)
    pool.sort(key=lambda item: (item["_rank"], item["id"]))
    chosen, ids, state_counts, command_counts, quota_counts = [], set(), Counter(), Counter(), Counter()

    def take(item):
        command, state = item["command"], item["state"]
        quota = command[0], item["changed"]
        if (item["id"] in ids or state_counts[state] >= 2
                or quota_counts[quota] >= QUOTAS[quota]):
            return False
        chosen.append(item)
        ids.add(item["id"])
        state_counts[state] += 1
        command_counts[command] += 1
        quota_counts[quota] += 1
        return True

    for command in COMMANDS:  # MOVE lexicographic, then SWAP lexicographic.
        minimum = 8 if command[0] == "MOVE" else 16
        for item in pool:
            if command_counts[command] >= minimum:
                break
            if item["command"] == command:
                take(item)
        check()
        if command_counts[command] < minimum:
            raise DataStop(f"Greedy command coverage failure: {command_id(command)}",
                           {"partial_train": chosen, "failing_command": command,
                            "pool_records": len(pool)})
    for item in pool:
        if len(chosen) == 512:
            break
        take(item)
    receipt = {"pool_records": len(pool), "selected_records": len(chosen),
               "distinct_prestates": len(state_counts),
               "max_records_per_prestate": max(state_counts.values(), default=0),
               "command_counts": {command_id(c): command_counts[c] for c in COMMANDS},
               "quota_counts": {f"{op}:{'changed' if changed else 'noop'}": quota_counts[(op, changed)]
                                for op, changed in QUOTAS}}
    if len(chosen) != 512 or dict(quota_counts) != QUOTAS or len(state_counts) < 256:
        raise DataStop("Greedy TRAIN quota/coverage failure", {"train_receipt": receipt, "partial_train": chosen})
    for item in chosen:
        item.pop("_rank", None)
    check()
    return chosen, receipt


def make_primitive_dev(partitions, check):
    train_set = set(partitions["train_states"])
    result = []
    for stratum, states in (("IID", partitions["iid_eval_states"]),
                            ("SHIFT", partitions["shift_eval_states"])):
        for state in states:
            for commands in (MOVE, SWAP):
                for changed, count, label in ((True, 2, "primitive-dev"), (False, 1, "noop-dev")):
                    pool = [record(state, c, stratum, noop=not changed) for c in commands
                            if (oracle(state, c) != state) == changed
                            and oracle(state, c) not in train_set]
                    pool.sort(key=lambda item: rank(label, f"{state_id(state)}|{command_id(item['command'])}"))
                    if len(pool) < count:
                        raise DataStop(f"DEV valid-command shortage: {stratum}/{state}/{commands[0][0]}")
                    for item in pool[:count]:
                        item.pop("_rank", None)
                        result.append(item)
            check()
    if len(result) != 576:
        raise DataStop("Primitive DEV count mismatch")
    covered = {r["command"] for r in result if r["changed"]}
    if covered != set(COMMANDS):
        raise DataStop("Global DEV argument coverage failure", {"primitive_partial": result})
    return result


def oracle_path(state, commands):
    path = [tuple(state)]
    for command in commands:
        path.append(oracle(path[-1], command))
    return tuple(path)


def path_valid(path, train_set):
    return (all(path[k] != path[k - 1] for k in range(1, len(path)))
            and all(s not in train_set for s in path)
            and path[-1] != path[0]
            and sum(a != b for a, b in zip(path[-1], path[0])) >= 2)


def make_sequences(partitions, check):
    train_set, result = set(partitions["train_states"]), []
    initial = []
    for stratum, key, count in (("IID", "iid_eval_states", 32), ("SHIFT", "shift_eval_states", 16)):
        pool = sorted(partitions[key], key=lambda s: rank("sequence-initial", state_id(s)))[:count]
        initial.extend((stratum, s) for s in pool)
    attempts_receipt = []
    for index, (stratum, state) in enumerate(initial):
        for family_index in range(3):
            rng = np.random.default_rng(np.random.SeedSequence([DATA_SEED, 6116, index, family_index]))
            accepted = None
            for attempt in range(1, 2001):
                if attempt % 16 == 0:
                    check()
                if family_index == 0:
                    a, b = MOVE[int(rng.integers(len(MOVE)))], SWAP[int(rng.integers(len(SWAP)))]
                    if a[1] not in b[1:3] or a[3] in (state[b[1]], state[b[2]]):
                        continue
                    candidate = (("AB", (a, b)), ("BA", (b, a)))
                else:
                    pattern = "ABAB" if family_index == 1 else "BABA"
                    commands = tuple((MOVE if op == "A" else SWAP)[int(rng.integers(20 if op == "A" else 10))]
                                     for op in pattern)
                    # Later commands must be fresh, rather than merely replaying
                    # the identical pair twice and obtaining easy cancellations.
                    if commands[0] == commands[2] or commands[1] == commands[3]:
                        continue
                    candidate = ((pattern, commands),)
                paths = [oracle_path(state, commands) for _, commands in candidate]
                if not all(path_valid(path, train_set) for path in paths):
                    continue
                if family_index == 0 and paths[0][-1] == paths[1][-1]:
                    continue
                accepted = candidate, paths
                break
            if accepted is None:
                raise DataStop(f"Sequence generation exhausted: initial={index}, family={family_index}, cap=2000",
                               {"sequences_partial": result, "attempts": attempts_receipt,
                                "failed_initial_index": index, "failed_family_index": family_index})
            attempts_receipt.append({"initial_index": index, "family_index": family_index, "attempts": attempt})
            candidate, paths = accepted
            for (family, commands), path in zip(candidate, paths):
                result.append({"id": f"{stratum}-{state_id(state)}-{family}", "stratum": stratum,
                               "family": family, "initial_state": state, "commands": commands,
                               "oracle_path": path, "initial_index": index,
                               "prefix_in_train": [s in train_set for s in path],
                               "prefix_occupancy": [occupancy(s) for s in path]})
        check()
    if len(result) != 192 or Counter(r["family"] for r in result) != dict.fromkeys(("AB", "BA", "ABAB", "BABA"), 48):
        raise DataStop("Sequence counts mismatch")
    coverage = {c for item in result for c in item["commands"]}
    if coverage != set(COMMANDS):
        raise DataStop("Global sequence argument coverage failure", {"sequences_partial": result})
    return result, attempts_receipt


def per_slot_majority(states):
    return tuple(max(range(4), key=lambda l: sum(s[o] == l for s in states)) for o in range(5))


class EpisodicControls:
    """Counts/retrieval only, no optimized coefficients or semantic edit rule."""
    names = ("identity", "last_destination_constant", "operator_only_majority",
             "argument_only_majority", "state_only_identity", "exact_lookup", "one_nn", "knn3")

    def __init__(self, training):
        self.by_command = defaultdict(list)
        self.exact = {}
        self.query_counts = Counter()
        self.distance_evaluations = Counter()
        by_op = defaultdict(list)
        for item in training:
            self.by_command[item["command"]].append(item)
            self.exact[(item["state"], item["command"])] = item["target"]
            by_op[item["command"][0]].append(item["target"])
        self.op_majority = {op: per_slot_majority(states) for op, states in by_op.items()}
        self.arg_majority = {c: per_slot_majority([r["target"] for r in items]) for c, items in self.by_command.items()}

    def predict(self, name, state, command):
        if state is None:
            return None
        if name in ("identity", "state_only_identity"):
            return tuple(state)
        if name == "last_destination_constant":
            return (command[3],) * 5 if command[0] == "MOVE" else (0,) * 5
        if name == "operator_only_majority":
            return self.op_majority[command[0]]
        if name == "argument_only_majority":
            return self.arg_majority[command]
        if name == "exact_lookup":
            return self.exact.get((tuple(state), command))
        if name not in ("one_nn", "knn3"):
            raise ValueError("Unknown episodic control")
        self.query_counts[name] += 1
        self.distance_evaluations[name] += len(self.by_command[command])
        items = sorted(self.by_command[command], key=lambda r: (
            sum(a != b for a, b in zip(state, r["state"])), r["id"]))
        if name == "one_nn":
            return items[0]["target"]
        return per_slot_majority([r["target"] for r in items[:3]])
