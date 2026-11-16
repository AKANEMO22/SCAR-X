"""Post-stop stdlib recount of one saved-state shortage; no D0 rerun/model.

Reads immutable raw partitions/TRAIN and computes the ten finite SWAP targets
independently to explain the already-recorded STOP_DATA. No new split/sampler.
"""
from collections import Counter
import hashlib
from itertools import combinations
import json
from pathlib import Path
import time

started = time.perf_counter()
root = Path(__file__).resolve().parent
raw_bytes = (root / "raw-data.json").read_bytes()
result = json.loads((root / "result.json").read_text(encoding="utf-8"))
assert hashlib.sha256(raw_bytes).hexdigest() == result["raw_data_sha256"]
raw = json.loads(raw_bytes)
assert result["status"] == "STOP_DATA" and result["stage"] == "primitive_dev"
state = (0, 0, 0, 2, 3)
train_partition = {tuple(s) for s in raw["partitions"]["train_states"]}
rows = []
for first, second in combinations(range(5), 2):
    target = tuple(state[second] if index == first else state[first] if index == second else value
                   for index, value in enumerate(state))
    changed = target != state
    rows.append({"command": ["SWAP", first, second, -1], "target": target,
                 "changed": changed, "target_in_train_partition": target in train_partition,
                 "valid_changed_command": changed and target not in train_partition})
valid = [row for row in rows if row["valid_changed_command"]]
training = raw["train"]
quotas = Counter((row["command"][0], row["changed"]) for row in training)
counts = {"all_SWAPS": len(rows), "changed_before_target_filter": sum(row["changed"] for row in rows),
          "changed_targets_blocked_by_train_partition": sum(row["changed"] and row["target_in_train_partition"] for row in rows),
          "valid_changed_after_filter": len(valid), "required_changed_SWAPS": 2,
          "TRAIN_examples": len(training), "TRAIN_distinct_prestates": len({tuple(r["state"]) for r in training}),
          "TRAIN_distinct_targets": len({tuple(r["target"]) for r in training}),
          "TRAIN_quotas": {f"{op}:{'changed' if changed else 'noop'}": n for (op, changed), n in quotas.items()}}
assert len(valid) < 2 and counts["TRAIN_examples"] == 512
audit = {"audit_level": "same-team independent stdlib tuple recount of saved data only",
         "source_sha": result["source_sha"], "D0_rerun": False,
         "raw_data_sha256_reverified": result["raw_data_sha256"], "state": state,
         "counts": counts, "all_SWAP_targets": rows, "valid_commands": valid,
         "compact_episodic_payload_bytes_if_stored": len(training) * 14,
         "memory_control_quality": "NOT_RUN: STOP_DATA preceded control evaluation",
         "models": 0, "fits": 0, "updates": 0,
         "elapsed_seconds": time.perf_counter() - started}
output = root / "shortage-audit.json"
with output.open("x", encoding="utf-8") as handle:
    json.dump(audit, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
print(json.dumps({"counts": counts, "valid_commands": valid, "D0_rerun": False,
                  "elapsed_seconds": audit["elapsed_seconds"]}, ensure_ascii=False))
