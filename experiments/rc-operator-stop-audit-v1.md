# RC operator STOP audit v1

2026-11-16, independent implementation by the same Codex team; not external
peer review. Lock this protocol and script before execution. This audit does
not reopen the failed family or authorize dataset/seed/quota changes.

Only source run: `runs/20261115T193825Z-rc-operator-pilot-v1-a76d1c`, source
`6cadbbe2113f94c88b44ecc87678adb5770bb552`. Read allowlisted raw-data.json,
result.json, manifest.json, control-predictions.json and the four source
snapshots named by that manifest. Hash bytes before metrics. No TEST input,
neural imports/forward/fitting, world module import, data generation, RNG,
greedy selection, sequence sampling or rescue. No subprocesses.

Recompute from saved data:

1. All512 TRAIN labels with a direct tuple implementation; command validity,
   record IDs, uniqueness, pre/target TRAIN membership, max2/prestate,
   >=256 distinct prestates, minimum command coverage8 MOVE/16 SWAP,
   quotas224 changed+32 noop per operator, and object/location balance20–30%.
   Match these counts to the saved receipt/balance.
2. Validate the saved512 TRAIN/448 IID_DEV/64 SHIFT partition covers exactly
   the1024 possible valid tuples with no overlap. Compute global-location
   shift orbits from these saved tuples:256 total,240 ordinary,16 SHIFT,
   including15 exactly-four and1 all-five. Verify TRAIN128/IID112 ordinary
   orbits and the registered deterministic SHA ranking of partitions/eval
   subsets. This is validation of saved sets, not regeneration or resampling.
3. At the recorded failing IID state00023, independently enumerate exactly
   ten unordered SWAP pairs. Save target, changed/noop and target-in-TRAIN
   flags. Verify fewer than2 changed SWAPs have target outside TRAIN. Match
   the failure reason/stage/status to the saved result.

PASS means the recorded STOP_DATA is arithmetically certified. It is not a
neural model NO_GO or a statement about ALIF/GRU capacity; all neural results
remain NOT_MEASURED. Zero model/fitting counts are recorded claims consistent
with the D0 source scope, not an audit of every process on the machine.

One stdlib-only CPU process, no children, BLAS/OpenMP env1, <=10s active wall,
Windows OS peak working set<100,000,000 bytes. Check time/peak RSS during every
TRAIN record and orbit audit; preserve failures without relaxed reruns.
Create an exclusive output directory, registration with input/source/protocol/
config hashes before metrics, then atomic audit/resource/terminal JSON.
The launcher supplies the locked audit commit SHA; it is recorded as a
launcher assertion, with executable source bytes separately hashed. Resource
timing includes audit/output work through final receipt preparation; launcher
wall includes interpreter startup and terminal serialization.
