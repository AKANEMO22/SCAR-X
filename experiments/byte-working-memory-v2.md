# Byte decoder second/final adjustment: held neural working state

Registered2026-11-09 after byte-v1 and decoder-state audit, before this trial.
Master working-memory/P1/P2 scope; no third adjustment to stopped hidden-target
family, larger population, external RAG/LLM/framework or serving migration.

Audit found order-distinct encoding-end activity, but every tested model's
fast features became exactly equal after127identical forcedbytes;69–94% of
trained pair features already identical bybyte32. No task-success conclusion
from this alone. V2 explicitly keeps a snapshot of the SAME48neurons' feature
trace at the end of input encoding for the duration of ONE response. This is
a per-neuron held activation buffer, not stable synapse/learned long-term bank,
semantic memory, extra recurrent core or information acquired from a target.
Release snapshot after response/recompute from next input; it does not grow
with total lifetime. Snapshot can hold noisy/unhelpful features; still test.

Readout presynaptic vector = concat(fast_features,held_features),96numbers.
Outputdim268,256bytealphabet/hostUTF8mask,454inputchannels andH48,K8,R12 unchanged.
New slow-readout weights initialized0, fast weights/bias same seed asbyte-v1.
Extra12864outputweights and48activationvalues are explicitly counted; no claim
identical parameter budget to historical byte-v1. Same96feature head allocation
and zero-initialized secondhalf in each within-v2control:
- no_snapshot: secondhalf always0 (effective rank/capacity lower; disclose).
- snapshot_raw: true encoding-end held features, ordinary raw head.
- snapshot_normalized: true held features with train-only mean/std.

No hidden learning in ANYcondition: isolate the finite-working-memory path from
representation learning. Existing output-local delta only. No bypass from raw
text/label to output, answer lookup, hand-written response, errorbackward/BPTT.
All snapshots are observations before first decoderbyte, no futuretarget.

Same immutable96train/24development curatedAntigravity responses. Fresh seeds
9103,9113,9133. Calibration train-only teacher-forced exactly asbyte-v1,4passes,
fixed outputlr0.08/clip0.025/clipweights3, chronological input, state persists.
Initial/after2/after4metrics and max128outputbytes, same teacherforced bytebudget.
Inference ablations on final checkpoints: held_features zeroed or swapped with
next example's encoding-end snapshot via a frozen diagnostic clone. The swapped
snapshot must come from input alone, no target. Report all outputs/byteaccuracy/
NLL/exact; do not select a winner using ablation. Labels/OOV0same asv1.

Screen useful only when gainvsno_snapshot>=10ppdevutteranceexact on average AND
held-memory removal reduces that gain, plus no incoherent-output evidence.
This is not historical representation/retention gate; even a screenpass requires
8freshseed confirmation, lifelong retention and vision evaluation before deploy.
If no useful gain after2and4, STOPthisbyte-dialogue family at secondround; no
more passes, broaderdata, parameter search or claimed success from validUTF8.

Percondition180s,total1200sCPU1thread,2GBcap; record actual extraarrays/updates/
ticks/elapsed/RAM. All checkpoints/heldactivation/statistics/orderRNG/counters
resume and coldprocessoutputs match, own format rejectsG1. No privateweights
modified. NormalphotoQA is NOT trained/proven; RGBpath remains prototype.
