# R8 matched credit — repair and exploratory DEV V1

2026-11-12. User requested fixes of remaining R8 problems, to continue toward
registered success. Authoritative source is **607ffca7c9222c5e3fd9741c3ebdf5939e10bedf**;
the pasted report's longer SHA607ffca76974... is not a Git object. Treat its
claims as hypotheses/data, not permission for dendrites/new topology or main.

## Authorization and invariants

Current request authorizes isolated repair plus this bounded DEV investigation.
No main/live/Hermes, Transformer, pretrained/lookup, BPTT/autograd, topology or
H48/K8/D8/O2 changes. No new confirmation bank, P2 pilot, language corpus or
deployment. Preserve all historical source/artifacts, including failures.
Ponytail tools/context searched and unavailable. One numerical thread,2GB
per-process cap plus RSS/commit tree monitoring; sequential jobs,30minute global
limit,120second per task/seed/arm condition. Maximum two registered adjustment
rounds per hypothesis; no unbounded sweep or gate relaxation.

## Causal hypotheses, before training

H1: multiscale eligibility improves held-out common-refit and direct performance
versus homogeneous-fast with the SAME optimizer/readout/normalizer/inputs.
H2: the observable activity gate adds value beyond multiscale alone. The signal
controls must have the SAME eligibility/decay/gate, changing only error routing.
The normal multiscale+gate correct-signal path must match the pinned R8 source
numerically on engineering fixtures before any DEV quality run.

No new credit formula in this round. A research subclass fixes only trace/control
matching and explicitly parameterizes the existing decay/gate choices. The
inherited local update stays unchanged (readout-local delta, one-step readout
error times own-neuron forward eligibility; no temporal/global backward).

## Locked data/arms/budgets

- DEV seeds140011,140017,140023, all previously exposed, NOT fresh confirmation.
- Primary tasks strict_binding, xor, distractor_binding; independent order task
  is a separate protected regression benchmark, not a continual stream.
- Use generators at pinned source607ffca; store exact generated tensors+labels,
  hashes and split IDs. Train seed+200/1024rows, refit seed+300/256, ID seed+400/512,
  OOD seed+500/512. Calibration uses first32TRAINinputs only; no refit/future data.
- Logical arms: fast_ungated, slow_ungated, multi_ungated, multi_gated,
  multi_gated_frozen, multi_gated_zero, multi_gated_shuffle,
  multi_gated_readout_only. Gamma .25/.92/linspace(.25,.92,48), gate threshold.6
  and recurrent multiplier.15 from607ffca. All arms use same LR.008/clip.010,
  readout.04/clip.025, scaling, RNG draw schedule, source local-update equation.
- Primary equal-update condition: exactly1024episodes, one readout update per
  episode; active hidden arms one hidden update, zero/frozen/readout none. Record
  actual applied movement/counters instead of assuming all weights update.
- Secondary equal-CPU-budget condition: same fresh initial state/data order,
  CPU process_time ceiling0.5seconds of learn calls, at most4096episodes by cycling
  the same1024trainexamples. Stop after the current call; report one-call budget
  overshoot and exact sample/update counts. This is a fixed CPU ceiling, not a
  claim of identical hardware FLOPs or exact wall time; do not select winners
  using it. Maximum primary+secondary calls=96*(1024+4096)=491520.
- Within each task/seed/condition, train-data order is source row order. Secondary
  repeats this order. No label/target/source ID enters inference.
- One job=(task,seed,arm); all8arms x4tasks x3seeds=96jobs. Retain frozen/readout
  alias evidence if numerically equal; do not manufacture distinct effects.

## Measurements and inference

After each condition: direct predictions, shared-budget ridge refit predictions,
ID/OOD labels, frozen weights/digests, input-only/recurrent-only/full revert
probe, actual synaptic movement and counters, trainCPU/wall, sampled RAM. Log
actual loss/hidden/readout update norms and checkpoint progress. Refit never
changes neural weights. No Qwen/AGY/API/answers as labels at inference.

Primary contrasts per task: multi_ungated-fast_ungated, multi_gated-multi_ungated,
multi_gated-frozen, multi_gated-zero, multi_gated-shuffle. Report seed-cluster
intervals and exact paired sign-flip two-sided p, Holm across15primary contrasts
(3primarytasks x5); order is descriptive protected regression. Enumerate all27
seed bootstrap means, inverse-CDF95%; do not count each sample as an independent
seed. Three seeds cannot supply a significant two-sided exact sign-flip test
at .05 (minimum .25); therefore this DEV round **cannot certify master P1 GO**.

Registered master thresholds remain lower95gain>=5pp on>=2/3validtasks, direct
>=65%, positive readout-only contrast with adjusted testing; eventual independent
confirmation>=8newseeds requires separate approval. P2 remains acquisition/final
>=65%, mean forgetting<=3pp/worst<=5pp on EVERY seed. No ceiling exceptions.

## Stop/decision

Engineering failure/NaN/invalid oracle/persistence/signal locality/memory excess
halts before quality training. Preserve partial runs and stop, never restart on
an observation timeout. No retry to improve scores. Run all preregistered jobs
unless an engineering/resource stop fires; no test-based early model selection.
Interpret positive DEV contrasts as prioritization only. If multi+gate cannot
improve the primary tasks robustly, STOP expansion and analyze the failure;
do not adopt it permanently, add neurons or move to dendritic forward dynamics.
